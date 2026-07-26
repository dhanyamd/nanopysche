"""FeedForward — SwiGLU / GeGLU gated feed-forward networks.

Matches OLMo-core nn.feed_forward.FeedForward pattern:
  - w1 (gate), w2 (down), w3 (up) naming
  - apply_tp() shards weights across TP ranks, forward uses local compute + all-reduce
  - Configurable activation function
  - Optional bias

TP approach (Megatron-LM style):
  - w1, w3: Colwise (split hidden dim across TP ranks)
  - w2: Rowwise (split input dim, all-reduce after)

Reference: Shazeer 2020 (arXiv:2002.05202)
           OLMo-core src/olmo_core/nn/feed_forward.py
"""

from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh


class ActivationFunction(Enum):
    """Supported activation functions for gated FFNs."""

    silu = "silu"
    gelu_tanh = "gelu_tanh"


class FeedForward(nn.Module):
    """SwiGLU/GeGLU feed-forward network.

    Architecture: w2(activation(w1(x)) * w3(x))

    :param d_model: model dimensionality.
    :param hidden_size: intermediate size. None = 8/3 * d_model rounded to 256.
    :param bias: whether to use bias in linear layers.
    :param activation: activation function ("silu" for SwiGLU, "gelu_tanh" for GeGLU).
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: Optional[int] = None,
        bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = int(8 / 3 * d_model)
            hidden_size = ((hidden_size + 255) // 256) * 256

        self.w1 = nn.Linear(d_model, hidden_size, bias=bias)
        self.w2 = nn.Linear(hidden_size, d_model, bias=bias)
        self.w3 = nn.Linear(d_model, hidden_size, bias=bias)

        if activation == "silu":
            self._activation_fn = F.silu
        elif activation == "gelu_tanh":
            self._activation_fn = lambda x: F.gelu(x, approximate="tanh")
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._tp_size = 1
        self._tp_group = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.w2(self._activation_fn(self.w1(x)) * self.w3(x))

        if self._tp_size > 1:
            dist.all_reduce(out, group=self._tp_group)

        return out

    def apply_tp(self, tp_mesh: DeviceMesh):
        """Apply Megatron-style tensor parallelism.

        w1, w3: Colwise (split hidden_size across TP ranks).
        w2: Rowwise (split input dim, which is hidden_size, across TP ranks).

        :param tp_mesh: tensor parallel DeviceMesh.
        """
        tp_size = tp_mesh.size()
        tp_group = tp_mesh.get_group()
        self._tp_size = tp_size
        self._tp_group = tp_group

        rank = dist.get_rank(tp_group)

        # w1 (gate): colwise — split output dim
        w1_weight = self.w1.weight.view(tp_size, -1, self.w1.weight.shape[-1])
        self.w1.weight = nn.Parameter(w1_weight[rank].contiguous())
        if self.w1.bias is not None:
            w1_bias = self.w1.bias.view(tp_size, -1)
            self.w1.bias = nn.Parameter(w1_bias[rank].contiguous())

        # w3 (up): colwise — split output dim
        w3_weight = self.w3.weight.view(tp_size, -1, self.w3.weight.shape[-1])
        self.w3.weight = nn.Parameter(w3_weight[rank].contiguous())
        if self.w3.bias is not None:
            w3_bias = self.w3.bias.view(tp_size, -1)
            self.w3.bias = nn.Parameter(w3_bias[rank].contiguous())

        # w2 (down): rowwise — split input dim
        w2_weight = self.w2.weight
        hidden_local = w2_weight.shape[1] // tp_size
        self.w2.weight = nn.Parameter(
            w2_weight[:, rank * hidden_local : (rank + 1) * hidden_local].contiguous()
        )
