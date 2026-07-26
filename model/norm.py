"""RMSNorm — Root Mean Square Layer Normalization.

Matches OLMo-core nn.layer_norm.RMSNorm pattern:
  - full_precision flag (compute in FP32 for training stability)
  - weight.type_as(x) for automatic dtype matching
  - No bias

Reference: Zhang & Sennrich 2019
           OLMo-core src/olmo_core/nn/layer_norm.py
"""

from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm with optional FP32 computation.

    :param size: the normalized size (d_model).
    :param eps: epsilon for numerical stability.
    :param full_precision: always compute in FP32 for training stability.
    """

    def __init__(self, size: int, *, eps: float = 1e-6, full_precision: bool = True):
        super().__init__()
        self.eps = eps
        self.full_precision = full_precision
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the last dimension.

        :param x: (..., size) — any leading batch/sequence dimensions.
        :returns: same shape, normalized, same dtype as input.
        """
        if self.full_precision and x.dtype != torch.float32:
            x = x.float()

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)

        out = x_normed * self.weight.type_as(x_normed)

        if self.full_precision and out.dtype != x.dtype:
            out = out.to(x.dtype)

        return out
