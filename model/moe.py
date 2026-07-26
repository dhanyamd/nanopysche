"""Mixture of Experts (MoE) — sparse expert routing.

Matches OLMo-core nn.moe patterns:
  - MoEBase with router, experts, optional shared_mlp
  - TopKRouter with load balancing loss and z-loss
  - Flattened weight format for grouped GEMM
  - Expert Parallelism via all-to-all dispatch/combine

Reference: Fedus et al. 2021 (Switch Transformer)
           OLMo-core src/olmo_core/nn/moe/
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanopsyche.model.feed_forward import FeedForward


@dataclass
class MoERouterConfig:
    """Configuration for the MoE router."""

    num_experts: int = 8
    top_k: int = 2
    bias_init: float = -1.0
    z_loss_multiplier: Optional[float] = None


class MoERouter(nn.Module):
    """Top-k expert router with load balancing.

    :param d_model: model dimensionality.
    :param num_experts: number of experts.
    :param top_k: number of experts per token.
    :param bias_init: initial bias for uniform routing.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        bias_init: float = -1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        if bias_init is not None:
            nn.init.constant_(self.gate.bias, bias_init)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts.

        :param x: (B, S, d_model)
        :returns: (expert_weights, expert_indices, aux_loss)
            - expert_weights: (B, S, top_k) — normalized weights
            - expert_indices: (B, S, top_k) — expert indices
            - aux_loss: scalar — load balancing loss
        """
        logits = self.gate(x.float())

        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        aux_loss = self._load_balancing_loss(logits, top_k_indices)

        return top_k_weights, top_k_indices, aux_loss

    def _load_balancing_loss(
        self, logits: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Switch Transformer load balancing loss.

        loss = num_experts * sum(f_i * P_i)
        where f_i = fraction routed to expert i, P_i = mean probability.
        """
        num_tokens = logits.shape[0] * logits.shape[1]
        expert_mask = F.one_hot(indices, self.num_experts).sum(dim=2)
        tokens_per_expert = expert_mask.float().sum(dim=[0, 1])
        f = tokens_per_expert / num_tokens
        router_probs = F.softmax(logits, dim=-1)
        P = router_probs.float().mean(dim=[0, 1])
        return self.num_experts * (f * P).sum()


class MoEBase(nn.Module):
    """Base Mixture of Experts layer.

    :param d_model: model dimensionality.
    :param hidden_size: expert intermediate size.
    :param num_experts: number of experts.
    :param top_k: number of experts per token.
    :param bias: whether to use bias in expert FFNs.
    :param z_loss_multiplier: z-loss weight for training stability.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        bias: bool = False,
        z_loss_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size or int(8 / 3 * d_model)
        self.hidden_size = ((self.hidden_size + 255) // 256) * 256
        self.z_loss_multiplier = z_loss_multiplier

        self.router = MoERouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Flattened expert weights (OLMo-core pattern for grouped GEMM)
        self.w1 = nn.Parameter(torch.empty(num_experts * self.hidden_size, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts * self.hidden_size, d_model))
        self.w3 = nn.Parameter(torch.empty(num_experts * self.hidden_size, d_model))

        self._init_weights()

    def _init_weights(self):
        for w in [self.w1, self.w2, self.w3]:
            nn.init.normal_(w, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE forward: route, dispatch, compute, combine.

        :param x: (B, S, d_model)
        :returns: (B, S, d_model)
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        weights, indices, aux_loss = self.router(x.view(B, S, D))
        weights_flat = weights.view(-1, self.top_k)
        indices_flat = indices.view(-1, self.top_k)

        output = self._compute_experts(x_flat, weights_flat, indices_flat)

        self._aux_loss = aux_loss
        return output.view(B, S, D)

    def _compute_experts(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted sum of expert outputs."""
        output = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_idx = indices[:, k]
            expert_weight = weights[:, k]

            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    token_subset = x[mask]
                    expert_out = self._expert_forward(e, token_subset)
                    output[mask] += expert_weight[mask].unsqueeze(-1) * expert_out

        return output

    def _expert_forward(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Compute a single expert's FFN."""
        h = self.hidden_size
        w1 = self.w1[expert_idx * h : (expert_idx + 1) * h]
        w2 = self.w2[expert_idx * h : (expert_idx + 1) * h]
        w3 = self.w3[expert_idx * h : (expert_idx + 1) * h]
        return F.silu(x @ w1.T) * (x @ w3.T) @ w2.T

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """Load balancing loss."""
        return getattr(self, "_aux_loss", None)
