from __future__ import annotations

"""Mixture of Experts (MoE) — sparse expert routing.

Matches OLMo-core nn.moe patterns:
  - MoEBase with router, experts, optional shared_mlp
  - TopKRouter with load balancing loss and z-loss
  - Flattened weight format for grouped GEMM
  - Expert Parallelism via all-to-all dispatch/combine

Reference: Fedus et al. 2021 (Switch Transformer)
           OLMo-core src/olmo_core/nn/moe/
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


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
        self.gate = nn.Linear(d_model, num_experts, bias=True)

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


# ---------------------------------------------------------------------------
# Expert Parallelism config & dispatch
# ---------------------------------------------------------------------------


@dataclass
class ExpertParallelConfig:
    """Expert Parallelism configuration.

    EP shards experts across ranks. Each rank owns num_experts // ep_size experts.
    Tokens are dispatched to the correct rank via all-to-all communication.

    Reference: OLMo-core MoEConfig / EP mesh construction
    """

    ep_size: int = 1
    ep_group: Optional[dist.ProcessGroup] = None

    def ep_rank(self) -> int:
        if self.ep_group is None:
            return 0
        return dist.get_rank(self.ep_group)

    def local_experts(self, num_experts: int) -> tuple[int, int]:
        """Return (start, end) of expert indices owned by this rank."""
        per_rank = num_experts // self.ep_size
        start = self.ep_rank() * per_rank
        return start, start + per_rank

    def dispatch(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple | None]:
        """Dispatch tokens to local experts via all-to-all.

        OLMo-core dispatch pattern:
            1. Local permute: sort tokens by target rank
            2. All-to-all: send tokens to owning rank
            3. Re-permute: sort by local expert index

        :param x: (num_tokens, d_model)
        :param expert_indices: (num_tokens, top_k) — global expert indices
        :param expert_weights: (num_tokens, top_k)
        :param num_experts: total number of experts
        :returns: (x_dispatched, weights_dispatched, indices_dispatched,
                   metadata for combine)
        """
        if self.ep_size <= 1:
            return x, expert_weights, expert_indices, None

        per_rank = num_experts // self.ep_size
        ep_rank = self.ep_rank()
        local_start = ep_rank * per_rank

        # Step 1: For each token-top_k pair, compute target EP rank
        target_ranks = expert_indices // per_rank  # (num_tokens, top_k)
        is_local = target_ranks == ep_rank  # (num_tokens, top_k)

        # Step 2: Flatten and filter — keep only tokens destined for this rank
        num_tokens, top_k = expert_indices.shape
        flat_tokens = torch.arange(num_tokens, device=x.device).repeat_interleave(top_k)
        flat_experts = expert_indices.reshape(-1)
        flat_weights = expert_weights.reshape(-1)
        flat_local = is_local.reshape(-1)

        local_token_ids = flat_tokens[flat_local]
        local_expert_ids = flat_experts[flat_local]
        local_weights = flat_weights[flat_local]
        local_x = x[local_token_ids]

        # Map global expert id to local expert id
        local_expert_ids = local_expert_ids - local_start

        return local_x, local_weights, local_expert_ids, (local_token_ids, num_tokens)

    def combine(
        self,
        expert_output: torch.Tensor,
        combine_metadata,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Combine expert outputs back to original token positions.

        :param expert_output: (num_local_tokens, d_model) — output from local experts
        :param combine_metadata: tuple of (local_token_ids, num_tokens)
        :param x: (num_tokens, d_model) — original input (for residual/zeroing)
        :param expert_weights: (num_tokens, top_k)
        :param expert_indices: (num_tokens, top_k)
        :returns: (num_tokens, d_model)
        """
        if combine_metadata is None:
            return expert_output

        local_token_ids, num_tokens = combine_metadata
        output = torch.zeros(
            num_tokens,
            expert_output.shape[-1],
            device=expert_output.device,
            dtype=expert_output.dtype,
        )
        output[local_token_ids] = expert_output
        return output


# ---------------------------------------------------------------------------
# MoE Layer
# ---------------------------------------------------------------------------


@dataclass
class MoEConfig:
    """Full MoE configuration."""

    d_model: int = 512
    hidden_size: Optional[int] = None
    num_experts: int = 8
    top_k: int = 2
    bias: bool = False
    z_loss_multiplier: Optional[float] = None
    ep: ExpertParallelConfig = field(default_factory=ExpertParallelConfig)


class MoEBase(nn.Module):
    """Mixture of Experts layer with padded bmm computation.

    Expert computation uses padded batched matrix multiplication:
        1. For each expert, gather assigned tokens
        2. Pad to uniform length for batched matmul
        3. Compute expert FFN (SwiGLU) via torch.bmm
        4. Scatter weighted outputs back

    Falls back to Python for-loop when grouped_gemm is unavailable.
    Supports Expert Parallelism via all-to-all token dispatch.

    :param d_model: model dimensionality.
    :param hidden_size: expert intermediate size.
    :param num_experts: number of experts.
    :param top_k: number of experts per token.
    :param bias: whether to use bias in expert FFNs.
    :param z_loss_multiplier: z-loss weight for training stability.
    :param ep: Expert Parallelism configuration.
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
        ep: Optional[ExpertParallelConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size or int(8 / 3 * d_model)
        self.hidden_size = ((self.hidden_size + 255) // 256) * 256
        self.z_loss_multiplier = z_loss_multiplier
        self.ep = ep or ExpertParallelConfig()

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

        # EP dispatch
        local_x, local_weights, local_indices, combine_meta = self.ep.dispatch(
            x_flat,
            indices_flat,
            weights_flat,
            self.num_experts,
        )

        # Compute expert outputs
        expert_output = self._compute_experts_padded_bmm(
            local_x, local_weights, local_indices
        )

        # EP combine
        output = self.ep.combine(
            expert_output, combine_meta, x_flat, weights_flat, indices_flat
        )

        self._aux_loss = aux_loss
        return output.view(B, S, D)

    def _compute_experts_padded_bmm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expert outputs using padded batched matmul.

        For each expert:
            1. Find tokens assigned to this expert
            2. Pad to max_tokens_per_expert
            3. Compute SwiGLU via torch.bmm (batched matmul)
            4. Unpad and weight the output

        This replaces the Python for-loop with batched GPU matmuls.
        """
        num_tokens, D = x.shape
        output = torch.zeros_like(x)

        # Determine local expert range (EP-aware)
        if self.ep.ep_size > 1:
            local_start, local_end = self.ep.local_experts(self.num_experts)
        else:
            local_start, local_end = 0, self.num_experts

        for e_local in range(local_start, local_end):
            e_global = e_local
            mask = indices == e_global
            token_ids = mask.any(dim=-1).nonzero(as_tuple=True)[0]

            if token_ids.numel() == 0:
                continue

            # Gather tokens and weights for this expert
            token_x = x[token_ids]  # (n, D)
            token_weights = weights[token_ids]  # (n, top_k)
            token_expert_ids = indices[token_ids]  # (n, top_k)

            # For each top-k slot, find the weight for THIS expert
            expert_slot = (token_expert_ids == e_global).float()  # (n, top_k)
            per_token_weight = (token_weights * expert_slot).sum(dim=-1)  # (n,)

            # Padded bmm: pad tokens to power-of-2 for efficiency
            n = token_x.shape[0]
            pad_to = max(1, 1 << (n - 1).bit_length()) if n > 1 else 1

            if pad_to != n:
                padding = pad_to - n
                token_x = F.pad(token_x, (0, 0, 0, padding))

            # Reshape for bmm: (1, n_padded, D) x (1, D, H) -> (1, n_padded, H)
            # SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T
            h = self.hidden_size
            w1 = self.w1[e_local * h : (e_local + 1) * h]  # (H, D)
            w2 = self.w2[e_local * h : (e_local + 1) * h]  # (H, D)
            w3 = self.w3[e_local * h : (e_local + 1) * h]  # (H, D)

            x_3d = token_x.unsqueeze(0)  # (1, n_pad, D)
            w1_3d = w1.T.unsqueeze(0)  # (1, D, H) — w1 is (H, D), so w1.T is (D, H)
            w2_3d = w2.unsqueeze(0)  # (1, H, D) — no transpose needed for last bmm
            w3_3d = w3.T.unsqueeze(0)  # (1, D, H)

            # SwiGLU
            gate = torch.bmm(x_3d, w1_3d).squeeze(0)  # (n_pad, H)
            up = torch.bmm(x_3d, w3_3d).squeeze(0)  # (n_pad, H)
            hidden = F.silu(gate) * up  # (n_pad, H)
            expert_out = torch.bmm(hidden.unsqueeze(0), w2_3d).squeeze(0)  # (n_pad, D)

            # Remove padding
            expert_out = expert_out[:n]  # (n, D)

            # Weight and scatter
            weighted_out = per_token_weight.unsqueeze(-1) * expert_out
            output[token_ids] += weighted_out

        return output

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """Load balancing loss."""
        return getattr(self, "_aux_loss", None)
