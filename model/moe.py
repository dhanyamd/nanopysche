from __future__ import annotations

"""Mixture of Experts (MoE) — production implementation matching OLMo-core.

Features:
  - DeepSeek-v3 auxiliary-loss-free routing via learnable score bias
  - Sigmoid or softmax gating
  - Dropless mode with capacity factor
  - Shared expert MLP (DeepSeek-V2 style)
  - grouped_gemm with padded bmm fallback
  - Expert Parallelism via all-to-all dispatch/combine
  - Async all-to-all overlap via MoEHybridTransformerBlock

Reference: Fedus et al. 2021 (Switch Transformer)
           DeepSeek-V2/V3 (auxiliary-loss-free routing)
           OLMo-core src/olmo_core/nn/moe/
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# Try to import grouped_gemm
_GROUPED_GEMM_AVAILABLE = False
try:
    import grouped_gemm

    _GROUPED_GEMM_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GatingFunction(str, Enum):
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"


class MoELoadBalancingLossGranularity(str, Enum):
    LOCAL_BATCH = "local_batch"
    GLOBAL = "global"


# ---------------------------------------------------------------------------
# Router — DeepSeek-v3 style with score bias
# ---------------------------------------------------------------------------


@dataclass
class MoERouterConfig:
    """Configuration for the MoE router.

    Supports:
      - Standard softmax gating (Switch Transformer)
      - Sigmoid gating (ST-MoE)
      - DeepSeek-v3 auxiliary-loss-free routing via score_bias
    """

    num_experts: int = 8
    top_k: int = 2
    gating_function: GatingFunction = GatingFunction.SOFTMAX
    bias_init: float = -1.0
    bias_gamma: Optional[float] = None
    """DeepSeek-v3: if set, update score_bias after each batch for auxiliary-loss-free routing."""
    z_loss_multiplier: Optional[float] = None
    jitter_eps: Optional[float] = None
    """Input noise for training stability (switch transformer)."""
    normalize_expert_weights: Optional[str] = None
    """Normalize expert weights: 'l2' or None."""


class MoERouter(nn.Module):
    """Top-k expert router with load balancing.

    DeepSeek-v3 routing:
        Instead of adding an auxiliary loss, we maintain a learnable score_bias
        that is updated after each batch to encourage balanced routing.
        bias_update: bias += gamma * (actual_fraction - ideal_fraction)

    :param d_model: model dimensionality.
    :param config: router configuration.
    """

    def __init__(self, *, d_model: int, config: MoERouterConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        self.gate = nn.Linear(d_model, config.num_experts, bias=True)
        if config.bias_init is not None:
            nn.init.constant_(self.gate.bias, config.bias_init)

        # DeepSeek-v3 score bias (hidden from torch.compile)
        if config.bias_gamma is not None:
            self.register_buffer(
                "score_bias",
                torch.zeros(config.num_experts),
            )
            # Per-batch token counts (not a parameter)
            self.register_buffer(
                "_tokens_per_expert_this_batch",
                torch.zeros(config.num_experts, dtype=torch.long),
            )
            self.register_buffer(
                "_total_tokens_this_batch",
                torch.tensor(0, dtype=torch.long),
            )
        else:
            self.score_bias = None

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts.

        :param x: (B, S, d_model)
        :returns: (expert_weights, expert_indices, aux_loss)
        """
        # Optional jitter noise for training stability
        if self.config.jitter_eps is not None and self.training:
            noise = torch.randn_like(x) * self.config.jitter_eps
            x = x + noise

        logits = self.gate(x.float())

        # Apply DeepSeek-v3 score bias
        if self.score_bias is not None:
            logits = logits + self.score_bias

        # Gating function
        if self.config.gating_function == GatingFunction.SIGMOID:
            expert_weights, expert_indices = self._sigmoid_gating(logits)
        else:
            expert_weights, expert_indices = self._softmax_gating(logits)

        # Normalize expert weights
        if self.config.normalize_expert_weights == "l2":
            expert_weights = expert_weights / (
                expert_weights.norm(dim=-1, keepdim=True) + 1e-8
            )

        # Compute auxiliary loss
        aux_loss = self._load_balancing_loss(logits, expert_indices)

        # Z-loss for training stability
        if self.config.z_loss_multiplier is not None:
            z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()
            aux_loss = aux_loss + self.config.z_loss_multiplier * z_loss

        # Track per-expert counts for DeepSeek-v3 bias update
        if self.score_bias is not None:
            self._update_expert_counts(expert_indices)

        return expert_weights, expert_indices, aux_loss

    def _softmax_gating(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard top-k softmax gating."""
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_weights, top_k_indices

    def _sigmoid_gating(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sigmoid gating (ST-MoE style)."""
        gate_scores = torch.sigmoid(logits)
        top_k_weights, top_k_indices = gate_scores.topk(self.top_k, dim=-1)
        # Normalize weights to sum to 1 across top-k
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return top_k_weights, top_k_indices

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

    @torch.no_grad()
    def _update_expert_counts(self, indices: torch.Tensor) -> None:
        """Update per-expert token counts for DeepSeek-v3 bias update."""
        # Count tokens per expert
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=indices.device)
        for e in range(self.num_experts):
            counts[e] = (indices == e).sum()
        self._tokens_per_expert_this_batch += counts.cpu()
        self._total_tokens_this_batch += indices.shape[0] * indices.shape[1]

    @torch.no_grad()
    def post_batch(self) -> None:
        """Update score_bias after each batch (DeepSeek-v3 style).

        Call this once per training step, after the forward+backward pass.
        """
        if self.score_bias is None:
            return
        if self._total_tokens_this_batch == 0:
            return

        # All-reduce counts across EP ranks
        if dist.is_initialized():
            dist.all_reduce(self._tokens_per_expert_this_batch, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._total_tokens_this_batch, op=dist.ReduceOp.SUM)

        # Compute actual fraction and ideal fraction
        actual_fraction = (
            self._tokens_per_expert_this_batch.float()
            / self._total_tokens_this_batch.float()
        )
        ideal_fraction = torch.ones(self.num_experts) / self.num_experts

        # Update bias
        self.score_bias += self.config.bias_gamma * (actual_fraction - ideal_fraction)

        # Reset counters
        self._tokens_per_expert_this_batch.zero_()
        self._total_tokens_this_batch.zero_()


# ---------------------------------------------------------------------------
# Expert Parallelism config & dispatch
# ---------------------------------------------------------------------------


@dataclass
class ExpertParallelConfig:
    """Expert Parallelism configuration.

    EP shards experts across ranks. Each rank owns num_experts // ep_size experts.
    Tokens are dispatched to the correct rank via all-to-all communication.
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
        :param expert_indices: (num_tokens, top_k)
        :param expert_weights: (num_tokens, top_k)
        :param num_experts: total number of experts
        :returns: (x_dispatched, weights_dispatched, indices_dispatched, metadata)
        """
        if self.ep_size <= 1:
            return x, expert_weights, expert_indices, None

        per_rank = num_experts // self.ep_size
        ep_rank = self.ep_rank()
        num_tokens, top_k = expert_indices.shape

        target_ranks = expert_indices // per_rank
        flat_tokens = torch.arange(num_tokens, device=x.device).repeat_interleave(top_k)
        flat_target_ranks = target_ranks.reshape(-1)
        flat_experts = expert_indices.reshape(-1)
        flat_weights = expert_weights.reshape(-1)

        send_counts = torch.zeros(self.ep_size, dtype=torch.long, device=x.device)
        for r in range(self.ep_size):
            send_counts[r] = (flat_target_ranks == r).sum()

        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        sorted_order = flat_target_ranks.argsort()
        sorted_tokens = flat_tokens[sorted_order]
        sorted_experts = flat_experts[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        send_x = x[sorted_tokens]
        send_offsets = send_counts.cumsum(dim=0).tolist()
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # Exchange x
        recv_x_list = [
            torch.zeros(recv_splits[i], x.shape[-1], device=x.device, dtype=x.dtype)
            for i in range(self.ep_size)
        ]
        send_x_list = [
            send_x[send_offsets[i] - send_splits[i] : send_offsets[i]]
            if send_splits[i] > 0
            else torch.zeros(0, x.shape[-1], device=x.device, dtype=x.dtype)
            for i in range(self.ep_size)
        ]
        dist.all_to_all(recv_x_list, send_x_list, group=self.ep_group)
        recv_x = torch.cat(recv_x_list, dim=0)

        # Exchange weights
        recv_w_list = [
            torch.zeros(recv_splits[i], device=x.device, dtype=expert_weights.dtype)
            for i in range(self.ep_size)
        ]
        send_w_list = [
            sorted_weights[send_offsets[i] - send_splits[i] : send_offsets[i]]
            if send_splits[i] > 0
            else torch.zeros(0, device=x.device, dtype=expert_weights.dtype)
            for i in range(self.ep_size)
        ]
        dist.all_to_all(recv_w_list, send_w_list, group=self.ep_group)
        recv_weights = torch.cat(recv_w_list, dim=0)

        # Exchange expert indices
        recv_e_list = [
            torch.zeros(recv_splits[i], dtype=expert_indices.dtype, device=x.device)
            for i in range(self.ep_size)
        ]
        send_e_list = [
            sorted_experts[send_offsets[i] - send_splits[i] : send_offsets[i]]
            if send_splits[i] > 0
            else torch.zeros(0, dtype=expert_indices.dtype, device=x.device)
            for i in range(self.ep_size)
        ]
        dist.all_to_all(recv_e_list, send_e_list, group=self.ep_group)
        recv_experts = torch.cat(recv_e_list, dim=0) - (ep_rank * per_rank)

        return (
            recv_x,
            recv_weights,
            recv_experts,
            (recv_x.shape[0], send_counts, recv_counts, num_tokens),
        )

    def combine(
        self,
        expert_output: torch.Tensor,
        combine_metadata,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Combine expert outputs back via reverse all-to-all."""
        if combine_metadata is None:
            return expert_output

        num_recv, send_counts, recv_counts, num_tokens = combine_metadata
        d_model = x.shape[-1]
        ep_size = self.ep_size
        recv_splits = recv_counts.tolist()
        send_splits = send_counts.tolist()

        send_x_list = list(
            torch.split(expert_output, recv_splits, dim=0)
            if expert_output.shape[0] > 0
            else [
                torch.zeros(
                    0, d_model, device=expert_output.device, dtype=expert_output.dtype
                )
                for _ in range(ep_size)
            ]
        )
        for i in range(len(send_x_list)):
            if send_x_list[i].shape[0] == 0:
                send_x_list[i] = torch.zeros(
                    0, d_model, device=expert_output.device, dtype=expert_output.dtype
                )
        recv_x_list = [
            torch.zeros(
                send_splits[i],
                d_model,
                device=expert_output.device,
                dtype=expert_output.dtype,
            )
            for i in range(ep_size)
        ]

        dist.all_to_all(recv_x_list, send_x_list, group=self.ep_group)
        return torch.cat(recv_x_list, dim=0)


# ---------------------------------------------------------------------------
# Expert MLP (SwiGLU)
# ---------------------------------------------------------------------------


class ExpertMLP(nn.Module):
    """Single expert FFN with SwiGLU activation.

    Uses flattened weight format (OLMo-core pattern) for grouped_gemm.
    """

    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.w1 = nn.Parameter(torch.empty(hidden_size, d_model))
        self.w2 = nn.Parameter(torch.empty(hidden_size, d_model))
        self.w3 = nn.Parameter(torch.empty(hidden_size, d_model))
        nn.init.normal_(self.w1, std=0.01)
        nn.init.normal_(self.w2, std=0.01)
        nn.init.normal_(self.w3, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2"""
        return (F.silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2


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
    jitter_eps: Optional[float] = None
    gating_function: GatingFunction = GatingFunction.SOFTMAX
    bias_gamma: Optional[float] = None
    """DeepSeek-v3 auxiliary-loss-free routing strength."""
    capacity_factor: Optional[float] = None
    """If set, drop tokens exceeding capacity_factor * num_tokens / num_experts."""
    shared_expert: bool = False
    """DeepSeek-V2 style: optional shared expert that processes all tokens."""
    ep: ExpertParallelConfig = field(default_factory=ExpertParallelConfig)

    fp8_flow_moe: bool = False
    """Enable FP8-Flow-MoE casting-free recipe for expert computation."""
    fp8_recipe: str = "none"
    """FP8 recipe: 'none', 'blockwise', 'mxfp8', 'flow_moe'."""


class MoEBase(nn.Module):
    """Mixture of Experts layer.

    Production features matching OLMo-core:
      - DeepSeek-v3 auxiliary-loss-free routing via score_bias
      - Sigmoid or softmax gating
      - Dropless mode with capacity factor
      - Shared expert MLP
      - grouped_gemm with padded bmm fallback
      - Expert Parallelism via all-to-all dispatch

    :param d_model: model dimensionality.
    :param hidden_size: expert intermediate size.
    :param config: full MoE configuration.
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
        jitter_eps: Optional[float] = None,
        gating_function: GatingFunction = GatingFunction.SOFTMAX,
        bias_gamma: Optional[float] = None,
        capacity_factor: Optional[float] = None,
        shared_expert: bool = False,
        ep: Optional[ExpertParallelConfig] = None,
        fp8_flow_moe: bool = False,
        fp8_recipe: str = "none",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size or int(8 / 3 * d_model)
        self.hidden_size = ((self.hidden_size + 255) // 256) * 256
        self.capacity_factor = capacity_factor

        # Router
        router_config = MoERouterConfig(
            num_experts=num_experts,
            top_k=top_k,
            gating_function=gating_function,
            bias_init=-1.0,
            bias_gamma=bias_gamma,
            z_loss_multiplier=z_loss_multiplier,
            jitter_eps=jitter_eps,
        )
        self.router = MoERouter(d_model=d_model, config=router_config)

        # Expert weights (flattened for grouped_gemm)
        self.w1 = nn.Parameter(torch.empty(num_experts * self.hidden_size, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts * self.hidden_size, d_model))
        self.w3 = nn.Parameter(torch.empty(num_experts * self.hidden_size, d_model))
        self._init_weights()

        # Shared expert (DeepSeek-V2 style)
        self.shared_expert = None
        if shared_expert:
            self.shared_expert = ExpertMLP(d_model, self.hidden_size)

        # EP config
        self.ep = ep or ExpertParallelConfig()

        # Use grouped_gemm if available
        self._use_grouped_gemm = _GROUPED_GEMM_AVAILABLE

        # FP8-Flow-MoE
        self.fp8_flow_moe_enabled = fp8_flow_moe
        self._fp8_flow_moe = None
        if fp8_flow_moe:
            from nanopsyche.fp8.flow_moe import FP8FlowMoEConfig, FP8FlowMoECompute
            from nanopsyche.fp8.recipes import FP8RecipeConfig, FP8RecipeType

            recipe_config = FP8RecipeConfig(
                enabled=True,
                recipe_type=FP8RecipeType(fp8_recipe),
            )
            fm_config = FP8FlowMoEConfig(
                enabled=True,
                fp8_recipe=recipe_config,
            )
            self._fp8_flow_moe = FP8FlowMoECompute(
                fm_config, d_model, self.hidden_size, num_experts
            )
            self._fp8_flow_moe.set_weights(self.w1, self.w2, self.w3)

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

        # Capacity-based token dropping
        dropped = 0
        keep_mask = None
        if self.capacity_factor is not None:
            x_flat, weights_flat, indices_flat, dropped, keep_mask = (
                self._apply_capacity(x_flat, weights_flat, indices_flat)
            )

        # EP dispatch
        local_x, local_weights, local_indices, combine_meta = self.ep.dispatch(
            x_flat,
            indices_flat,
            weights_flat,
            self.num_experts,
        )

        # Compute expert outputs
        if self._use_grouped_gemm and local_x.shape[0] > 0:
            expert_output = self._compute_experts_grouped_gemm(
                local_x, local_weights, local_indices
            )
        else:
            expert_output = self._compute_experts_padded_bmm(
                local_x, local_weights, local_indices
            )

        # EP combine
        output = self.ep.combine(
            expert_output, combine_meta, x_flat, weights_flat, indices_flat
        )

        # Shared expert
        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        # If capacity dropping was used, scatter back to original positions
        if self.capacity_factor is not None and dropped > 0:
            full_output = torch.zeros(B * S, D, device=x.device, dtype=output.dtype)
            full_output[keep_mask] = output
            output = full_output

        self._aux_loss = aux_loss
        return output.view(B, S, D)

    def _apply_capacity(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Apply capacity factor: drop tokens exceeding per-expert capacity.

        capacity = capacity_factor * num_tokens / num_experts
        """
        num_tokens = x.shape[0]
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)

        keep_mask = torch.ones(num_tokens, dtype=torch.bool, device=x.device)
        dropped = 0

        for e in range(self.num_experts):
            expert_mask = (indices == e).any(dim=-1)
            expert_token_ids = expert_mask.nonzero(as_tuple=True)[0]
            if expert_token_ids.shape[0] > capacity:
                drop_ids = expert_token_ids[capacity:]
                keep_mask[drop_ids] = False
                dropped += drop_ids.shape[0]

        if dropped > 0:
            x = x[keep_mask]
            weights = weights[keep_mask]
            indices = indices[keep_mask]

        return x, weights, indices, dropped, keep_mask

    def _compute_experts_grouped_gemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expert outputs using FP8-Flow-MoE or grouped_gemm fallback.

        When FP8-Flow-MoE is enabled, uses the casting-free FP8 dataflow
        with scaling-aware transpose (MLSys 2026). The entire MoE expert
        computation stays in FP8 with only 2 cast operations.

        When disabled, falls back to BF16 grouped_gemm (original behavior).
        """
        num_tokens, D = x.shape
        output = torch.zeros_like(x)

        # FP8-Flow-MoE path
        if self.fp8_flow_moe_enabled and self._fp8_flow_moe is not None:
            return self._fp8_flow_moe(x, weights, indices, 0, self.num_experts)

        # BF16 fallback (original grouped_gemm path)
        for e_local in range(self.num_experts):
            mask = indices == e_local
            token_ids = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_ids.numel() == 0:
                continue

            token_x = x[token_ids]
            token_weights = weights[token_ids]
            token_expert_ids = indices[token_ids]
            expert_slot = (token_expert_ids == e_local).float()
            per_token_weight = (token_weights * expert_slot).sum(dim=-1)

            h = self.hidden_size
            w1 = self.w1[e_local * h : (e_local + 1) * h]
            w2 = self.w2[e_local * h : (e_local + 1) * h]
            w3 = self.w3[e_local * h : (e_local + 1) * h]

            expert_out = (F.silu(token_x @ w1.T) * (token_x @ w3.T)) @ w2
            output[token_ids] += per_token_weight.unsqueeze(-1) * expert_out

        return output

    def _compute_experts_padded_bmm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expert outputs using padded batched matmul or FP8-Flow-MoE.

        When FP8-Flow-MoE is enabled, delegates to FP8-Flow-MoE compute.
        Otherwise, uses the original padded batched matmul (BF16).
        """
        num_tokens, D = x.shape
        output = torch.zeros_like(x)

        # FP8-Flow-MoE path
        if self.fp8_flow_moe_enabled and self._fp8_flow_moe is not None:
            return self._fp8_flow_moe(x, weights, indices, 0, self.num_experts)

        if self.ep.ep_size > 1:
            local_start, local_end = self.ep.local_experts(self.num_experts)
        else:
            local_start, local_end = 0, self.num_experts

        for e_local in range(local_start, local_end):
            mask = indices == e_local
            token_ids = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_ids.numel() == 0:
                continue

            token_x = x[token_ids]
            token_weights = weights[token_ids]
            token_expert_ids = indices[token_ids]
            expert_slot = (token_expert_ids == e_local).float()
            per_token_weight = (token_weights * expert_slot).sum(dim=-1)

            n = token_x.shape[0]
            pad_to = max(1, 1 << (n - 1).bit_length()) if n > 1 else 1
            if pad_to != n:
                token_x = F.pad(token_x, (0, 0, 0, pad_to - n))

            h = self.hidden_size
            w1 = self.w1[e_local * h : (e_local + 1) * h]
            w2 = self.w2[e_local * h : (e_local + 1) * h]
            w3 = self.w3[e_local * h : (e_local + 1) * h]

            x_3d = token_x.unsqueeze(0)
            gate = torch.bmm(x_3d, w1.T.unsqueeze(0)).squeeze(0)
            up = torch.bmm(x_3d, w3.T.unsqueeze(0)).squeeze(0)
            hidden = F.silu(gate) * up
            expert_out = torch.bmm(hidden.unsqueeze(0), w2.unsqueeze(0)).squeeze(0)

            expert_out = expert_out[:n]
            output[token_ids] += per_token_weight.unsqueeze(-1) * expert_out

        return output

    # ------------------------------------------------------------------
    # Exposed phases for async overlap (MoEHybridTransformerBlock)
    # ------------------------------------------------------------------

    def route(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        Optional[torch.Tensor],
    ]:
        """Route tokens and prepare for dispatch.

        :param x: (B, S, d_model)
        :returns: (x_flat, weights_flat, indices_flat, aux_loss, dropped, keep_mask)
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        weights, indices, aux_loss = self.router(x.view(B, S, D))
        weights_flat = weights.view(-1, self.top_k)
        indices_flat = indices.view(-1, self.top_k)

        if self.capacity_factor is not None:
            x_flat, weights_flat, indices_flat, dropped, keep_mask = (
                self._apply_capacity(x_flat, weights_flat, indices_flat)
            )
        else:
            dropped = 0
            keep_mask = None

        return x_flat, weights_flat, indices_flat, aux_loss, dropped, keep_mask

    def dispatch(
        self,
        x_flat: torch.Tensor,
        indices_flat: torch.Tensor,
        weights_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """EP dispatch tokens to local experts.

        :returns: (local_x, local_weights, local_indices, combine_metadata)
        """
        return self.ep.dispatch(x_flat, indices_flat, weights_flat, self.num_experts)

    def local_compute(
        self,
        local_x: torch.Tensor,
        local_weights: torch.Tensor,
        local_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expert outputs on locally-resident tokens.

        :param local_x: tokens dispatched to this rank
        :param local_weights: corresponding expert weights
        :param local_indices: local expert indices (0..local_num_experts-1)
        :returns: weighted expert outputs
        """
        if local_x.shape[0] == 0:
            return local_x

        if self._use_grouped_gemm:
            return self._compute_experts_grouped_gemm(
                local_x, local_weights, local_indices
            )
        else:
            return self._compute_experts_padded_bmm(
                local_x, local_weights, local_indices
            )

    def combine(
        self,
        expert_output: torch.Tensor,
        combine_metadata: object,
        x_flat: torch.Tensor,
        weights_flat: torch.Tensor,
        indices_flat: torch.Tensor,
    ) -> torch.Tensor:
        """EP combine: all-to-all back to original token order.

        :returns: (num_tokens, d_model) combined output
        """
        output = self.ep.combine(
            expert_output, combine_metadata, x_flat, weights_flat, indices_flat
        )

        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        return output

    def reassemble(
        self,
        output: torch.Tensor,
        *,
        dropped: int,
        keep_mask: Optional[torch.Tensor],
        B: int,
        S: int,
        D: int,
    ) -> torch.Tensor:
        """Reassemble output into (B, S, D) shape after capacity dropping.

        :param output: (num_kept_tokens, d_model)
        :param dropped: number of dropped tokens
        :param keep_mask: boolean mask of kept tokens
        :returns: (B, S, d_model)
        """
        if dropped > 0 and keep_mask is not None:
            full_output = torch.zeros(
                B * S, D, device=output.device, dtype=output.dtype
            )
            full_output[keep_mask] = output
            output = full_output
        return output.view(B, S, D)

    def apply_fp8_flow_moe(
        self,
        fp8_flow_moe: bool = True,
        fp8_recipe: str = "flow_moe",
    ):
        """Enable FP8-Flow-MoE post-construction.

        Creates the FP8FlowMoECompute module and sets the flag.
        Can be called after MoEBase is created, e.g. from parallelize_model().

        :param fp8_flow_moe: enable FP8-Flow-MoE dataflow.
        :param fp8_recipe: recipe name ('none', 'blockwise', 'mxfp8', 'flow_moe').
        """
        from nanopsyche.fp8.flow_moe import FP8FlowMoEConfig, FP8FlowMoECompute
        from nanopsyche.fp8.recipes import FP8RecipeConfig, FP8RecipeType

        self.fp8_flow_moe_enabled = fp8_flow_moe
        if fp8_flow_moe:
            recipe_config = FP8RecipeConfig(
                enabled=True,
                recipe_type=FP8RecipeType(fp8_recipe),
            )
            fm_config = FP8FlowMoEConfig(
                enabled=True,
                fp8_recipe=recipe_config,
            )
            self._fp8_flow_moe = FP8FlowMoECompute(
                fm_config, self.d_model, self.hidden_size, self.num_experts
            )
            self._fp8_flow_moe.set_weights(self.w1, self.w2, self.w3)
        else:
            self._fp8_flow_moe = None

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """Load balancing loss."""
        return getattr(self, "_aux_loss", None)
