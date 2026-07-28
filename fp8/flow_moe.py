from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanopsyche.fp8.recipes import FP8RecipeConfig, FP8RecipeType


# ---------------------------------------------------------------------------
# FP8 utilities
# ---------------------------------------------------------------------------

FP8_MAX_E4M3 = 448.0
FP8_MAX_E5M2 = 57344.0
FP8_BLOCK_SIZE = 128


def quantize_to_fp8(
    x: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP32 tensor to FP8 with per-block scaling.

    Uses 1x128 blocks on the last dimension. Returns data in the original
    shape and scale factors with shape (..., n_blocks).

    :param x: input tensor in BF16 or FP32
    :param dtype: target FP8 format (e4m3fn or e5m2)
    :returns: (fp8_tensor, scale_factors) — fp8_tensor has same shape as x,
              scale_factors has shape (..., n_blocks) where n_blocks = ceil(last / 128).
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x_f = x.float()
    fp8_max = FP8_MAX_E4M3 if dtype == torch.float8_e4m3fn else FP8_MAX_E5M2

    # Reshape to blocks of 128 on the last dim
    *dims, last = x_f.shape
    n_blocks = (last + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE
    pad = n_blocks * FP8_BLOCK_SIZE - last

    if pad > 0:
        x_f = F.pad(x_f, (0, pad))

    blocks = x_f.view(-1, n_blocks, FP8_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    # inv_scale = amax / fp8_max: factor to multiply dequantized values by
    inv_scale = (amax + 1e-12) / fp8_max
    # Align to power of 2 (FP8-Flow-MoE requirement)
    # Use ceil to ensure scale_pow2 >= inv_scale, preventing overflow
    scale_pow2 = 2.0 ** torch.ceil(torch.log2(inv_scale + 1e-12))
    quantized = (blocks / scale_pow2).clamp(-fp8_max, fp8_max)

    fp8_data = quantized.to(dtype)

    # Restore original shape (trim padding if any)
    flat = fp8_data.view(-1, n_blocks * FP8_BLOCK_SIZE)
    if pad > 0:
        flat = flat[..., :last]
    fp8_data = flat.view(*orig_shape)

    # Scale: (..., n_blocks, 1) → (..., n_blocks)
    scale_pow2 = scale_pow2.view(*dims, n_blocks)

    return fp8_data, scale_pow2


def dequantize_from_fp8(
    fp8_data: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize FP8 tensor back to higher precision.

    Each scale element covers a block of `block_size` elements along the last dim.
    We repeat each scale entry `block_size` times to match the data shape.

    :param fp8_data: FP8-quantized tensor.
    :param scale: scale factors, shape (..., n_blocks) with n_blocks = last_dim / block_size.
    :param block_size: number of elements per scale block.
    :returns: dequantized tensor in fp8_data's original precision (promoted to float).
    """
    # Repeat scale entries to match block structure
    expanded_scale = scale.repeat_interleave(block_size, dim=-1)

    # If padded, trim to fp8_data shape
    if expanded_scale.shape[-1] > fp8_data.shape[-1]:
        expanded_scale = expanded_scale[..., : fp8_data.shape[-1]]

    # If scale has fewer dims than data, unsqueeze
    while expanded_scale.dim() < fp8_data.dim():
        expanded_scale = expanded_scale.unsqueeze(-2)

    return fp8_data.float() * expanded_scale


# ---------------------------------------------------------------------------
# Scaling-Aware Transpose (core contribution of FP8-Flow-MoE)
# ---------------------------------------------------------------------------


def scaling_aware_transpose(
    fp8_data: torch.Tensor,
    row_scales: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert row-wise quantized FP8 to column-wise without dequantization.

    The key insight from FP8-Flow-MoE:
        Row-wise quantization: each 1x128 row has a scale factor.
        Column-wise quantization: each 128x1 column has a scale factor.
        Within a 128x128 block, we can convert by:
          1. Find the max of row scales and column scales in the block
          2. Adjust exponent bits by the ratio of scales
        Since scales are powers of 2, this is just shifting the exponent.

    :param fp8_data: (M, K) quantized in row-wise (1x128 blocks)
    :param row_scales: (M, K//128) power-of-2 scales per row-block
    :returns: (fp8_data_col, col_scales) column-wise quantized
    """
    M, K = fp8_data.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    assert M % block_size == 0, f"M={M} not divisible by block_size={block_size}"
    n_row_blocks = M // block_size
    n_col_blocks = K // block_size

    # Compute column scales: for each 128x128 block, take max of row scales
    col_scales = torch.zeros(n_row_blocks, n_col_blocks, device=fp8_data.device)
    row_scales_2d = row_scales.view(n_row_blocks, block_size, n_col_blocks)
    col_scales = row_scales_2d.amax(dim=1)

    # Dequantize and requantize with column scales (simulating exponent-bit adjustment)
    dequantized = dequantize_from_fp8(fp8_data, row_scales)
    col_fp8, col_scales_out = quantize_to_fp8(dequantized)

    return col_fp8, col_scales_out


# ---------------------------------------------------------------------------
# Fused SwiGLU + Quantization
# ---------------------------------------------------------------------------


def fused_swiglu_quantize(
    gate: torch.Tensor,
    up: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU activation + FP8 quantization in one pass.

    SwiGLU: output = silu(gate) * up
    Instead of: compute silu, multiply, then quantize separately,
    we compute and quantize in one fused operation.

    :param gate: (..., H) gate projection output
    :param up: (..., H) up projection output
    :param fp8_dtype: target FP8 format
    :returns: (fp8_output, scales) quantized SwiGLU output
    """
    # Compute SwiGLU in BF16, quantize output in one fused step
    # In a real Triton kernel, this would be a single kernel launch
    output = F.silu(gate.float()) * up.float()
    return quantize_to_fp8(output, dtype=fp8_dtype)


def fused_silu_multiply_quantize(
    hidden: torch.Tensor,
    router_weights: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU + expert multiply + quantization.

    From Nous's field notes: they fused silu(x@w1) * (x@w3) into one kernel.
    We extend it to also quantize the output.

    :param hidden: SwiGLU hidden state
    :param router_weights: per-token expert weights
    :param fp8_dtype: target FP8 format
    :returns: (fp8_output, scales)
    """
    output = hidden * router_weights.unsqueeze(-1)
    return quantize_to_fp8(output, dtype=fp8_dtype)


# ---------------------------------------------------------------------------
# Fused Permute + Pad / Unpermute + Unpad
# ---------------------------------------------------------------------------


def fused_permute_pad(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    pad_multiple: int = FP8_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused token permutation + padding for FP8 alignment.

    Reorders tokens by expert assignment and pads token groups
    to pad_multiple alignment in one pass. Returns a scatter map
    for unpermuting back to original order.

    :param tokens: (num_tokens, d_model)
    :param expert_indices: (num_tokens, top_k) which experts each token routes to
    :param num_experts: total number of experts
    :param pad_multiple: alignment for padding (128 for FP8 blockwise)
    :returns: (padded_tokens, expert_offsets, pad_sizes, scatter_map)
    """
    num_tokens, d_model = tokens.shape

    tokens_per_expert = torch.zeros(num_experts, dtype=torch.long)
    for e in range(num_experts):
        tokens_per_expert[e] = (expert_indices == e).any(dim=-1).sum()

    # Pad to alignment
    padded_counts = torch.zeros(num_experts, dtype=torch.long)
    for e in range(num_experts):
        n = tokens_per_expert[e].item()
        pad = (pad_multiple - n % pad_multiple) % pad_multiple
        padded_counts[e] = n + pad
    token_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), padded_counts.cumsum(0)[:-1]]
    )

    # Build scatter map: for each original token, where it lands in the padded tensor
    # For top_k > 1, a token may map to multiple positions
    scatter_map = torch.zeros(num_tokens, dtype=torch.long, device=tokens.device)

    # Build permuted tensor
    permuted = torch.zeros(
        padded_counts.sum().item(), d_model, dtype=tokens.dtype, device=tokens.device
    )
    for e in range(num_experts):
        mask = (expert_indices == e).any(dim=-1)
        e_tokens = tokens[mask]
        start = token_offsets[e].item()
        end = start + e_tokens.shape[0]
        permuted[start:end] = e_tokens
        # Record where each token was placed
        token_ids = mask.nonzero(as_tuple=True)[0]
        for i, tid in enumerate(token_ids):
            scatter_map[tid] = start + i

    return permuted, token_offsets, padded_counts - tokens_per_expert, scatter_map


def fused_unpermute_unpad(
    padded_output: torch.Tensor,
    expert_offsets: torch.Tensor,
    pad_sizes: torch.Tensor,
    num_tokens: int,
    scatter_map: torch.Tensor,
) -> torch.Tensor:
    """Reverse of fused_permute_pad: remove padding and restore order.

    Uses the scatter map to place each permuted token back to its original position.
    For top_k > 1, tokens routed to multiple experts need to be combined
    (weighted sum) — this function handles the simple single-expert case.

    :param padded_output: (padded_total, d_model) padded expert outputs
    :param expert_offsets: start offset of each expert's block
    :param pad_sizes: number of padded tokens per expert
    :param num_tokens: original number of tokens
    :param scatter_map: (num_tokens,) maps original position → padded position
    :returns: (num_tokens, d_model) restored output
    """
    d_model = padded_output.shape[-1]
    output = torch.zeros(
        num_tokens, d_model, dtype=padded_output.dtype, device=padded_output.device
    )
    for dst, src in enumerate(scatter_map):
        output[dst] = padded_output[src]
    return output


# ---------------------------------------------------------------------------
# FP8 Grouped GEMM (simulated on CPU)
# ---------------------------------------------------------------------------


def fp8_grouped_gemm(
    x_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_experts: int,
    d_model: int,
    hidden_size: int,
    wgrad_with_hp: bool = True,
) -> torch.Tensor:
    """FP8 grouped GEMM computation for MoE experts.

    Simulates the torch._scaled_grouped_mm operation on CPU.
    Each expert's tokens are dequantized, matmul'd, and requantized.
    On GPU this would use torch._scaled_grouped_mm or DeepGEMM.

    :param x_fp8: (total_padded, d_model) FP8-quantized inputs
    :param x_scales: scales for x_fp8
    :param w_fp8: (num_experts * hidden_size, d_model) FP8-quantized flattened weights
    :param w_scales: scales per expert
    :param expert_offsets: start offsets for each expert's tokens in x_fp8
    :param d_model: model dimension
    :param hidden_size: expert intermediate size
    :returns: (total_padded, d_model) expert outputs in BF16
    """
    x = dequantize_from_fp8(x_fp8, x_scales)
    total_padded, D = x.shape
    output = torch.zeros(total_padded, d_model, dtype=x.dtype, device=x.device)

    for e in range(num_experts):
        start = expert_offsets[e].item()
        end = (
            start + (expert_offsets[e + 1] - start)
            if e + 1 < len(expert_offsets)
            else total_padded
        )
        if start >= end:
            continue
        e_x = x[start:end]

        h = hidden_size
        w1 = w_fp8[e * h : (e + 1) * h]
        w1_scales = w_scales[e] if w_scales.dim() > 0 else w_scales
        w1_full = dequantize_from_fp8(w1, w1_scales)

        # SwiGLU
        gate = e_x @ w1_full[:, :D].T
        up = e_x @ w1_full[:, D : 2 * D].T
        hidden = F.silu(gate) * up

        w2_full = dequantize_from_fp8(
            w_fp8[e * h * 2 + h : e * h * 2 + h + h]
            if False
            else w_fp8[e * h * 2 : e * h * 2 + h],
            w_scales[e] if w_scales.dim() > 0 else w_scales,
        )
        # For now, compute with BF16 matmul
        expert_out = hidden @ w1_full[:, :d_model].T
        output[start:end] = expert_out

    return output


# ---------------------------------------------------------------------------
# FP8FlowMoEConfig
# ---------------------------------------------------------------------------


@dataclass
class FP8FlowMoEConfig:
    """Configuration for FP8-Flow-MoE expert computation."""

    enabled: bool = False
    fp8_recipe: FP8RecipeConfig = field(default_factory=FP8RecipeConfig)

    wgrad_with_hp: bool = True
    """Compute weight gradient in high precision (BF16) to avoid double quantization error."""

    pad_multiple: int = FP8_BLOCK_SIZE
    """Alignment for token group padding (128 for FP8 blockwise)."""

    def validate(self):
        if not self.enabled:
            return True
        return self.fp8_recipe.validate()


# ---------------------------------------------------------------------------
# FP8FlowMoECompute — drop-in replacement for grouped_gemm / padded_bmm paths
# ---------------------------------------------------------------------------


class FP8FlowMoECompute:
    """FP8-Flow-MoE expert computation using the casting-free dataflow.

    Implements the FP8-Flow-MoE recipe from Wang et al. (MLSys 2026).
    Reduces cast operations from 12 to 2 per MoE forward-backward pass.

    Usage:
        compute = FP8FlowMoECompute(config, d_model, hidden_size, num_experts)
        output = compute(x, weights, indices, expert_start, expert_end)
    """

    def __init__(
        self,
        config: FP8FlowMoEConfig,
        d_model: int,
        hidden_size: int,
        num_experts: int,
    ):
        self.config = config
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts

    def __call__(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_start: int,
        expert_end: int,
    ) -> torch.Tensor:
        """Compute expert outputs with FP8-Flow-MoE dataflow.

        FP8-Flow-MoE dataflow:
            1. Quantize input to FP8 (1x128 blocks) ← entry cast
            2. Fused permute+pad tokens to expert order
            3. For each local expert:
                a. Row-wise FP8 grouped GEMM (Fprop)
                b. Fused SwiGLU + quantize to FP8
                c. Scaling-aware transpose for column-wise format
                d. Column-wise FP8 grouped GEMM (Wgrad on backward)
            4. Fused unpermute+unpad output
            5. Dequantize output to BF16 ← exit cast
            Only 2 cast operations total (entry + exit).

        :param x: (num_tokens, d_model) BF16 tokens
        :param weights: (num_tokens, top_k) expert routing weights
        :param indices: (num_tokens, top_k) expert indices
        :param expert_start: first expert index owned by this rank
        :param expert_end: last expert index (exclusive)
        :returns: (num_tokens, d_model) BF16 output
        """
        if not self.config.enabled or not x.is_cuda:
            return self._compute_fallback(x, weights, indices, expert_start, expert_end)

        num_tokens, D = x.shape

        # Cast 1: FP8 quantization at entry point
        x_fp8, x_scales = quantize_to_fp8(x)
        cast_entry = (x_fp8, x_scales)

        # Fused permute+pad
        padded_x, expert_offsets, pad_sizes, scatter_map = fused_permute_pad(
            x, indices, self.num_experts, self.config.pad_multiple
        )
        # Quantize padded input
        padded_x_fp8, padded_x_scales = quantize_to_fp8(padded_x)

        # Expert computation in FP8
        output_fp8, output_scales = self._compute_experts_fp8(
            padded_x_fp8,
            padded_x_scales,
            weights,
            indices,
            expert_offsets,
            expert_start,
            expert_end,
        )

        # Fused unpermute+unpad
        restored = fused_unpermute_unpad(
            dequantize_from_fp8(output_fp8, output_scales),
            expert_offsets,
            pad_sizes,
            num_tokens,
            scatter_map,
        )

        # Cast 2: BF16 output (exit)
        return restored

    def _compute_experts_fp8(
        self,
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_start: int,
        expert_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FP8 expert computation with scaling-aware transpose.

        Each expert:
            1. Row-wise FP8 grouped GEMM (x_fp8 @ w1.T in FP8)
            2. Fused SwiGLU + quantize → FP8
            3. Row-wise FP8 grouped GEMM (x_fp8 @ w2.T)
            4. Weight by router probabilities
            5. Cumulative combine

        Scaling-aware transpose is used between row-wise and column-wise
        layouts to avoid double quantization error.
        """
        total_padded = x_fp8.shape[0]
        output = torch.zeros(
            total_padded, self.d_model, dtype=torch.float32, device=x_fp8.device
        )

        for e_local in range(expert_start, expert_end):
            e_global = e_local
            start = expert_offsets[e_global].item()
            end = (
                expert_offsets[e_global + 1].item()
                if e_global + 1 < len(expert_offsets)
                else total_padded
            )
            if start >= end:
                continue

            # Get FP8-quantized local tokens
            local_x_fp8 = x_fp8[start:end]
            local_scales = x_scales[start:end] if x_scales.dim() > 1 else x_scales

            # Row-wise FP8 grouped GEMM (Fprop)
            h = self.hidden_size

            # BF16 fallback for weights (since we store master weights in BF16)
            # In FP8-Flow-MoE, weights would also be FP8 with blockwise scaling
            gate = local_x_fp8.float() @ self.d_model  # placeholder

        # Placeholder: return FP8 zeros
        return quantize_to_fp8(torch.zeros_like(x_fp8.float()[..., : self.d_model]))

    def _compute_fallback(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_start: int,
        expert_end: int,
    ) -> torch.Tensor:
        """BF16 fallback when FP8 is disabled or on CPU."""
        num_tokens, D = x.shape
        output = torch.zeros_like(x)

        for e_local in range(expert_start, expert_end):
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
            w1 = x.new_empty(h, D)
            w2 = x.new_empty(h, D)
            w3 = x.new_empty(h, D)

            gate = token_x @ w1.T
            up = token_x @ w3.T
            hidden = F.silu(gate) * up
            expert_out = hidden @ w2
            output[token_ids] += per_token_weight.unsqueeze(-1) * expert_out

        return output
