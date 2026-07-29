"""nanopsyche.fp8 — Float8 training support.

This module re-exports from the standalone ``fp8_flow_moe`` package
for the FP8-Flow-MoE operators, and provides torchao-based FP8 training
for dense layers via ``training.py``.
"""

from fp8_flow_moe import (
    FP8FlowMoEConfig,
    FP8FlowMoECompute,
    FP8RecipeConfig,
    FP8RecipeType,
    BlockScalingGranularity,
    quantize_to_fp8,
    dequantize_from_fp8,
    scaling_aware_transpose,
    fused_swiglu_quantize,
    fused_silu_multiply_quantize,
    fused_permute_pad,
    fused_unpermute_unpad,
    fp8_grouped_gemm,
)

from fp8_flow_moe.flow_moe import FP8_BLOCK_SIZE

from nanopsyche.fp8.training import FP8Config, apply_fp8

__all__ = [
    "FP8Config",
    "apply_fp8",
    "FP8FlowMoEConfig",
    "FP8FlowMoECompute",
    "FP8RecipeConfig",
    "FP8RecipeType",
    "BlockScalingGranularity",
    "quantize_to_fp8",
    "dequantize_from_fp8",
    "scaling_aware_transpose",
    "fused_swiglu_quantize",
    "fused_silu_multiply_quantize",
    "fused_permute_pad",
    "fused_unpermute_unpad",
    "fp8_grouped_gemm",
    "FP8_BLOCK_SIZE",
]
