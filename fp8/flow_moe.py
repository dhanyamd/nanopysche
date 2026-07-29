"""Re-exported from ``fp8_flow_moe`` for backwards compatibility."""

from fp8_flow_moe.flow_moe import (  # noqa: F401
    FP8FlowMoECompute,
    FP8FlowMoEConfig,
    FP8_BLOCK_SIZE,
    dequantize_from_fp8,
    fp8_grouped_gemm,
    fused_permute_pad,
    fused_silu_multiply_quantize,
    fused_swiglu_quantize,
    fused_unpermute_unpad,
    quantize_to_fp8,
    scaling_aware_transpose,
)
