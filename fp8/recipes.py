from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FP8RecipeType(str, Enum):
    NONE = "none"
    BLOCKWISE = "blockwise"
    MXFP8 = "mxfp8"
    FLOW_MOE = "flow_moe"


class BlockScalingGranularity(str, Enum):
    PER_TENSOR = "per_tensor"
    PER_ROW_1x128 = "per_row_1x128"
    PER_BLOCK_128x128 = "per_block_128x128"
    PER_BLOCK_1x32 = "per_block_1x32"


@dataclass
class FP8RecipeConfig:
    enabled: bool = False
    recipe_type: FP8RecipeType = FP8RecipeType.NONE
    forward_dtype: str = "float8_e4m3fn"
    backward_dtype: str = "float8_e5m2"

    act_block_scaling: BlockScalingGranularity = BlockScalingGranularity.PER_ROW_1x128
    weight_block_scaling: BlockScalingGranularity = (
        BlockScalingGranularity.PER_BLOCK_128x128
    )

    moe_fp8_comm: bool = False
    """If True, keep FP8 during all-to-all dispatch/combine (DeepSeek-V3 style)."""

    force_wgrad_bf16: bool = True
    """If True, compute weight gradient in BF16 (avoids double quantization error)."""

    scaling_aware_transpose: bool = False
    """If True, use FP8 scaling-aware transpose instead of deq→trans→req."""

    fused_permute_pad: bool = False
    """If True, fuse permute+pad and unpermute+unpad."""

    fused_swiglu_quantize: bool = False
    """If True, fuse SwiGLU activation + FP8 quantization."""

    def validate(self) -> bool:
        """Check recipe consistency."""
        if not self.enabled:
            return True
        if self.recipe_type == FP8RecipeType.FLOW_MOE:
            if not self.scaling_aware_transpose:
                return False
            if not self.force_wgrad_bf16:
                return False
        return True

    @classmethod
    def blockwise_deepseek_v3(cls) -> FP8RecipeConfig:
        return cls(
            enabled=True,
            recipe_type=FP8RecipeType.BLOCKWISE,
            act_block_scaling=BlockScalingGranularity.PER_ROW_1x128,
            weight_block_scaling=BlockScalingGranularity.PER_BLOCK_128x128,
            moe_fp8_comm=True,
            force_wgrad_bf16=True,
            scaling_aware_transpose=False,
        )

    @classmethod
    def mxfp8_blackwell(cls) -> FP8RecipeConfig:
        return cls(
            enabled=True,
            recipe_type=FP8RecipeType.MXFP8,
            act_block_scaling=BlockScalingGranularity.PER_BLOCK_1x32,
            weight_block_scaling=BlockScalingGranularity.PER_BLOCK_1x32,
            moe_fp8_comm=True,
            force_wgrad_bf16=True,
            scaling_aware_transpose=False,
        )

    @classmethod
    def flow_moe(cls) -> FP8RecipeConfig:
        return cls(
            enabled=True,
            recipe_type=FP8RecipeType.FLOW_MOE,
            act_block_scaling=BlockScalingGranularity.PER_ROW_1x128,
            weight_block_scaling=BlockScalingGranularity.PER_BLOCK_128x128,
            moe_fp8_comm=True,
            force_wgrad_bf16=True,
            scaling_aware_transpose=True,
            fused_permute_pad=True,
            fused_swiglu_quantize=True,
        )
