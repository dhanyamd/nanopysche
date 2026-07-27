from __future__ import annotations

"""Float8 (FP8) training — reduced precision for faster matmul.

FP8 uses 8-bit floating point for forward and backward passes.
Two FP8 formats exist:
    - E4M3: 4-bit exponent, 3-bit mantissa (range: ~1e-7 to 448)
    - E5M2: 5-bit exponent, 2-bit mantissa (range: ~1e-14 to 57344)

Production usage (OLMo-core):
    - Forward activations: E4M3 (better precision for activations)
    - Backward gradients: E5M2 (better range for gradients)
    - Master weights: FP32 or BF16 (unchanged)
    - Loss scaling: dynamic to prevent underflow

Memory savings:
    - BF16: 2 bytes per element
    - FP8:  1 byte per element
    - 2x memory reduction for activations

Speed:
    - H100 FP8 Tensor Core: ~2x BF16 throughput
    - H100 peak FP8: ~1979 TFLOPS vs ~989 TFLOPS BF16

OLMo-core integration:
    - Uses torchao's Float8LinearConfig
    - FSDP2 with FP8: Float8ColwiseParallel, Float8RowwiseParallel
    - Weight recompute in backward for memory efficiency

Reference: Micikevicius et al. 2022 (FP8 Formats for DDL)
           PyTorch torchao.float8
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

# Try to import torchao FP8
_TORCHAO_AVAILABLE = False
try:
    from torchao.float8 import (
        Float8LinearConfig,
        convert_to_float8_training,
        precompute_float8_dynamic_scale_for_fsdp,
    )

    _TORCHAO_AVAILABLE = True
except ImportError:
    pass


@dataclass
class FP8Config:
    """Configuration for FP8 training via torchao.

    All fields map directly to torchao's Float8LinearConfig when available.

    Attributes:
        enabled: whether to use FP8
        forward_dtype: FP8 format for forward activations ("float8_e4m3fn" or "float8_e5m2")
        backward_dtype: FP8 format for backward gradients ("float8_e5m2" typically)
        recipe_name: scaling recipe ("max" for MaxOptimal, "delayed" for DelayedScaling)
        force_recompute_fp8_weight_in_bwd: recompute weight in backward to save memory
        cast_input_text: whether to cast input to FP8 before matmul
        filter_fn: optional module filter — only convert matching modules to FP8
    """

    enabled: bool = False
    forward_dtype: str = "float8_e4m3fn"
    backward_dtype: str = "float8_e5m2"
    recipe_name: str = "max"
    force_recompute_fp8_weight_in_bwd: bool = True
    cast_input_text: bool = True
    filter_fn: Optional[str] = None

    def build_torchao_config(self) -> Optional[object]:
        """Build torchao Float8LinearConfig from this config."""
        if not _TORCHAO_AVAILABLE:
            return None
        if not self.enabled:
            return None

        return Float8LinearConfig(
            forward_dtype=self.forward_dtype,
            backward_dtype=self.backward_dtype,
            recipe_name=self.recipe_name,
            force_recompute_fp8_weight_in_bwd=self.force_recompute_fp8_weight_in_bwd,
        )


def apply_fp8(
    model: nn.Module,
    config: Optional[FP8Config] = None,
) -> nn.Module:
    """Apply FP8 training to a model.

    When torchao is available, uses convert_to_float8_training() which:
        - Converts Linear layers to FP8 computation
        - Handles dynamic scaling of activations
        - Supports weight recompute in backward
        - Integrates with FSDP2 (Float8ColwiseParallel, Float8RowwiseParallel)

    When torchao is unavailable (e.g. macOS), this is a no-op.

    :param model: model to convert
    :param config: FP8 configuration (default: disabled)
    :returns: model with FP8 applied (or unchanged if torchao unavailable)
    """
    if config is None or not config.enabled:
        return model

    if not _TORCHAO_AVAILABLE:
        print(
            "[FP8] torchao not available — skipping FP8 conversion. "
            "Install with: pip install torchao"
        )
        return model

    torchao_config = config.build_torchao_config()
    model = convert_to_float8_training(model, config=torchao_config)
    return model


def precompute_fp8_dynamic_scales(model: nn.Module) -> None:
    """Precompute FP8 dynamic scales for FSDP integration.

    Must be called after FSDP sharding but before the forward pass.
    This computes per-parameter scales based on weight magnitudes.

    :param model: model with FP8 applied
    """
    if not _TORCHAO_AVAILABLE:
        return

    precompute_float8_dynamic_scale_for_fsdp(model)
