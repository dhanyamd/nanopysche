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

import torch
import torch.nn as nn


class FP8Config:
    """Configuration for FP8 training.

    Attributes:
        enabled: whether to use FP8
        forward_dtype: FP8 format for forward (E4M3 or E5M2)
        backward_dtype: FP8 format for backward (E5M2 typically)
        recipe: training recipe (" DelayedScaling" or "MaxOptimal")
        force_recompute_fp8_weight_in_bwd: recompute weight in backward to save memory
    """

    def __init__(
        self,
        enabled: bool = False,
        forward_dtype: str = "e4m3",
        backward_dtype: str = "e5m2",
        recipe: str = "DelayedScaling",
        force_recompute_fp8_weight_in_bwd: bool = True,
    ):
        self.enabled = enabled
        self.forward_dtype = forward_dtype
        self.backward_dtype = backward_dtype
        self.recipe = recipe
        self.force_recompute_fp8_weight_in_bwd = force_recompute_fp8_weight_in_bwd


def apply_fp8(
    model: nn.Module,
    config: FP8Config | None = None,
) -> nn.Module:
    """Apply FP8 training to a model.

    This converts linear layers to use FP8 computation. The model
    must be on CUDA for FP8 to work.

    Usage:
        config = FP8Config(enabled=True)
        model = apply_fp8(model, config)

    Production note:
        In real training, use torchao's Float8LinearConfig:
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            config = Float8LinearConfig(
                cast_input_text=True,
                force_recompute_fp8_weight_in_bwd=True,
            )
            convert_to_float8_training(model, config=config)

        This handles:
            - Automatic casting of inputs to FP8 before matmul
            - Dynamic scaling of activations
            - Weight recompute in backward
            - Integration with FSDP2 (Float8ColwiseParallel, Float8RowwiseParallel)
    """
    if config is None or not config.enabled:
        return model

    # In production, use torchao:
    # from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    # convert_to_float8_training(model, config=Float8LinearConfig(...))

    # For learning, we show the manual approach:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Convert to FP8 linear (simplified — production uses torchao)
            module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.bfloat16)

    return model
