"""Model factory — build a user model or fall back to the built-in Transformer.

Usage (CLI)::

    # Built-in architecture (default)
    nanopsyche train --model 125m

    # Custom model factory
    nanopsyche train --model-factory my_project.models:build_model

The factory callable receives a :class:`ModelBuildContext` and must return either:

    - ``nn.Module`` with ``forward(input_ids, labels=...) -> {"loss": ...}``
    - ``(model, optimizer, scheduler)`` — optimizer/scheduler are optional extras

Reference: OLMo-core model + train_module separation pattern.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

ModelFactoryResult = Union[
    nn.Module,
    tuple[nn.Module, Any],
    tuple[nn.Module, Any, Any],
]


@dataclass
class ModelBuildContext:
    """Arguments passed to a custom model factory."""

    vocab_size: int
    seq_len: int
    device: str
    model_preset: str
    preset: Optional[dict[str, Any]]
    use_moe: bool
    num_experts: int
    moe_top_k: int
    fp8_recipe: str
    dtype: torch.dtype
    fp8_gemm_threshold: int = 1000000


def load_factory(spec: str) -> Callable[[ModelBuildContext], ModelFactoryResult]:
    """Load ``module.path:callable`` as a model factory."""
    if ":" not in spec:
        raise ValueError(
            f"Invalid --model-factory {spec!r}. Expected format 'module.path:callable'."
        )
    module_path, attr = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    factory = getattr(module, attr, None)
    if factory is None or not callable(factory):
        raise ValueError(
            f"Factory {spec!r} not found or not callable. "
            f"Define a function like `def build_model(ctx): ...` in {module_path}."
        )
    return factory


def build_default_transformer(ctx: ModelBuildContext) -> nn.Module:
    """Build the built-in nanopsyche Transformer from a size preset."""
    from nanopsyche.config import TransformerConfig

    if ctx.preset is None:
        raise ValueError("preset is required for the default Transformer builder")

    config = TransformerConfig(
        vocab_size=ctx.vocab_size,
        d_model=ctx.preset["d_model"],
        n_layers=ctx.preset["n_layers"],
        n_heads=ctx.preset["n_heads"],
        n_kv_heads=ctx.preset.get("n_kv_heads"),
        ffn_hidden=ctx.preset.get("ffn_hidden"),
        max_seq_len=ctx.seq_len,
        rope_base=500000.0,
        use_moe=ctx.use_moe,
        num_experts=ctx.num_experts if ctx.use_moe else 0,
        moe_top_k=ctx.moe_top_k if ctx.use_moe else 0,
        fp8_flow_moe=(ctx.fp8_recipe == "flow_moe"),
        fp8_recipe=ctx.fp8_recipe,
        fp8_gemm_threshold=ctx.fp8_gemm_threshold,
    )
    return config.build()


def apply_moe_hybrid(model: nn.Module, ctx: ModelBuildContext) -> nn.Module:
    """Replace dense FFN blocks with MoE hybrid blocks (built-in Transformer only)."""
    if not ctx.use_moe:
        return model

    if not hasattr(model, "replace_block_with_moe_hybrid"):
        log.warning(
            "use_moe=True but %s has no replace_block_with_moe_hybrid(); skipping MoE.",
            type(model).__name__,
        )
        return model

    if ctx.preset is None:
        log.warning("MoE hybrid conversion requires a model preset with layer count.")
        return model

    from nanopsyche.model.moe import MoEBase as MoEBaseCLS

    preset = ctx.preset
    ffn_hidden = preset.get("ffn_hidden") or int(8 / 3 * preset["d_model"])
    ffn_hidden = ((ffn_hidden + 255) // 256) * 256
    moe_interval = 2

    for block_idx in range(preset["n_layers"]):
        if block_idx % moe_interval == 0:
            moe = MoEBaseCLS(
                d_model=preset["d_model"],
                hidden_size=ffn_hidden,
                num_experts=ctx.num_experts,
                top_k=ctx.moe_top_k,
                fp8_flow_moe=(ctx.fp8_recipe == "flow_moe"),
                fp8_recipe=ctx.fp8_recipe,
                fp8_gemm_threshold=ctx.fp8_gemm_threshold,
            ).to(ctx.device)
            model.replace_block_with_moe_hybrid(block_idx, moe)

    return model


def build_model(
    ctx: ModelBuildContext,
    *,
    factory_spec: Optional[str] = None,
) -> tuple[nn.Module, Optional[Any], Optional[Any]]:
    """Build a model from a custom factory or the default Transformer.

    :returns: ``(model, optional_optimizer, optional_scheduler)``
    """
    optimizer = None
    scheduler = None

    if factory_spec:
        log.info("Building model from factory %s", factory_spec)
        result = load_factory(factory_spec)(ctx)
    else:
        log.info("Building default Transformer preset %s", ctx.model_preset)
        result = build_default_transformer(ctx)

    if isinstance(result, tuple):
        model = result[0]
        if len(result) > 1:
            optimizer = result[1]
        if len(result) > 2:
            scheduler = result[2]
    else:
        model = result

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Model factory must return nn.Module, got {type(model).__name__}"
        )

    model = apply_moe_hybrid(model, ctx)

    if factory_spec:
        from nanopsyche.model.parallel_adapter import ensure_parallel_hooks

        model = ensure_parallel_hooks(model)

    return model, optimizer, scheduler
