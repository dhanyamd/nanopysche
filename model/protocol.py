"""Optional model interface for nanopsyche parallelism hooks.

Custom models work out of the box for training if ``forward`` returns a dict
with ``"loss"`` when ``labels`` are passed. Advanced parallelism (TP/CP/PP/EP)
requires optional ``apply_*`` methods — otherwise those modes are skipped with
a warning and generic paths (FP8, activation checkpointing, root FSDP) still apply.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import torch.nn as nn

log = logging.getLogger(__name__)

PARALLEL_HOOKS: tuple[str, ...] = (
    "apply_pp",
    "apply_tp",
    "apply_cp",
    "apply_ep",
    "apply_fsdp",
    "apply_compile",
    "init_weights",
)

MOE_HOOKS: tuple[str, ...] = ("replace_block_with_moe_hybrid",)


def has_hook(model: nn.Module, name: str) -> bool:
    return callable(getattr(model, name, None))


def unsupported_parallelism(
    model: nn.Module,
    *,
    tp: bool = False,
    cp: bool = False,
    pp: bool = False,
    ep: bool = False,
    fsdp: bool = False,
    compile_model: bool = False,
) -> list[str]:
    """Return parallelism modes requested but not supported by ``model``."""
    checks: list[tuple[bool, str, str]] = [
        (tp, "apply_tp", "tensor parallelism (TP)"),
        (cp, "apply_cp", "context parallelism (CP)"),
        (pp, "apply_pp", "pipeline parallelism (PP)"),
        (ep, "apply_ep", "expert parallelism (EP)"),
        (fsdp, "apply_fsdp", "model-specific FSDP wrapping"),
        (compile_model, "apply_compile", "per-block torch.compile"),
    ]
    missing: list[str] = []
    for enabled, hook, label in checks:
        if enabled and not has_hook(model, hook):
            missing.append(label)
    return missing


def warn_unsupported_parallelism(model: nn.Module, missing: Sequence[str]) -> None:
    if not missing:
        return
    name = type(model).__name__
    modes = ", ".join(missing)
    log.warning(
        "%s does not implement hooks for %s — those modes will use generic "
        "fallbacks or be skipped. Implement apply_* methods for full support, "
        "or use the built-in Transformer (omit --model-factory).",
        name,
        modes,
    )


def model_display_name(model: nn.Module, preset: Optional[str] = None) -> str:
    if preset:
        return preset
    return type(model).__name__


def count_parameters(model: nn.Module) -> dict[str, int]:
    num_params = sum(p.numel() for p in model.parameters())
    embed_params = 0
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "weight"):
        embed_params = model.embeddings.weight.numel()
    elif hasattr(model, "embed") and hasattr(model.embed, "weight"):
        embed_params = model.embed.weight.numel()
    return {
        "num_params": num_params,
        "num_non_embed_params": num_params - embed_params,
    }
