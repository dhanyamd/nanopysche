"""Generic parallelism hooks for arbitrary ``nn.Module`` models.

Custom models get full TP/CP/PP/EP/FSDP/compile support automatically via
:func:`ensure_parallel_hooks`, which attaches ``apply_*`` methods that delegate
to submodule-native hooks (Attention, FeedForward, MoEBase) or generic fallbacks.

Models with a layer stack (``.blocks``, ``.layers``, ``.h``) get the best results.
"""

from __future__ import annotations

import logging
import types
from typing import Callable, Iterator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from nanopsyche.model.protocol import PARALLEL_HOOKS, has_hook

log = logging.getLogger(__name__)

LAYER_STACK_ATTRS: tuple[str, ...] = ("blocks", "layers", "layer", "h")
EMBED_ATTRS: tuple[str, ...] = ("embeddings", "embed", "wte", "token_embedding")
HEAD_ATTRS: tuple[str, ...] = ("lm_head", "head", "output", "proj")


def find_layer_stack(model: nn.Module) -> Optional[tuple[str, nn.ModuleList | nn.ModuleDict]]:
    """Return ``(attr_name, container)`` for the model's repeatable layer stack."""
    for attr in LAYER_STACK_ATTRS:
        container = getattr(model, attr, None)
        if isinstance(container, (nn.ModuleList, nn.ModuleDict)):
            if len(container) > 0:
                return attr, container
    return None


def layer_count(model: nn.Module) -> int:
    found = find_layer_stack(model)
    if found is not None:
        return len(found[1])
    return 0


def iter_layers(model: nn.Module) -> Iterator[nn.Module]:
    found = find_layer_stack(model)
    if found is None:
        return iter(())
    container = found[1]
    if isinstance(container, nn.ModuleDict):
        for key in sorted(container.keys(), key=lambda k: int(k) if str(k).isdigit() else k):
            yield container[key]
    else:
        yield from container


def _find_attr(model: nn.Module, names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        if getattr(model, name, None) is not None:
            return name
    return None


def _is_builtin_parallel_module(module: nn.Module) -> bool:
    """True if module ships its own production parallel hooks."""
    from nanopsyche.model.attention import Attention
    from nanopsyche.model.feed_forward import FeedForward
    from nanopsyche.model.moe import MoEBase
    from nanopsyche.model.transformer import Transformer, TransformerBlock

    return isinstance(
        module,
        (Attention, FeedForward, MoEBase, TransformerBlock, Transformer),
    )


def _apply_tp_to_linears(module: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Megatron-style TP on a module's direct ``nn.Linear`` children."""
    tp_size = tp_mesh.size()
    if tp_size <= 1:
        return

    tp_group = tp_mesh.get_group()
    rank = dist.get_rank(tp_group)
    linears = [(name, child) for name, child in module.named_children() if isinstance(child, nn.Linear)]
    if not linears:
        return

    for idx, (name, linear) in enumerate(linears):
        weight = linear.weight
        if idx < len(linears) - 1:
            # Colwise: shard output dimension
            if weight.shape[0] % tp_size != 0:
                log.warning(
                    "Skipping TP colwise on %s.%s: out_features %d not divisible by tp_size %d",
                    type(module).__name__,
                    name,
                    weight.shape[0],
                    tp_size,
                )
                continue
            local = weight.view(tp_size, -1, weight.shape[-1])[rank].contiguous()
            linear.weight = nn.Parameter(local)
            if linear.bias is not None:
                linear.bias = nn.Parameter(
                    linear.bias.view(tp_size, -1)[rank].contiguous()
                )
        else:
            # Rowwise: shard input dimension; forward all-reduces in Megatron MLP
            if weight.shape[1] % tp_size != 0:
                log.warning(
                    "Skipping TP rowwise on %s.%s: in_features %d not divisible by tp_size %d",
                    type(module).__name__,
                    name,
                    weight.shape[1],
                    tp_size,
                )
                continue
            local_width = weight.shape[1] // tp_size
            linear.weight = nn.Parameter(
                weight[:, rank * local_width : (rank + 1) * local_width].contiguous()
            )
            _wrap_linear_with_tp_allreduce(linear, tp_group)


def _wrap_linear_with_tp_allreduce(linear: nn.Linear, tp_group: dist.ProcessGroup) -> None:
    """All-reduce rowwise-linear output across TP ranks."""
    from nanopsyche.model.tp_utils import differentiable_all_reduce

    original_forward = linear.forward

    def forward_with_ar(x: torch.Tensor) -> torch.Tensor:
        out = original_forward(x)
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            out = differentiable_all_reduce(out, group=tp_group)
        return out

    linear.forward = forward_with_ar  # type: ignore[method-assign]


def apply_generic_tp(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Apply tensor parallelism to any model."""
    if tp_mesh.size() <= 1:
        return

    seen: set[int] = set()
    for module in model.modules():
        if id(module) in seen or module is model:
            continue
        if _is_builtin_parallel_module(module) and has_hook(module, "apply_tp"):
            module.apply_tp(tp_mesh)
            seen.add(id(module))
            for child in module.modules():
                seen.add(id(child))

    for layer in iter_layers(model):
        if id(layer) in seen:
            continue
        if _is_builtin_parallel_module(layer) and has_hook(layer, "apply_tp"):
            layer.apply_tp(tp_mesh)
            continue
        _apply_tp_to_linears(layer, tp_mesh)

    model._tp_enabled = True  # type: ignore[attr-defined]
    log.info("Applied generic TP to %s", type(model).__name__)


def apply_generic_cp(
    model: nn.Module,
    cp_mesh: DeviceMesh,
    *,
    ring=None,
    uly=None,
) -> None:
    """Apply context parallelism to attention modules inside any model."""
    applied = False
    for module in model.modules():
        if has_hook(module, "apply_cp") and _is_builtin_parallel_module(module):
            module.apply_cp(cp_mesh, ring=ring, uly=uly)
            applied = True
        elif hasattr(module, "w_q") and has_hook(module, "apply_cp"):
            module.apply_cp(cp_mesh, ring=ring, uly=uly)
            applied = True

    if not applied:
        log.warning(
            "Generic CP: no Attention modules found in %s — CP requires attention layers.",
            type(model).__name__,
        )
    else:
        model._cp_enabled = True  # type: ignore[attr-defined]


def apply_generic_pp(model: nn.Module, pp_mesh: DeviceMesh) -> None:
    """Mark model as PP-enabled; runtime uses :func:`split_model`."""
    model._pp_enabled = True  # type: ignore[attr-defined]
    model._pp_mesh = pp_mesh  # type: ignore[attr-defined]
    log.info("Applied generic PP marker to %s (degree=%d)", type(model).__name__, pp_mesh.size())


def apply_generic_ep(model: nn.Module, ep_mesh: DeviceMesh) -> None:
    """Apply expert parallelism to any ``MoEBase`` modules in the model."""
    from nanopsyche.model.moe import ExpertParallelConfig, MoEBase

    ep_config = ExpertParallelConfig(
        ep_size=ep_mesh.size(),
        ep_group=ep_mesh.get_group(),
    )
    count = 0
    for module in model.modules():
        if isinstance(module, MoEBase):
            module.ep = ep_config
            count += 1

    if count == 0:
        log.warning(
            "Generic EP: no MoEBase modules in %s — EP has no effect.",
            type(model).__name__,
        )
    else:
        model._ep_enabled = True  # type: ignore[attr-defined]
        log.info("Applied generic EP to %d MoE layer(s) in %s", count, type(model).__name__)


def apply_generic_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: Optional[torch.dtype] = None,
    reduce_dtype: torch.dtype = torch.float32,
) -> None:
    """Per-layer FSDP for models with a layer stack, else root wrap."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype or torch.bfloat16,
        reduce_dtype=reduce_dtype,
    )

    for layer in iter_layers(model):
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)

    embed_attr = _find_attr(model, EMBED_ATTRS)
    if embed_attr is not None:
        fully_shard(getattr(model, embed_attr), mesh=dp_mesh, mp_policy=mp_policy)

    head_attr = _find_attr(model, HEAD_ATTRS)
    if head_attr is not None:
        head = getattr(model, head_attr)
        if isinstance(head, nn.Linear):
            fully_shard(head, mesh=dp_mesh, reshard_after_forward=False, mp_policy=mp_policy)

    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
    model._fsdp_enabled = True  # type: ignore[attr-defined]
    log.info("Applied generic FSDP to %s", type(model).__name__)


def apply_generic_compile(model: nn.Module) -> None:
    """Compile each layer in the stack, or the root module."""
    compiled_any = False
    found = find_layer_stack(model)
    if found is not None:
        attr, container = found
        if isinstance(container, nn.ModuleDict):
            for key, layer in container.items():
                container[key] = torch.compile(layer)  # type: ignore[index]
                compiled_any = True
        else:
            for i, layer in enumerate(container):
                container[i] = torch.compile(layer)
                compiled_any = True

    if not compiled_any:
        log.info("No layer stack on %s — use root torch.compile", type(model).__name__)
        return

    model._compile_enabled = True  # type: ignore[attr-defined]
    log.info("Applied per-layer torch.compile to %s", type(model).__name__)


def init_generic_weights(
    model: nn.Module,
    *,
    device: Optional[torch.device] = None,
    world_mesh: Optional[DeviceMesh] = None,
) -> None:
    """Standard init for embeddings and linear layers."""
    del world_mesh
    if device is not None:
        model.to(device)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


def _bind(model: nn.Module, name: str, fn: Callable) -> None:
    setattr(model, name, types.MethodType(fn, model))


def ensure_parallel_hooks(model: nn.Module) -> nn.Module:
    """Attach generic ``apply_*`` hooks to models that lack them.

    Built-in :class:`~nanopsyche.model.transformer.Transformer` is unchanged.
    Custom factory models are upgraded automatically.
    """
    from nanopsyche.model.transformer import Transformer

    if isinstance(model, Transformer):
        return model

    if getattr(model, "_nanopsyche_parallel_hooks", False):
        return model

    generic_map = {
        "apply_tp": apply_generic_tp,
        "apply_cp": apply_generic_cp,
        "apply_pp": apply_generic_pp,
        "apply_ep": apply_generic_ep,
        "apply_fsdp": apply_generic_fsdp,
        "apply_compile": apply_generic_compile,
        "init_weights": init_generic_weights,
    }

    attached = []
    for hook_name, fn in generic_map.items():
        if not has_hook(model, hook_name):
            _bind(model, hook_name, fn)
            attached.append(hook_name)

    if attached:
        log.info(
            "Attached generic parallel hooks to %s: %s",
            type(model).__name__,
            ", ".join(attached),
        )

    model._nanopsyche_parallel_hooks = True  # type: ignore[attr-defined]
    return model


def missing_parallel_hooks(model: nn.Module) -> list[str]:
    """Return standard hook names not present on ``model``."""
    return [name for name in PARALLEL_HOOKS if not has_hook(model, name)]
