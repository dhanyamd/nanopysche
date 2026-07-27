"""Parallelize model — THE KEY COMPOSITION FUNCTION.

Applies all parallelism transforms to the model in the correct order.
This is the nanopsyche equivalent of OLMo-core's
    src/olmo_core/train/train_module/transformer/common.py::parallelize_model()

Application order (from OLMo-core):
    1. Pipeline Parallelism (split model into stages)
    2. FP8 (swap linear layers to Float8)
    3. Context Parallelism (shard sequence across CP ranks)
    4. Tensor Parallelism (shard weights along hidden dim)
    5. Expert Parallelism (MoE only)
    6. Activation Checkpointing
    7. torch.compile (per block)
    8. Data Parallelism (FSDP/DDP — MUST BE LAST)
    9. Weight initialization (materialize from meta)

Reference: OLMo-core src/olmo_core/train/train_module/transformer/common.py
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from nanopsyche.parallel import (
    get_dp_model_mesh,
    get_tp_mesh,
    get_cp_mesh,
    get_pp_mesh,
)
from nanopsyche.distributed.tensor_parallel import TensorParallelConfig
from nanopsyche.distributed.context_parallel import ContextParallelConfig
from nanopsyche.distributed.pipeline_parallel import PipelineParallelConfig
from nanopsyche.distributed.fsdp import (
    DataParallelConfig,
    apply_activation_checkpointing,
)
from nanopsyche.fp8.training import FP8Config, apply_fp8

log = logging.getLogger(__name__)


def parallelize_model(
    model: nn.Module,
    *,
    world_mesh: DeviceMesh,
    device: Optional[torch.device] = None,
    max_sequence_length: Optional[int] = None,
    tp_config: Optional[TensorParallelConfig] = None,
    cp_config: Optional[ContextParallelConfig] = None,
    pp_config: Optional[PipelineParallelConfig] = None,
    dp_config: Optional[DataParallelConfig] = None,
    fp8_config: Optional[FP8Config] = None,
    compile_model: bool = False,
    ac_mode: Optional[str] = None,
    ac_block_interval: Optional[int] = None,
) -> nn.Module:
    """Apply all parallelism transforms to the model.

    This is the single entry point for composing all parallelism dimensions.
    The order matters — it matches OLMo-core's parallelize_model().

    :param model: the transformer model.
    :param world_mesh: multi-dimensional DeviceMesh from build_world_mesh().
    :param device: target device for initialization.
    :param max_sequence_length: maximum sequence length (for RoPE warmup).
    :param tp_config: tensor parallelism config.
    :param cp_config: context parallelism config.
    :param pp_config: pipeline parallelism config.
    :param dp_config: data parallelism config.
    :param fp8_config: FP8 training configuration (None to disable).
    :param compile_model: enable torch.compile per block.
    :param ac_mode: activation checkpointing mode ("full", "selected_blocks").
    :param ac_block_interval: for "selected_blocks", checkpoint every N-th.
    :returns: model with all parallelism transforms applied.
    """
    log.info("Parallelizing model...")

    # 1. Pipeline Parallelism (split model into stages)
    if pp_config is not None:
        log.info(f"  Applying PP (degree={pp_config.degree})")
        if hasattr(model, "apply_pp"):
            model.apply_pp(get_pp_mesh(world_mesh))

    # 2. FP8 (swap linear layers to Float8)
    if fp8_config is not None and fp8_config.enabled:
        log.info("  Applying FP8")
        model = apply_fp8(model, config=fp8_config)

    # 3. Context Parallelism (shard sequence across CP ranks)
    if cp_config is not None:
        log.info(f"  Applying CP (degree={cp_config.degree})")
        if hasattr(model, "apply_cp"):
            model.apply_cp(get_cp_mesh(world_mesh))

    # 4. Tensor Parallelism (shard weights along hidden dim)
    if tp_config is not None:
        log.info(f"  Applying TP (degree={tp_config.degree})")
        tp_mesh = get_tp_mesh(world_mesh)
        if hasattr(model, "apply_tp"):
            model.apply_tp(tp_mesh)
        tp_config.maybe_enable_async_tp(tp_mesh)

    # 5. Expert Parallelism (MoE only)
    if hasattr(model, "apply_ep") and world_mesh.mesh_dim_names is not None:
        from nanopsyche.parallel import MeshDimName

        if MeshDimName.ep in world_mesh.mesh_dim_names:
            from nanopsyche.parallel import get_ep_mesh

            log.info("  Applying EP")
            model.apply_ep(get_ep_mesh(world_mesh))

    # 6. Activation Checkpointing
    if ac_mode is not None:
        log.info(f"  Applying AC (mode={ac_mode})")
        apply_activation_checkpointing(
            model, mode=ac_mode, block_interval=ac_block_interval
        )

    # 7. torch.compile
    if compile_model:
        log.info("  Applying torch.compile")
        if hasattr(model, "apply_compile"):
            model.apply_compile()

    # 8. Data Parallelism (FSDP/DDP — MUST BE LAST)
    if dp_config is not None:
        log.info(f"  Applying DP (type={dp_config.name})")
        dp_mesh = get_dp_model_mesh(world_mesh)
        if hasattr(model, "apply_fsdp"):
            model.apply_fsdp(
                dp_mesh=dp_mesh,
                param_dtype=dp_config.param_dtype,
                reduce_dtype=dp_config.reduce_dtype,
            )

    # 9. Weight initialization (materialize from meta)
    if device is not None and hasattr(model, "init_weights"):
        log.info("  Initializing weights")
        model.init_weights(device=device, world_mesh=world_mesh)

    log.info("Model parallelization complete")
    return model
