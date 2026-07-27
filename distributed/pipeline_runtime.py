from __future__ import annotations

"""Pipeline Parallelism Runtime — wraps torch.distributed.pipelining.

Provides:
    split_model(): splits a Transformer into pipeline stages via deepcopy + layer deletion
    PipelineSchedule: thin wrapper around PyTorch schedule classes (GPipe, 1F1B, Interleaved1F1B, etc.)
    PipelineRunner: orchestrates training with PP (forward/backward through schedule.step())

Reference: OLMo-core src/olmo_core/distributed/parallel/pipeline_parallel.py
           OLMo-core src/olmo_core/train/train_module/transformer/pipeline_train_module.py
"""

import copy
import logging
import math
from dataclasses import dataclass
import enum
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, enum.Enum):
        pass


from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)

log = logging.getLogger(__name__)


class PipelineScheduleType(StrEnum):
    """Supported pipeline schedule types.

    Maps to PyTorch's built-in schedule classes via get_schedule_class().
    """

    single_1F1B = "1F1B"
    interleaved_1F1B = "Interleaved1F1B"
    gpipe = "GPipe"
    looped_bfs = "LoopedBFS"
    interleaved_zero_bubble = "InterleavedZeroBubble"
    zbv_zero_bubble = "ZBVZeroBubble"


class PipelineSplitStyle(StrEnum):
    """How stages are assigned to ranks for multi-stage schedules."""

    loop = "loop"
    v = "v"


@dataclass
class PipelineParallelConfig:
    """Configuration for pipeline parallelism.

    :param degree: the PP degree (number of pipeline stages).
    :param schedule: schedule type name.
    :param split_points: explicit layer indices where to split. If None, auto-computed.
    :param num_microbatches: number of microbatches. If None, defaults to pp degree.
    """

    degree: int
    schedule: str = "1F1B"
    split_points: Optional[List[int]] = None
    num_microbatches: Optional[int] = None

    @property
    def is_single_stage(self) -> bool:
        return self.schedule in ("1F1B", "GPipe")

    @property
    def stages_per_rank(self) -> int:
        return 1 if self.is_single_stage else 2

    @property
    def total_stages(self) -> int:
        return self.degree * self.stages_per_rank

    @property
    def split_style(self) -> PipelineSplitStyle:
        if self.schedule == "ZBVZeroBubble":
            return PipelineSplitStyle.v
        return PipelineSplitStyle.loop

    def get_split_points(self, n_layers: int) -> List[int]:
        """Compute layer split points for the given number of layers.

        If explicit split_points are provided, use those.
        Otherwise, distribute layers as evenly as possible across total_stages.
        """
        if self.split_points is not None:
            return self.split_points

        total_stages = self.total_stages
        if total_stages > n_layers:
            raise ValueError(
                f"Total stages ({total_stages}) cannot exceed number of layers ({n_layers})"
            )

        base_interval = n_layers // total_stages
        extra_layers = n_layers % total_stages

        splits: List[int] = []
        current_layer = 0
        for i in range(total_stages - 1):
            if i == 0:
                current_layer += base_interval
            else:
                if extra_layers > 0:
                    current_layer += base_interval + 1
                    extra_layers -= 1
                else:
                    current_layer += base_interval
            splits.append(current_layer)
        return splits

    def stage_ids_this_rank(self, pp_rank: int, num_stages: int) -> Tuple[int, ...]:
        """Which stage indices this rank owns.

        For single-stage schedules: just (pp_rank,).
        For multi-stage schedules: depends on split_style.
        """
        if self.is_single_stage:
            return (pp_rank,)

        stages_per_rank = num_stages // self.degree

        if self.split_style == PipelineSplitStyle.loop:
            # Loop: rank 0 gets stages [0, 4], rank 1 gets [1, 5], etc.
            return tuple(pp_rank + s * self.degree for s in range(stages_per_rank))
        elif self.split_style == PipelineSplitStyle.v:
            # V: rank 0 gets [0, N-1], rank 1 gets [1, N-2], etc.
            stage_v_pairs = list(
                zip(range(self.degree), range(num_stages - 1, self.degree - 1, -1))
            )
            return stage_v_pairs[pp_rank]

        raise ValueError(f"Unknown split style: {self.split_style}")


def split_model(
    model: nn.Module,
    *,
    pp_config: PipelineParallelConfig,
    pp_rank: int,
    device: torch.device,
    pp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[List[PipelineStage], List[nn.Module]]:
    """Split a Transformer model into pipeline stages.

    Each rank gets one or more model chunks (stages). The split is done by
    deepcopying the model and deleting layers outside the stage's range.

    This follows OLMo-core's pattern: deepcopy + layer deletion (not tracer-based).

    :param model: the full Transformer model.
    :param pp_config: pipeline parallelism configuration.
    :param pp_rank: this rank's index within the PP group.
    :param device: device to place stages on.
    :param pp_group: process group for PP communication.
    :returns: (list of PipelineStage, list of model chunks).
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")

    # Find the number of transformer layers
    if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleDict):
        n_layers = len(model.blocks)
    else:
        raise ValueError(
            "Model must have a 'blocks' attribute (nn.ModuleDict of transformer blocks)"
        )

    split_points = pp_config.get_split_points(n_layers)
    num_stages = len(split_points) + 1

    if pp_group is None:
        pp_group = dist.group.WORLD

    stages: List[PipelineStage] = []
    model_chunks: List[nn.Module] = []

    for stage_idx in pp_config.stage_ids_this_rank(pp_rank, num_stages):
        start_layer = split_points[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = split_points[stage_idx] if stage_idx < num_stages - 1 else None

        stage, model_chunk = _build_stage(
            model,
            stage_idx=stage_idx,
            num_stages=num_stages,
            start_layer=start_layer,
            stop_layer=stop_layer,
            is_first=(stage_idx == 0),
            is_last=(stage_idx == num_stages - 1),
            device=device,
            pp_group=pp_group,
        )
        stages.append(stage)
        model_chunks.append(model_chunk)

    return stages, model_chunks


def _build_stage(
    model: nn.Module,
    *,
    stage_idx: int,
    num_stages: int,
    start_layer: Optional[int],
    stop_layer: Optional[int],
    is_first: bool,
    is_last: bool,
    device: torch.device,
    pp_group: dist.ProcessGroup,
) -> Tuple[PipelineStage, nn.Module]:
    """Build a single pipeline stage from the full model.

    Deepcopy the model, then strip layers outside [start_layer, stop_layer).
    Remove embeddings from non-first stages and lm_head from non-last stages.
    """
    model_chunk = copy.deepcopy(model)

    # Strip embeddings on non-first stages
    if not is_first:
        model_chunk.embeddings = None
        if hasattr(model_chunk, "embedding_norm"):
            model_chunk.embedding_norm = None

    # Keep only the contiguous layer range [start_layer, stop_layer)
    if hasattr(model_chunk, "blocks") and isinstance(model_chunk.blocks, nn.ModuleDict):
        drop_layers = start_layer is not None
        blocks_to_keep = []
        for block_idx in range(len(model_chunk.blocks)):
            block_key = str(block_idx)
            if block_idx == start_layer:
                drop_layers = False
            if block_idx == stop_layer:
                drop_layers = True
            if not drop_layers:
                blocks_to_keep.append((block_key, model_chunk.blocks[block_key]))
            else:
                del model_chunk.blocks[block_key]

        # Re-index blocks if needed (not strictly necessary, but clean)
        # Keep original keys for compatibility with apply_tp/apply_fsdp

    # Strip LM head on non-last stages
    if not is_last:
        model_chunk.lm_head = None

    # Move to device
    model_chunk = model_chunk.to(device)

    stage = PipelineStage(
        model_chunk,
        stage_idx,
        num_stages,
        device,
        group=pp_group,
    )

    return stage, model_chunk


class PipelineSchedule:
    """Thin wrapper around PyTorch pipeline schedule classes.

    Usage:
        schedule = PipelineSchedule(
            stages=stages,
            pp_config=pp_config,
            loss_fn=cross_entropy_loss,
        )
        losses = schedule.step(input_ids, target=labels)
    """

    def __init__(
        self,
        *,
        stages: List[PipelineStage],
        pp_config: PipelineParallelConfig,
        loss_fn: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        num_microbatches: Optional[int] = None,
    ):
        self.stages = stages
        self.pp_config = pp_config
        self.loss_fn = loss_fn

        if num_microbatches is None:
            num_microbatches = pp_config.num_microbatches or len(stages)
        self.num_microbatches = num_microbatches

        schedule_class = get_schedule_class(pp_config.schedule)

        if pp_config.is_single_stage:
            if len(stages) != 1:
                raise ValueError(
                    f"Single-stage schedule '{pp_config.schedule}' requires exactly 1 stage, "
                    f"got {len(stages)}"
                )
            self._schedule = schedule_class(
                stages[0],
                n_microbatches=self.num_microbatches,
                loss_fn=self.loss_fn,
            )
        else:
            self._schedule = schedule_class(
                stages,
                n_microbatches=self.num_microbatches,
                loss_fn=self.loss_fn,
            )

    def step(self, *args, target=None, **kwargs):
        """Run one training step through the pipeline.

        :returns: (output, losses) tuple.
        """
        losses = []
        output = self._schedule.step(*args, target=target, losses=losses, **kwargs)
        loss_tensor = torch.stack(losses) if losses else None
        return output, loss_tensor


def compute_bubble_fraction(schedule_type: str, p: int, m: int, v: int = 1) -> float:
    """Compute the theoretical bubble fraction for a schedule.

    :param schedule_type: "gpipe", "1f1b", "Interleaved1F1B", "LoopedBFS", etc.
    :param p: number of pipeline stages.
    :param m: number of microbatches.
    :param v: virtual pipeline chunks (for interleaved schedules).
    :returns: bubble fraction (0.0 = no bubble, 1.0 = all bubble).
    """
    if schedule_type in ("GPipe", "1F1B"):
        return (p - 1) / m
    elif schedule_type in ("Interleaved1F1B", "LoopedBFS", "InterleavedZeroBubble"):
        return (p - 1) / (v * m)
    elif schedule_type == "ZBVZeroBubble":
        return 0.0
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def clip_grad_norm_pp(
    model_parts: List[nn.Module],
    pp_group: dist.ProcessGroup,
    max_grad_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Clip gradient norms across pipeline-parallel ranks.

    Different ranks hold different layers, so we need to all-reduce the
    total norm across the PP group before clipping.

    :param model_parts: list of model chunks on this rank.
    :param pp_group: PP process group.
    :param max_grad_norm: maximum gradient norm.
    :param norm_type: norm type (default: 2.0).
    :returns: total gradient norm before clipping.
    """
    parameters = [p for m in model_parts for p in m.parameters()]
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return torch.tensor(0.0)

    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type=norm_type, foreach=False
    )

    # All-reduce across PP group
    if dist.get_world_size(pp_group) > 1:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_group)
        else:
            total_norm = total_norm**norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_group)
            total_norm = total_norm ** (1.0 / norm_type)

    # Clip
    torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm)

    return total_norm
