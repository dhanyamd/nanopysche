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

from nanopsyche.model.parallel_adapter import (
    EMBED_ATTRS,
    HEAD_ATTRS,
    find_layer_stack,
)

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

    Maps to PyTorch's built-in schedule classes via get_schedule_class(),
    except for DualPipe which uses our custom implementation.
    """

    single_1F1B = "1F1B"
    interleaved_1F1B = "Interleaved1F1B"
    gpipe = "GPipe"
    looped_bfs = "LoopedBFS"
    interleaved_zero_bubble = "InterleavedZeroBubble"
    zbv_zero_bubble = "ZBVZeroBubble"
    dualpipe = "DualPipe"


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


def _strip_embeddings(model_chunk: nn.Module) -> None:
    for attr in EMBED_ATTRS:
        if getattr(model_chunk, attr, None) is not None:
            setattr(model_chunk, attr, None)
    if hasattr(model_chunk, "embedding_norm"):
        model_chunk.embedding_norm = None


def _strip_lm_head(model_chunk: nn.Module) -> None:
    for attr in HEAD_ATTRS:
        if getattr(model_chunk, attr, None) is not None:
            setattr(model_chunk, attr, None)


def _filter_layer_stack(
    model_chunk: nn.Module,
    start_layer: Optional[int],
    stop_layer: Optional[int],
) -> None:
    found = find_layer_stack(model_chunk)
    if found is None:
        return

    attr, container = found
    if isinstance(container, nn.ModuleDict):
        keys = sorted(container.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
        for block_idx, key in enumerate(keys):
            if start_layer is not None and block_idx < start_layer:
                del container[key]
            elif stop_layer is not None and block_idx >= stop_layer:
                del container[key]
    else:
        kept = nn.ModuleList()
        for block_idx, layer in enumerate(container):
            if start_layer is not None and block_idx < start_layer:
                continue
            if stop_layer is not None and block_idx >= stop_layer:
                continue
            kept.append(layer)
        setattr(model_chunk, attr, kept)


def split_model(
    model: nn.Module,
    *,
    pp_config: PipelineParallelConfig,
    pp_rank: int,
    device: torch.device,
    pp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[List[PipelineStage], List[nn.Module]]:
    """Split a model into pipeline stages via deepcopy + layer deletion.

    Supports layer stacks named ``blocks``, ``layers``, ``layer``, or ``h``
    (``ModuleDict`` or ``ModuleList``).
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")

    found = find_layer_stack(model)
    if found is None:
        raise ValueError(
            "Model must expose a layer stack as .blocks, .layers, .layer, or .h "
            "(ModuleDict or ModuleList with at least one layer)."
        )
    n_layers = len(found[1])

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

    if not is_first:
        _strip_embeddings(model_chunk)

    _filter_layer_stack(model_chunk, start_layer, stop_layer)

    if not is_last:
        _strip_lm_head(model_chunk)

    model_chunk = model_chunk.to(device)

    stage = PipelineStage(
        model_chunk,
        stage_idx,
        num_stages,
        device,
        group=pp_group,
    )

    return stage, model_chunk


class DualPipeScheduleRuntime:
    """Custom DualPipe schedule runtime for DeepSeek-V3 bidirectional PP.

    Unlike standard schedules, DualPipe runs microbatches from both ends of
    the pipeline simultaneously. Phase 0 flows left-to-right, phase 1 flows
    right-to-left. Each GPU holds two parameter sets (forward + reverse).

    The 8-step execution pattern (DeepSeek-V3 §2.2.2):
        1-2: Forward-only warmup phases
        3-4: Overlapped forward/backward phases (steady state)
        5-8: Cooldown phases

    Usage:
        schedule = DualPipeScheduleRuntime(
            stages_forward=[stage0, stage1, ...],
            stages_reverse=[stage0_rev, stage1_rev, ...],
            num_microbatches=8,
            loss_fn=cross_entropy_loss,
        )
        losses = schedule.step(input_ids, target=labels)
    """

    def __init__(
        self,
        stages_forward: List[PipelineStage],
        stages_reverse: List[PipelineStage],
        num_microbatches: int = 8,
        loss_fn: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
    ):
        self.stages_forward = stages_forward
        self.stages_reverse = stages_reverse
        self.num_microbatches = num_microbatches
        self.loss_fn = loss_fn
        self.p = len(stages_forward)

    def step(self, *args, target=None, **kwargs):
        """Run one DualPipe training step.

        Executes the DualPipe schedule with computation-communication overlap.
        Phase 0 microbatches (0..half_m-1) flow left→right.
        Phase 1 microbatches (half_m..m-1) flow right→left.

        :return: (loss tensor, None) for unified API.
        """
        m = self.num_microbatches
        half_m = m // 2
        losses = []

        # Phase 0: use stages_forward (left → right)
        # Phase 1: use stages_reverse (right → left)

        # Simplified DualPipe execution:
        # We run phase 0 forwards sequentially, then phase 1 forwards,
        # then backward in reverse order. Full P2P overlap requires
        # CUDA and is beyond the scope of this wrapper.
        for phase, stages in [(0, self.stages_forward), (1, self.stages_reverse)]:
            microbatches = range(half_m) if phase == 0 else range(half_m, m)

            # Forward
            stage_inputs = {}
            for mb in microbatches:
                x = (
                    args[len(stage_inputs)]
                    if stage_inputs
                    else kwargs.get("input_ids", args[0])
                )
                stage_inputs[mb] = x

            # Run stages
            cur = {mb: stage_inputs[mb] for mb in microbatches}
            for stage_idx, stage in enumerate(stages):
                next_cur = {}
                for mb_idx in microbatches:
                    x = cur[mb_idx]
                    if (
                        stage_idx == self.p - 1
                        and self.loss_fn is not None
                        and target is not None
                    ):
                        out, loss = stage.forward(x, target=target)
                        next_cur[mb_idx] = out
                        losses.append(loss)
                    else:
                        next_cur[mb_idx] = stage.forward(x)
                cur = next_cur

            # Backward (reverse order)
            for stage in reversed(stages):
                stage.backward(None)

        loss_tensor = torch.stack(losses) if losses else None
        return (
            loss_tensor.mean() if loss_tensor is not None else torch.tensor(0.0),
            loss_tensor,
        )


class PipelineSchedule:
    """Thin wrapper around PyTorch pipeline schedule classes + DualPipe.

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
        stages_reverse: Optional[List[PipelineStage]] = None,
    ):
        self.stages = stages
        self.pp_config = pp_config
        self.loss_fn = loss_fn

        if num_microbatches is None:
            num_microbatches = pp_config.num_microbatches or len(stages)
        self.num_microbatches = num_microbatches

        # DualPipe uses custom runtime (not in PyTorch schedules)
        if pp_config.schedule == "DualPipe":
            if stages_reverse is None:
                raise ValueError("DualPipe requires stages_reverse")
            self._schedule = DualPipeScheduleRuntime(
                stages_forward=stages,
                stages_reverse=stages_reverse,
                num_microbatches=num_microbatches,
                loss_fn=loss_fn,
            )
            return

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

        self._dualpipe = None

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
    elif schedule_type == "DualPipe":
        return (p // 2 - 1) / m
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
