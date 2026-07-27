from __future__ import annotations

"""Pipeline Parallelism — GPipe, 1F1B, and communication primitives.

Pipeline parallelism splits the model vertically across GPUs:
    GPU 0: layers 0-7    (stage 0)
    GPU 1: layers 8-15   (stage 1)
    GPU 2: layers 16-23  (stage 2)
    GPU 3: layers 24-31  (stage 3)

Microbatches: the batch is split into M microbatches that flow through
the pipeline like an assembly line. Different stages process different
microbatches simultaneously.

Communication:
    Forward:  stage[i] sends activations to stage[i+1]
    Backward: stage[i+1] sends gradients to stage[i]
    Tensors:  hidden states (B, S, H) in both directions

Schedules:
    GPipe:    all forwards, then all backward. Simple but high memory.
    1F1B:     alternate 1 forward, 1 backward after warmup. Same bubble, less memory.
    DualPipe: bidirectional, from both ends. Reduces bubbles by ~2x.
    Zero-Bubble: split backward into B (input grad) and W (weight grad), fill bubbles.

Bubble fractions:
    GPipe/1F1B:  (p-1) / m
    DualPipe:    ~ (p/2-1) / m  (roughly half)
    Zero-Bubble: 0 (with 2x memory)

Reference: Huang et al. 2018 (GPipe)
           Narayanan et al. 2019 (1F1B / PipeDream-Flush)
           DeepSeek-V3 (DualPipe)
           Qi et al. 2024 (Zero-Bubble)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from enum import Enum, auto
from dataclasses import dataclass


class ScheduleType(Enum):
    FWD = auto()
    BWD = auto()
    IDLE = auto()


@dataclass
class PipelineParallelConfig:
    """Configuration for pipeline parallelism.

    :param degree: the PP degree (number of pipeline stages).
    :param schedule: schedule type ("1f1b", "gpipe", "dualpipe", "zero_bubble").
    """

    degree: int
    schedule: str = "1f1b"


@dataclass
class Action:
    """A single action in the pipeline schedule."""

    type: ScheduleType
    microbatch_idx: int
    stage_idx: int


class PipelineStage(nn.Module):
    """A single pipeline stage — owns a subset of transformer layers.

    Each GPU holds one PipelineStage. The stage:
    1. Receives input activations from the previous stage (or from the data loader)
    2. Runs its local layers
    3. Sends output activations to the next stage (or computes loss on the last stage)

    The forward pass MUST be a single chunk that can be differentiated through
    for the backward pass. We use torch tensors directly (not autograd.Function)
    because the P2P communication is handled outside the forward pass.
    """

    def __init__(
        self,
        stage_idx: int,
        num_stages: int,
        layers: nn.ModuleList,
        loss_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.layers = layers
        self.loss_fn = loss_fn

        # P2P communication groups
        self.send_rank = stage_idx + 1 if stage_idx < num_stages - 1 else None
        self.recv_rank = stage_idx - 1 if stage_idx > 0 else None

    def forward_chunk(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        is_last_chunk: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one chunk through this stage's layers.

        Args:
            x: (B, S, H) — input activations
            labels: (B, S) — labels for loss computation (last stage only)
            is_last_chunk: whether this is the final stage

        Returns:
            Hidden states, or (hidden states, loss) on the last stage.
        """
        for layer in self.layers:
            x = layer(x)

        if is_last_chunk and self.loss_fn is not None and labels is not None:
            logits = self.loss_fn(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return x, loss

        return x

    def send_forward(self, x: torch.Tensor):
        """Send activations to the next stage (non-blocking)."""
        if self.send_rank is not None:
            dist.send(x, dst=self.send_rank)

    def recv_forward(self, x: torch.Tensor):
        """Receive activations from the previous stage."""
        if self.recv_rank is not None:
            dist.recv(x, src=self.recv_rank)

    def send_backward(self, grad: torch.Tensor):
        """Send gradients to the previous stage."""
        if self.recv_rank is not None:
            dist.send(grad, dst=self.recv_rank)

    def recv_backward(self, grad: torch.Tensor):
        """Receive gradients from the next stage."""
        if self.send_rank is not None:
            dist.recv(grad, src=self.send_rank)


class GPipeSchedule:
    """GPipe: all forwards, then all backward.

    Bubble fraction: (p-1) / m
    Memory: O(m) — all microbatches' activations stored simultaneously

    Timing diagram (p=4, m=8):
        GPU 0: [F0][F1][F2][F3][F4][F5][F6][F7]                    [B0][B1]...
        GPU 1:     [F0][F1][F2][F3][F4][F5][F6][F7]                [B0][B1]...
        GPU 2:         [F0][F1][F2][F3][F4][F5][F6][F7]            [B0][B1]...
        GPU 3:             [F0][F1][F2][F3][F4][F5][F6][F7]        [B0][B1]...
    """

    def __init__(self, stages: list[PipelineStage], num_microbatches: int):
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.p = len(stages)

    def generate_schedule(self) -> list[list[Action]]:
        """Generate the GPipe schedule.

        Phase 1: all forwards for all microbatches
        Phase 2: all backward for all microbatches
        """
        schedule = []
        for m in range(self.num_microbatches):
            for s in range(self.p):
                schedule.append([Action(ScheduleType.FWD, m, s)])

        for m in range(self.num_microbatches):
            for s in range(self.p - 1, -1, -1):
                schedule.append([Action(ScheduleType.BWD, m, s)])

        return schedule


class OneForwardOneBackwardSchedule:
    """1F1B (PipeDream-Flush): alternate 1 forward, 1 backward after warmup.

    Three phases:
    1. Warmup: stage s performs (p-1-s) forward passes
    2. Steady: alternate 1 backward, 1 forward (the "1F1B" pattern)
    3. Cooldown: drain remaining backward passes

    Bubble fraction: (p-1) / m  [same as GPipe]
    Memory: O(p) — only p microbatches in-flight at any time (vs O(m) for GPipe)

    Timing diagram (p=4, m=8):
        GPU 0: [F0][F1][F2][B0][F3][B1][F4][B2][F5][B3][F6][B4][F7][B5]    [B6][B7]
        GPU 1:     [F0][F1][B0][F2][B1][F3][B2][F4][B3][F5][B4][F6][B5][F7][B6][B7]
        GPU 2:         [F0][B0][F1][B1][F2][B2][F3][B3][F4][B4][F5][B5][F6][B6][F7][B7]
        GPU 3:             [B0]  [B1]  [B2]  [B3]  [B4]  [B5]  [B6][B7]

    Key insight: 1F1B has the SAME bubble as GPipe but O(p) memory instead of O(m).
    With m=100 and p=4: GPipe stores 100 microbatches, 1F1B stores 4.
    """

    def __init__(self, stages: list[PipelineStage], num_microbatches: int):
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.p = len(stages)

    def generate_schedule(self) -> list[list[Action]]:
        """Generate the 1F1B schedule for each rank."""
        all_schedules = []
        for rank in range(self.p):
            schedule = self._generate_rank_schedule(rank)
            all_schedules.append(schedule)
        return all_schedules

    def _generate_rank_schedule(self, rank: int) -> list[Action]:
        """Generate schedule for a single rank in 1F1B."""
        schedule = []
        p = self.p
        m = self.num_microbatches

        # Phase 1: Warmup — perform (p - 1 - rank) forward passes
        num_warmup = p - 1 - rank
        for i in range(num_warmup):
            schedule.append(Action(ScheduleType.FWD, i, rank))

        # Phase 2: Steady state — alternate 1 backward, 1 forward
        fwd_idx = num_warmup
        bwd_idx = 0
        while fwd_idx < m:
            schedule.append(Action(ScheduleType.BWD, bwd_idx, rank))
            schedule.append(Action(ScheduleType.FWD, fwd_idx, rank))
            fwd_idx += 1
            bwd_idx += 1

        # Phase 3: Cooldown — remaining backward passes
        while bwd_idx < m:
            schedule.append(Action(ScheduleType.BWD, bwd_idx, rank))
            bwd_idx += 1

        return schedule


class DualPipeSchedule:
    """DualPipe (DeepSeek-V3): bidirectional pipeline from both ends.

    Key innovation: microbatches enter from BOTH ends of the pipeline simultaneously.
    Phase 0 flows left-to-right; Phase 1 flows right-to-left.

    Each GPU holds 2x parameter sets (forward + reverse direction).
    The backward of one phase is overlapped with the forward of the other.

    Bubble fraction: ~(p/2 - 1) / m  [roughly half of 1F1B]
    Memory: O(p+1) per stage, 2x parameter memory

    The 8-step schedule (from DeepSeek-V3 §2.2.2):
        1. nF0:   Phase 0 forward-only warmup
        2. nF0F1: Interleave Phase 0 and Phase 1 forwards
        3. nB1W1F1: Phase 1 backward + weight update + Phase 1 forward
        4. nF0B1F1B0: MAIN STEADY STATE — overlapped (this is the bulk)
        5. nB1F1B0: Phase 1 backward + Phase 1/0 mixed
        6. nB1B0: Two backward phases
        7. nWB0: Weight update + Phase 0 backward
        8. nW: Final weight updates

    Reference: DeepSeek-V3 Technical Report, §2.2.2
    """

    def __init__(
        self,
        stages_forward: list[PipelineStage],
        stages_reverse: list[PipelineStage],
        num_microbatches: int,
    ):
        self.stages_forward = stages_forward
        self.stages_reverse = stages_reverse
        self.num_microbatches = num_microbatches
        self.p = len(stages_forward)
        self.half_p = self.p // 2

    def generate_schedule(self) -> dict[int, list[Action]]:
        """Generate DualPipe schedule for each rank.

        Returns dict mapping rank -> list of Actions (the per-rank schedule).
        """
        m = self.num_microbatches
        p = self.p
        half_p = self.half_p

        # Each microbatch has a phase: 0 (left-to-right) or 1 (right-to-left)
        # Microbatches 0..half_m-1 are phase 0, half_m..m-1 are phase 1
        # But they interleave differently per rank

        schedules: dict[int, list[Action]] = {r: [] for r in range(p)}

        # Helper: latency of the deepest stage
        # Phase 0 enters at stage 0; Phase 1 enters at stage p-1
        # For a stage r, Phase 0 is r steps away, Phase 1 is (p-1-r) steps away

        def send_timing(stage: int, phase: int) -> int:
            """Step index when stage first receives a phase microbatch."""
            if phase == 0:
                return stage
            else:
                return p - 1 - stage

        # Bubble fraction is (p/2 - 1) / m — half of 1F1B
        # We schedule microbatches in waves

        # --- Step 1: Forward-only warmup for Phase 0 ---
        for step in range(p):
            for r in range(p):
                if (
                    step >= send_timing(r, 0)
                    and len(
                        [
                            a
                            for a in schedules[r]
                            if a.type == ScheduleType.FWD and a.microbatch_idx >= 0
                        ]
                    )
                    < step
                ):
                    mb_id = step - send_timing(r, 0)
                    if mb_id < m // 2 and mb_id >= 0:
                        schedules[r].append(Action(ScheduleType.FWD, mb_id, r))

        # --- Steps 2-7: Full interleaving of phases 0 and 1 ---
        # For simplicity, generate the core overlapped pattern
        # Phase 0 microbatches: 0..half_m-1 (flow left->right)
        # Phase 1 microbatches: half_m..m-1 (flow right->left)

        half_m = m // 2

        for rank in range(p):
            schedule = schedules[rank]
            pp_rank = rank

            # When does this rank see each phase?
            phase0_start = pp_rank  # steps before first phase 0 mb arrives
            phase1_start = p - 1 - pp_rank  # steps before first phase 1 mb arrives

            # Warmup: emit phase 0 forwards
            f0_count = max(0, phase1_start - phase0_start)
            for i in range(min(f0_count, half_m)):
                schedule.append(Action(ScheduleType.FWD, i, pp_rank))

            # Core interleaved: alternate phase 0 and phase 1
            f0_idx = max(0, phase1_start - phase0_start)
            f1_idx = 0
            b1_idx = 0
            b0_idx = 0

            # Steady state: while there's work to do in either phase
            while (
                f0_idx < half_m or f1_idx < half_m or b0_idx < half_m or b1_idx < half_m
            ):
                # Phase 1 backward (happens as soon as phase 1 is deep enough)
                if b1_idx < half_m and b1_idx < f1_idx:
                    schedule.append(Action(ScheduleType.BWD, half_m + b1_idx, pp_rank))
                    b1_idx += 1

                # Phase 0 forward
                if f0_idx < half_m:
                    schedule.append(Action(ScheduleType.FWD, f0_idx, pp_rank))
                    f0_idx += 1

                # Phase 1 forward
                if f1_idx < half_m:
                    schedule.append(Action(ScheduleType.FWD, half_m + f1_idx, pp_rank))
                    f1_idx += 1

                # Phase 0 backward
                if b0_idx < half_m and b0_idx < f0_idx:
                    schedule.append(Action(ScheduleType.BWD, b0_idx, pp_rank))
                    b0_idx += 1

            # Drain remaining backward
            while b1_idx < half_m:
                schedule.append(Action(ScheduleType.BWD, half_m + b1_idx, pp_rank))
                b1_idx += 1
            while b0_idx < half_m:
                schedule.append(Action(ScheduleType.BWD, b0_idx, pp_rank))
                b0_idx += 1

        return schedules


class ZeroBubbleSchedule:
    """Zero-Bubble Pipeline Parallelism.

    Key insight: split backward into B (backward-for-input) and W (backward-for-weight).
    B depends on the next stage's B, but W can be scheduled anywhere after its own B.
    This allows filling pipeline bubbles with W operations.

    Variants:
        ZB-1P: bubble = (p-1) / (3*(m+p-1))  — same memory as 1F1B
        ZB-2P: bubble = 0                       — 2x memory
        ZB-V:  bubble = 0                       — same memory, 2x communication

    The W operations at the tail form a rectangle, filling the trapezoid's bubbles.

    Reference: Qi et al. 2024 (ICLR 2024)
    """

    def __init__(self, stages: list[PipelineStage], num_microbatches: int):
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.p = len(stages)

    def generate_schedule(self) -> list[Action]:
        """Generate Zero-Bubble schedule.

        Split each backward into B (input grad) and W (weight grad).
        Schedule W operations to fill pipeline bubbles.
        """
        schedule = []
        p = self.p
        m = self.num_microbatches

        # For each rank, generate the schedule
        for rank in range(p):
            # Warmup: forward passes
            for i in range(p - 1 - rank):
                schedule.append(Action(ScheduleType.FWD, i, rank))

            # Steady state: interleave B, F, W
            for i in range(m - (p - 1 - rank)):
                schedule.append(Action(ScheduleType.BWD, i, rank))
                schedule.append(Action(ScheduleType.FWD, p - 1 - rank + i, rank))

            # Cooldown: remaining B and W
            for i in range(rank + 1):
                schedule.append(Action(ScheduleType.BWD, m - 1 - i, rank))

            # W operations fill the bubbles
            for i in range(m):
                schedule.append(Action(ScheduleType.BWD, i, rank))  # W phase

        return schedule


def compute_bubble_fraction(schedule_type: str, p: int, m: int, v: int = 1) -> float:
    """Compute the bubble fraction for a given schedule.

    Args:
        schedule_type: "gpipe", "1f1b", "interleaved", "dualpipe", "zb_1p", "zb_2p", "zb_v"
        p: number of pipeline stages
        m: number of microbatches
        v: virtual pipeline chunks (for interleaved schedules)

    Returns:
        Bubble fraction (0.0 = no bubble, 1.0 = all bubble)
    """
    if schedule_type in ("gpipe", "1f1b"):
        return (p - 1) / m
    elif schedule_type == "interleaved":
        return (p - 1) / (v * m)
    elif schedule_type == "dualpipe":
        return (p // 2 - 1) / m
    elif schedule_type == "zb_1p":
        return (p - 1) / (3 * (m + p - 1))
    elif schedule_type in ("zb_2p", "zb_v"):
        return 0.0
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
