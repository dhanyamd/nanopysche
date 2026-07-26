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

    Key innovation: microbatches enter from both ends of the pipeline,
    reducing bubble fraction by ~2x compared to 1F1B.

    Each GPU holds 2x parameters (forward direction + reverse direction).
    Communication is bidirectional: forward sends to next, backward sends to prev.

    Bubble fraction: ~(p/2 - 1) / m  [roughly half of 1F1B]
    Memory: O(p+1) per stage, 2x parameter memory

    The 4-component chunk decomposition:
        Forward:  [Attn] [MoE_Dispatch] [FFN] [MoE_Combine]
        Backward: [B_input_FFN] [B_weight_FFN] [MoE_Combine_bwd] [B_input_Attn] [B_weight_Attn] [MoE_Dispatch_bwd]

    Computation-communication overlap:
        While computing forward for phase 0, simultaneously handle backward
        communication for phase 1. This is the key to DualPipe's efficiency.
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

    def generate_schedule(self) -> list[Action]:
        """Generate DualPipe schedule.

        The schedule has 8 steps (from DeepSeek-V3 paper):
        1. nF0: forward-only warmup in phase 0
        2. nF0F1: interleave forward phase 0 and phase 1
        3. nB1W1F1: backward phase 1 + weight update + forward phase 1
        4. nF0B1F1B0: main steady state (overlapped F+B)
        5. nB1F1B0: backward phase 1 + forward/backward phase 0
        6. nB1B0: two backward phases
        7. nWB0: weight updates + backward phase 0
        8. nW: final weight updates
        """
        # Simplified: generate the microbatch ordering
        # Full implementation follows the 8-step pattern from DeepSeek-V3
        half_p = self.p // 2
        m = self.num_microbatches
        half_m = m // 2

        schedule = []

        # Step 1: Forward-only warmup
        for i in range(half_p):
            schedule.append(Action(ScheduleType.FWD, i, 0))

        # Step 2: Interleaved forward
        for i in range(half_m - half_p):
            schedule.append(Action(ScheduleType.FWD, half_p + i, 0))
            schedule.append(Action(ScheduleType.FWD, i, 1))

        # Step 4: Main steady state — overlapped forward + backward
        for i in range(half_m):
            schedule.append(Action(ScheduleType.BWD, i, 1))
            schedule.append(Action(ScheduleType.FWD, half_m + i, 0))
            schedule.append(Action(ScheduleType.BWD, half_m + i, 1))
            schedule.append(Action(ScheduleType.FWD, i, 0))

        return schedule


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
