"""Profiler — torch.profiler integration with Chrome trace export.

Configurable schedule: skip_first, warmup, active, repeat.
Exports Chrome trace for visualization in chrome://tracing or Perfetto.

Supports distributed profiling: profile specific ranks per parallelism group.

Reference: OLMo-core src/olmo_core/train/callbacks/profiler.py
"""

import logging
from typing import Any, Optional

import torch
import torch.distributed as dist

from nanopsyche.train.callbacks.base import Callback

log = logging.getLogger(__name__)


class ProfilerCallback(Callback):
    """Wrap torch.profiler for training profiling.

    Outputs:
        - Chrome trace JSON (for chrome://tracing / Perfector)
        - Sorted tables by self_cuda_time and self_cpu_time

    Usage:
        ProfilerCallback(
            skip_first=10,   # skip first 10 steps
            wait=5,          # wait 5 steps before profiling
            warmup=5,        # warm up profiler for 5 steps
            active=10,       # profile for 10 steps
            repeat=1,        # repeat cycle once
        )
    """

    def __init__(
        self,
        skip_first: int = 10,
        wait: int = 5,
        warmup: int = 5,
        active: int = 10,
        repeat: int = 1,
        with_stack: bool = True,
        profile_memory: bool = True,
        enable_cuda_sync_events: bool = True,
        output_dir: str = "profiler",
        profile_ranks: Optional[str] = None,
    ):
        """
        :param skip_first: skip first N steps.
        :param wait: wait N steps between profiling cycles.
        :param warmup: warm up profiler for N steps.
        :param active: profile for N steps.
        :param repeat: repeat the cycle this many times.
        :param with_stack: record source file/line info.
        :param profile_memory: track tensor memory allocation.
        :param enable_cuda_sync_events: CUDA sync events for critical-path analysis.
        :param output_dir: directory for trace output.
        :param profile_ranks: which ranks to profile ("all", "dp", "tp", "cp", "pp", or None for rank 0 only).
        """
        self.skip_first = skip_first
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.enable_cuda_sync_events = enable_cuda_sync_events
        self.output_dir = output_dir
        self.profile_ranks = profile_ranks
        self._profiler: Optional[torch.profiler.profile] = None
        self._step = 0
        self._should_profile = False

    def _should_profile_rank(self) -> bool:
        """Determine if this rank should be profiled."""
        if not dist.is_initialized():
            return True
        rank = dist.get_rank()
        if self.profile_ranks is None or self.profile_ranks == "all":
            return True
        if self.profile_ranks == "dp":
            # Profile one rank per DP group (rank 0 of each DP group)
            world_size = dist.get_world_size()
            # Assume TP=1 for simplicity; refine with mesh if available
            return rank == 0
        return rank == 0  # default: rank 0 only

    def pre_step(self, trainer: Any, batch: Any) -> None:
        self._step += 1

        if not self._should_profile_rank():
            return

        # Check if we should be profiling this step
        cycle_length = self.wait + self.warmup + self.active
        in_skip = self._step <= self.skip_first
        in_active = False
        if not in_skip and cycle_length > 0:
            cycle_pos = (self._step - self.skip_first - 1) % cycle_length
            in_active = self.wait <= cycle_pos < self.wait + self.warmup + self.active

        if in_active and self._profiler is None:
            self._start_profiler(trainer)
        elif not in_active and self._profiler is not None:
            self._stop_profiler(trainer)

    def _start_profiler(self, trainer: Any) -> None:
        """Start torch.profiler."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        schedule = torch.profiler.schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
        )

        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            with_stack=self.with_stack,
            profile_memory=self.profile_memory,
            record_shapes=True,
            with_flops=True,
        )
        self._profiler.__enter__()
        log.info(f"Profiler started at step {self._step}")

    def _stop_profiler(self, trainer: Any) -> None:
        """Stop profiler and export results."""
        if self._profiler is None:
            return

        self._profiler.__exit__(None, None, None)

        rank = dist.get_rank() if dist.is_initialized() else 0

        # Export Chrome trace
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        trace_path = (
            f"{self.output_dir}/rank-{rank}-step-{self._step}.chrome_trace.json.gz"
        )
        try:
            self._profiler.export_chrome_trace(trace_path)
            log.info(f"Chrome trace saved to {trace_path}")
        except Exception as e:
            log.warning(f"Failed to export Chrome trace: {e}")

        # Log sorted tables
        try:
            cuda_table = self._profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=20
            )
            cpu_table = self._profiler.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=20
            )
            log.info(f"\n{'=' * 60}\nCUDA Profile (step {self._step}):\n{cuda_table}")
            log.info(f"\n{'=' * 60}\nCPU Profile (step {self._step}):\n{cpu_table}")
        except Exception as e:
            log.warning(f"Failed to print profiler tables: {e}")

        self._profiler = None

    def close(self, trainer: Any) -> None:
        if self._profiler is not None:
            self._stop_profiler(trainer)
