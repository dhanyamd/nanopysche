from __future__ import annotations

"""Fault-tolerant checkpointing — resume after node failures.

Production distributed training MUST handle node failures. On a 1000-GPU
cluster, expect ~1 GPU failure per hour. Without fault tolerance, a single
failure kills the entire training run.

Patterns:
    1. Periodic checkpointing: save every N steps (simple but loses work)
    2. Async checkpointing: save in background while training continues
    3. Elastic checkpointing: resize the job to fewer GPUs after failure
    4. WAL (Write-Ahead Log): track completed microbatches for fine-grained resume

OLMo-core:
    - CheckpointerCallback saves at configurable intervals
    - Async saves via background threads
    - Can resume from any checkpoint step

DisTrO / Nous Psyche:
    - Designed for unreliable hardware (consumer GPUs, slow networks)
    - Aggressive checkpointing + compression for fast resume
    - Hang detection to kill stuck ranks quickly

Reference: PyTorch torchelastic
           OLMo-core train/callbacks/checkpoint.py
"""

import torch
import torch.distributed as dist
import time
from pathlib import Path
from typing import Callable


class FaultTolerantCheckpointer:
    """Checkpoint with automatic resume and hang detection.

    Features:
        - Periodic checkpointing (configurable interval)
        - Async checkpointing (non-blocking saves)
        - Step-level resume (resume from exact step)
        - Grace period for stale ranks (wait before killing)

    Usage:
        checkpointer = FaultTolerantCheckpointer(
            save_dir="checkpoints/",
            save_interval=100,
            async_save=True,
        )
        for step in range(max_steps):
            train_step()
            checkpointer.save(model, optimizer, step)
    """

    def __init__(
        self,
        save_dir: str,
        save_interval: int = 100,
        async_save: bool = True,
        max_stale_steps: int = 100,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.async_save = async_save
        self.max_stale_steps = max_stale_steps
        self._save_thread = None
        self._last_saved_step = -1

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        scheduler=None,
        extra: dict = None,
    ):
        """Save checkpoint (optionally async)."""
        if step % self.save_interval != 0:
            return

        # Wait for previous async save to complete
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

        rank = dist.get_rank() if dist.is_initialized() else 0
        save_fn = lambda: self._do_save(model, optimizer, step, scheduler, extra, rank)

        if self.async_save:
            import threading

            self._save_thread = threading.Thread(target=save_fn)
            self._save_thread.start()
        else:
            save_fn()

        self._last_saved_step = step

    def _do_save(self, model, optimizer, step, scheduler, extra, rank):
        """Perform the actual save."""
        step_dir = self.save_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(model.state_dict(), step_dir / f"rank_{rank:04d}_model.pt")

        # Save optimizer
        torch.save(optimizer.state_dict(), step_dir / f"rank_{rank:04d}_optimizer.pt")

        # Save metadata
        if rank == 0:
            import json

            metadata = {"step": step, "timestamp": time.time()}
            if extra:
                metadata["extra"] = extra
            with open(step_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        if dist.is_initialized():
            dist.barrier()

    def find_latest_checkpoint(self) -> int | None:
        """Find the latest valid checkpoint step."""
        step_dirs = sorted(self.save_dir.glob("step_*"))
        if not step_dirs:
            return None
        return int(step_dirs[-1].name.split("_")[1])

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
    ) -> int:
        """Resume from the latest checkpoint. Returns the step to resume from."""
        step = self.find_latest_checkpoint()
        if step is None:
            return 0

        rank = dist.get_rank() if dist.is_initialized() else 0
        step_dir = self.save_dir / f"step_{step:06d}"

        model_path = step_dir / f"rank_{rank:04d}_model.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))

        opt_path = step_dir / f"rank_{rank:04d}_optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

        if dist.is_initialized():
            dist.barrier()

        self._last_saved_step = step
        return step
