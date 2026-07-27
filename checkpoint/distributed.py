from __future__ import annotations

"""Distributed checkpointing — save/load model state across ranks.

Production distributed training requires checkpointing that works with
sharded parameters (FSDP, TP, PP). Saving a full model copy on every
rank wastes disk and memory.

Two approaches:
    1. Rank 0 saves full model (simple but memory-hungry)
    2. Each rank saves its shard, metadata tracks the layout (OLMo-core pattern)

OLMo-core uses torch.distributed.checkpoint:
    - Each rank saves only its local shard
    - Metadata file describes the global layout
    - Loading: unshard to any parallelism configuration
    - Async checkpointing: save in background while training continues

Reference: PyTorch torch.distributed.checkpoint
           OLMo-core distributed/checkpoint.py
"""

import torch
import torch.distributed as dist
import json
from pathlib import Path
from typing import Any


class DistributedCheckpointer:
    """Distributed checkpoint save/load.

    Saves model state, optimizer state, scheduler state, and training config.
    Each rank saves only its local shard to save disk space.

    Usage:
        checkpointer = DistributedCheckpointer("checkpoints/")
        checkpointer.save(model, optimizer, step=1000)
        checkpointer.load(model, optimizer, step=1000)
    """

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        step: int = 0,
        config: Any = None,
        extra: dict | None = None,
    ):
        """Save distributed checkpoint.

        Each rank saves its local shard. A metadata file describes
        the global layout for reconstruction.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        step_dir = self.save_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save model state (each rank's local shard)
        model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(model_state, step_dir / f"rank_{rank:04d}_model.pt")

        # Save optimizer state
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            torch.save(opt_state, step_dir / f"rank_{rank:04d}_optimizer.pt")

        # Save scheduler state
        if scheduler is not None:
            sched_state = scheduler.state_dict()
            torch.save(sched_state, step_dir / f"rank_{rank:04d}_scheduler.pt")

        # Save metadata (only rank 0)
        if rank == 0:
            metadata = {
                "step": step,
                "world_size": world_size,
                "model_keys": list(model_state.keys()),
            }
            if config is not None:
                metadata["config"] = (
                    config.as_dict() if hasattr(config, "as_dict") else str(config)
                )
            if extra is not None:
                metadata["extra"] = extra

            with open(step_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        # Barrier to ensure all ranks finish saving
        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            print(f"Saved checkpoint to {step_dir}")

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        step: int | None = None,
    ) -> dict:
        """Load distributed checkpoint.

        If step is None, loads the latest checkpoint.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Find checkpoint directory
        if step is not None:
            step_dir = self.save_dir / f"step_{step:06d}"
        else:
            # Find latest
            step_dirs = sorted(self.save_dir.glob("step_*"))
            if not step_dirs:
                raise FileNotFoundError(f"No checkpoints found in {self.save_dir}")
            step_dir = step_dirs[-1]

        # Load model state
        model_path = step_dir / f"rank_{rank:04d}_model.pt"
        if model_path.exists():
            model_state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(model_state, strict=False)

        # Load optimizer state
        if optimizer is not None:
            opt_path = step_dir / f"rank_{rank:04d}_optimizer.pt"
            if opt_path.exists():
                opt_state = torch.load(opt_path, map_location="cpu")
                optimizer.load_state_dict(opt_state)

        # Load scheduler state
        if scheduler is not None:
            sched_path = step_dir / f"rank_{rank:04d}_scheduler.pt"
            if sched_path.exists():
                sched_state = torch.load(sched_path, map_location="cpu")
                scheduler.load_state_dict(sched_state)

        # Barrier to ensure all ranks finish loading
        if dist.is_initialized():
            dist.barrier()

        # Load metadata
        metadata_path = step_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists() and rank == 0:
            with open(metadata_path) as f:
                metadata = json.load(f)

        return metadata

    def latest_step(self) -> int | None:
        """Get the latest checkpoint step number."""
        step_dirs = sorted(self.save_dir.glob("step_*"))
        if not step_dirs:
            return None
        return int(step_dirs[-1].name.split("_")[1])
