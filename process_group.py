"""nanopsyche — process group foundations (Lesson 1.1)

The bedrock of ALL distributed training: how N separate OS processes become
one coordinated training job.

Mental model — SPMD (Single Program, Multiple Data):
  - You launch the SAME script N times (normally one process per GPU).
  - Each process gets a unique `rank` in [0, N) and knows the `world_size` N.
  - Processes coordinate via a `backend` (NCCL on GPU, Gloo on CPU) using
    `collectives` (all_reduce, all_gather, ...).
  - `torchrun --nproc_per_node=N script.py` sets the env vars (RANK,
    WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT) and spawns the processes.

Key distinction:
  - rank        = global id across the whole job  (0 .. world_size-1)
  - local_rank  = id WITHIN one machine/node      (0 .. gpus_per_node-1)
                  -> this is which physical GPU this process drives.

Reference: PyTorch distributed docs (torch.distributed), HF Ultra-Scale
Playbook "A0: Parallel programming crash course".
"""

import os
import torch
import torch.distributed as dist


def setup() -> None:
    """Initialize the default process group from torchrun's env vars.

    NCCL when CUDA is available (real GPU training); Gloo on CPU so you can
    learn collectives locally on your laptop for FREE before spending on GPUs.
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:
    """True only on global rank 0 — use to guard printing / checkpoint saves."""
    return rank() == 0


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank()}")
    return torch.device("cpu")


def print_once(*args, **kwargs) -> None:
    """Print only from rank 0 (avoids N duplicate lines)."""
    if is_main():
        print(*args, **kwargs)
