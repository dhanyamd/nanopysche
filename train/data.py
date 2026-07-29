from __future__ import annotations

"""Distributed data loading — rank-aware batching and sharding.

In production distributed training, each rank must see DIFFERENT data
(same data = wasted compute, model sees same examples multiple times).

Patterns:
    1. Shard dataset by rank: rank i gets examples [i, i+world_size, i+2*world_size, ...]
    2. DistributedSampler (PyTorch): handles shuffling and sharding
    3. Data loading in background: num_workers > 0 for CPU-side prefetching

OLMo-core data loading:
    - NumpyFSLDataset: fixed-sequence-length, memory-mapped numpy arrays
    - NumpyDataLoader: distributed-aware, global batch size in TOKENS (not instances)
    - SourceMixtureDataset: composable data mixing from multiple sources

Variable Sequence Length (VSL) curriculum:
    - Start training with shorter sequences, grow to full length
    - Reduces wasted compute on padding early in training
    - OLMo-core uses grow_p2 with 8 cycles (unbalanced)

Reference: OLMo-core data/data_loader.py
"""

import torch
import torch.distributed as dist
import torch.utils.data as data
import numpy as np
from typing import Iterator


class DistributedDataset(data.Dataset):
    """Simple distributed dataset.

    Splits a flat token array into fixed-length sequences.
    Sharding across ranks is handled by DistributedSampler in
    DistributedDataLoader, NOT by rank-based slicing here.

    For real training, replace this with:
        - NumpyFSLDataset (OLMo-core): memory-mapped, fixed-sequence-length
        - HuggingFace datasets with streaming
        - WebDataset for web-crawled data
    """

    def __init__(
        self,
        data_source: torch.Tensor | np.ndarray,
        sequence_length: int = 2048,
    ):
        self.sequence_length = sequence_length

        if isinstance(data_source, torch.Tensor):
            data_source = data_source.numpy()
        self.data = data_source

        # Precompute number of sequences
        self.num_sequences = (len(self.data) - 1) // sequence_length

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.sequence_length
        end = start + self.sequence_length + 1  # +1 for labels

        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class DistributedDataLoader:
    """Distributed data loader with automatic sharding and prefetching.

    Wraps PyTorch's DataLoader with rank-aware sampling. Each rank
    independently loads its own shard — no communication needed.

    Production patterns (OLMo-core):
        - Global batch size specified in TOKENS, not instances
        - Automatic microbatch splitting for gradient accumulation
        - num_workers > 0 for CPU-side prefetching
        - pin_memory=True for faster CPU->GPU transfer
    """

    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        # Use DistributedSampler when distributed is initialized,
        # otherwise use a plain RandomSampler for single-GPU / CPU training.
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
            )
        else:
            sampler = data.RandomSampler(dataset)

        self._loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling (important for training diversity)."""
        if hasattr(self._loader.sampler, "set_epoch"):
            self._loader.sampler.set_epoch(epoch)


def create_distributed_loader(
    data_source: np.ndarray | torch.Tensor,
    batch_size: int,
    sequence_length: int = 2048,
    num_workers: int = 4,
) -> DistributedDataLoader:
    """Create a distributed data loader from raw data.

    Sharding is handled by DistributedSampler inside DistributedDataLoader,
    not by rank-based slicing here.

    Args:
        data_source: tokenized data as numpy array or torch tensor
        batch_size: microbatch size per GPU
        sequence_length: fixed sequence length
        num_workers: number of data loading workers

    Returns:
        DistributedDataLoader ready for training
    """
    dataset = DistributedDataset(data_source, sequence_length)

    return DistributedDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
