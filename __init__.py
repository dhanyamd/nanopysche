"""nanopsyche — Production-grade distributed training framework.

Built from scratch using PyTorch native distributed primitives.
Covers all 5 parallelism dimensions (DP, FSDP, TP, PP, CP, EP),
with training infrastructure, checkpointing, FP8, and fault tolerance.

Architecture:
    model/              — RMSNorm, RoPE, Attention, FeedForward, MoE, Transformer
    distributed/        — PP schedules (1F1B/GPipe/DualPipe/ZB), CP (Ring/Ulysses)
    nn/                 — TP wrapper utilities
    train/              — Trainer, data loading, LR schedulers
    config.py           — Dataclass config with build() pattern
    parallel.py         — DeviceMesh composition (DP x TP x PP x CP)
    parallelize.py      — parallelize_model() composition layer
    bench.py            — MFU, throughput, memory profiling

Usage:
    from nanopsyche.config import ExperimentConfig
    config = ExperimentConfig()
    model, optimizer, scheduler, train_config = config.build()
"""

from nanopsyche.model import (
    Transformer,
    TransformerBlock,
    RMSNorm,
    Attention,
    FeedForward,
    MoEBase,
)
from nanopsyche.train import Trainer
from nanopsyche.config import ExperimentConfig, TransformerConfig, TrainingConfig

__version__ = "0.1.0"

__all__ = [
    "Transformer",
    "TransformerBlock",
    "RMSNorm",
    "Attention",
    "FeedForward",
    "MoEBase",
    "Trainer",
    "ExperimentConfig",
    "TransformerConfig",
    "TrainingConfig",
]
