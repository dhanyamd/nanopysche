"""nanopsyche.optim — Optimizers for distributed training.

AdamW with weight decay groups (standard for LLMs).
DiLoCo-style gradient compression for communication-efficient training.
"""

from nanopsyche.optim.adamw import AdamW
from nanopsyche.optim.diko import DiLoCoCompressor

__all__ = ["AdamW", "DiLoCoCompressor"]
