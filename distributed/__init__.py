"""nanopsyche.distributed — Production distributed parallelism.

Pipeline Parallelism schedules (1F1B, GPipe, DualPipe, Zero-Bubble)
and Context Parallelism (Ring Attention, Ulysses) are original implementations
based on the cited papers.

Tensor Parallelism and FSDP use PyTorch native primitives
(parallelize_module, fully_shard) — not custom autograd functions.

Reference: OLMo-core src/olmo_core/distributed/
"""

from nanopsyche.distributed.pipeline_parallel import (
    PipelineStage,
    GPipeSchedule,
    OneForwardOneBackwardSchedule,
    DualPipeSchedule,
    ZeroBubbleSchedule,
    compute_bubble_fraction,
)
from nanopsyche.distributed.context_parallel import RingAttention

__all__ = [
    "PipelineStage",
    "GPipeSchedule",
    "OneForwardOneBackwardSchedule",
    "DualPipeSchedule",
    "ZeroBubbleSchedule",
    "compute_bubble_fraction",
    "RingAttention",
]
