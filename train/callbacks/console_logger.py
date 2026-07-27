"""Console logger — log metrics to Python logging with glob filtering.

Reference: OLMo-core src/olmo_core/train/callbacks/console_logger.py
"""

import fnmatch
import logging
import time
from typing import Any, Dict, Optional

import torch.distributed as dist

from nanopsyche.train.callbacks.base import Callback

log = logging.getLogger(__name__)


# Default metrics to log to console
DEFAULT_LOG_PATTERNS = [
    "train/CE loss",
    "train/PPL",
    "optim/total grad norm",
    "optim/LR*",
    "throughput/device/TPS",
    "throughput/device/MFU",
    "throughput/device/MFU (actual avg)",
    "gpu_memory/GPU active mem (GiB)",
    "gpu_memory/GPU reserved mem (GiB)",
]


class ConsoleLoggerCallback(Callback):
    """Log a configurable subset of metrics to console.

    Uses glob patterns for metric filtering.
    """

    def __init__(
        self,
        metrics_to_log: Optional[list[str]] = None,
        log_interval: int = 1,
    ):
        self.metrics_to_log = metrics_to_log or DEFAULT_LOG_PATTERNS
        self.log_interval = log_interval
        self._train_start: float = 0.0
        self._max_steps: int = 0

    def pre_train(self, trainer: Any) -> None:
        self._train_start = time.time()
        self._max_steps = getattr(trainer.config, "max_steps", 0)

    def log_metrics(self, metrics: Dict[str, Any], step: int, trainer: Any) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        # Filter metrics by glob patterns
        filtered = {}
        for name, value in metrics.items():
            for pattern in self.metrics_to_log:
                if fnmatch.fnmatch(name, pattern):
                    filtered[name] = value
                    break

        if not filtered:
            return

        # Build log line
        elapsed = time.time() - self._train_start
        parts = [f"step={step:6d}"]
        for name, value in filtered.items():
            short_name = name.split("/")[-1] if "/" in name else name
            if isinstance(value, float):
                parts.append(f"{short_name}={value:.4f}")
            else:
                parts.append(f"{short_name}={value}")

        if self._max_steps > 0:
            pct = 100.0 * step / self._max_steps
            parts.append(f"({pct:.1f}%)")

        if step > 0 and elapsed > 0:
            eta = elapsed / step * (self._max_steps - step)
            hours, remainder = divmod(int(eta), 3600)
            minutes, seconds = divmod(remainder, 60)
            parts.append(f"eta={hours:02d}:{minutes:02d}:{seconds:02d}")

        log.info(" | ".join(parts))
