"""GPU memory monitor — track VRAM usage per step.

Logs peak active and reserved GPU memory in GiB and as % of capacity.

Reference: OLMo-core src/olmo_core/train/callbacks/gpu_memory_monitor.py
"""

from typing import Any, Optional

import torch
import torch.distributed as dist

from nanopsyche.train.callbacks.base import Callback


class GPUMemoryMonitorCallback(Callback):
    """Monitor GPU memory usage per step.

    Metrics logged:
        gpu_memory/GPU active mem (GiB) — peak active memory
        gpu_memory/GPU active mem (%) — peak active as % of capacity
        gpu_memory/GPU reserved mem (GiB) — peak reserved memory
        gpu_memory/GPU reserved mem (%) — peak reserved as % of capacity
    """

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device

    def post_step(self, trainer: Any) -> None:
        device = self._device or getattr(trainer, "device", None)
        if device is None or device.type != "cuda":
            return

        rank = dist.get_rank() if dist.is_initialized() else 0

        # Only log from rank 0 (or a specific rank per DP group)
        if rank != 0:
            return

        peak_active_bytes = torch.cuda.max_memory_allocated(device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
        total_bytes = torch.cuda.get_device_properties(device).total_mem

        peak_active_gib = peak_active_bytes / (1024**3)
        peak_reserved_gib = peak_reserved_bytes / (1024**3)
        total_gib = total_bytes / (1024**3)

        active_pct = 100.0 * peak_active_bytes / total_bytes if total_bytes > 0 else 0.0
        reserved_pct = (
            100.0 * peak_reserved_bytes / total_bytes if total_bytes > 0 else 0.0
        )

        record = trainer.record_metric
        record("gpu_memory/GPU active mem (GiB)", peak_active_gib)
        record("gpu_memory/GPU active mem (%)", active_pct)
        record("gpu_memory/GPU reserved mem (GiB)", peak_reserved_gib)
        record("gpu_memory/GPU reserved mem (%)", reserved_pct)

        # Reset peak stats for next step
        torch.cuda.reset_peak_memory_stats(device)
