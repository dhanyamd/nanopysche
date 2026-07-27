"""Speed monitor — throughput and MFU tracking.

Tracks tokens/sec, batches/sec, FLOPS/sec, and Model FLOPS Utilization (MFU).
Auto-detects GPU peak FLOPS by name (H100, A100, B200).

Reference: OLMo-core src/olmo_core/train/callbacks/speed_monitor.py
           PaLM paper (Fedus et al. 2021) for MFU definition
"""

import time
from typing import Any, Optional

import torch
import torch.distributed as dist

from nanopsyche.train.callbacks.base import Callback


# GPU peak FLOPS (BF16, with 0.5x dense correction from spec sheets)
_GPU_PEAK_FLOPS = {
    "NVIDIA H100 80GB HBM3": 1979e12 * 0.5,  # SXM
    "NVIDIA H100": 1979e12 * 0.5,
    "NVIDIA B200": 4.5e15 * 0.5,
    "NVIDIA A100-SXM4-80GB": 624e12 * 0.5,
    "NVIDIA A100-SXM4-40GB": 312e12 * 0.5,
    "NVIDIA A100-PCIE-40GB": 312e12 * 0.5,
    "NVIDIA A100": 624e12 * 0.5,
}
_DEFAULT_PEAK_FLOPS = 624e12 * 0.5  # A100 fallback


def _get_device_peak_flops(device: torch.device) -> float:
    """Get peak FLOPS for the current GPU."""
    if device.type != "cuda":
        return _DEFAULT_PEAK_FLOPS
    gpu_name = torch.cuda.get_device_name(device)
    for name, flops in _GPU_PEAK_FLOPS.items():
        if name in gpu_name:
            return flops
    return _DEFAULT_PEAK_FLOPS


class SpeedMonitorCallback(Callback):
    """Monitor training throughput and MFU.

    Metrics logged:
        throughput/device/TPS — tokens per second (instantaneous)
        throughput/device/TPS (actual avg) — tokens per second (running avg)
        throughput/device/BPS — batches per second
        throughput/device/flopsPS — FLOPS per second
        throughput/device/MFU — Model FLOPS utilization %
        throughput/device/MFU (actual avg) — MFU running average
        throughput/device/data loading (s) — data loading time
        throughput/total tokens — cumulative tokens seen
        throughput/total petaflops — cumulative petaflops computed
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
        window_size: int = 100,
    ):
        """
        :param device: GPU device for peak FLOPS detection.
        :param batch_size: tokens per step (global_batch_size).
        :param window_size: number of steps for running average.
        """
        self._device = device
        self._batch_size = batch_size
        self._window_size = window_size

        self._step_start: float = 0.0
        self._data_load_start: float = 0.0
        self._data_load_end: float = 0.0

        # Running totals
        self._total_tokens: int = 0
        self._total_flops: float = 0.0
        self._total_step_time: float = 0.0
        self._window_step_times: list[float] = []
        self._window_tokens: list[int] = []
        self._window_flops: list[float] = []
        self._peak_flops: Optional[float] = None

    def _ensure_initialized(self, trainer: Any) -> None:
        """Lazy initialization on first call (after trainer is fully set up)."""
        if self._peak_flops is not None:
            return
        if self._device is None:
            self._device = getattr(trainer, "device", torch.device("cpu"))
        if self._batch_size is None:
            self._batch_size = getattr(trainer.config, "global_batch_size", 1)
        self._peak_flops = _get_device_peak_flops(self._device)

    def pre_load_batch(self, trainer: Any) -> None:
        self._ensure_initialized(trainer)
        self._data_load_start = time.perf_counter()

    def pre_step(self, trainer: Any, batch: Any) -> None:
        self._data_load_end = time.perf_counter()
        self._step_start = time.perf_counter()

    def post_step(self, trainer: Any) -> None:
        step_time = time.perf_counter() - self._step_start
        data_load_time = self._data_load_end - self._data_load_start

        if step_time <= 0:
            return

        tokens = self._batch_size
        flops_per_step = self._estimate_flops(trainer)

        self._total_tokens += tokens
        self._total_flops += flops_per_step
        self._total_step_time += step_time

        self._window_step_times.append(step_time)
        self._window_tokens.append(tokens)
        self._window_flops.append(flops_per_step)
        if len(self._window_step_times) > self._window_size:
            self._window_step_times.pop(0)
            self._window_tokens.pop(0)
            self._window_flops.pop(0)

        # Instantaneous metrics
        tps = tokens / step_time
        bps = 1.0 / step_time
        flops_ps = flops_per_step / step_time
        mfu = 100.0 * flops_ps / self._peak_flops if self._peak_flops > 0 else 0.0
        data_load_pct = 100.0 * data_load_time / step_time

        # Running average
        avg_step_time = sum(self._window_step_times) / len(self._window_step_times)
        avg_tps = sum(self._window_tokens) / avg_step_time
        avg_flops_ps = sum(self._window_flops) / avg_step_time
        avg_mfu = (
            100.0 * avg_flops_ps / self._peak_flops if self._peak_flops > 0 else 0.0
        )

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            record = trainer.record_metric
            record("throughput/device/TPS", tps)
            record("throughput/device/TPS (actual avg)", avg_tps)
            record("throughput/device/BPS", bps)
            record("throughput/device/flopsPS", flops_ps)
            record("throughput/device/flopsPS (actual avg)", avg_flops_ps)
            record("throughput/device/MFU", mfu)
            record("throughput/device/MFU (actual avg)", avg_mfu)
            record("throughput/device/data loading (s)", data_load_time)
            record("throughput/device/data loading (%)", data_load_pct)
            record("throughput/total tokens", self._total_tokens)
            record("throughput/total petaflops", self._total_flops / 1e15)

    def _estimate_flops(self, trainer: Any) -> float:
        """Estimate FLOPS for one training step.

        Standard estimate: 6 * num_params * tokens_per_step
        (2 for forward + 1 for backward, each touching params twice)
        """
        model = getattr(trainer, "model", None)
        if model is None:
            return 0.0
        num_params = sum(p.numel() for p in model.parameters())
        return 6.0 * num_params * self._batch_size
