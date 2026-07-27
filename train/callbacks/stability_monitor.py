"""Stability monitor — detect loss/gradient spikes.

Tracks rolling spike rate over a sliding window. A spike is when a value
exceeds mean + threshold_std * std over the window.

Reference: OLMo-core src/olmo_core/train/callbacks/stability_monitor.py
"""

from collections import deque
from typing import Any, Optional

import torch
import torch.distributed as dist

from nanopsyche.train.callbacks.base import Callback


class StabilityMonitorCallback(Callback):
    """Monitor training stability via spike detection.

    Metrics logged:
        spike/SpikeScore — rolling spike rate over last window
        spike/SpikeScore (total) — cumulative spike rate
    """

    def __init__(
        self,
        window_size: int = 128,
        threshold_std: float = 10.0,
        metric_name: str = "train/CE loss",
    ):
        """
        :param window_size: number of steps in the sliding window.
        :param threshold_std: number of std deviations for spike detection.
        :param metric_name: which metric to monitor for spikes.
        """
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.metric_name = metric_name
        self._values: deque[float] = deque(maxlen=window_size)
        self._total_steps: int = 0
        self._total_spikes: int = 0

    def log_metrics(self, metrics: dict[str, Any], step: int, trainer: Any) -> None:
        value = metrics.get(self.metric_name)
        if value is None or not isinstance(value, (int, float, torch.Tensor)):
            return
        if isinstance(value, torch.Tensor):
            value = value.item()

        self._total_steps += 1
        is_spike = False

        if len(self._values) >= 2:
            values_list = list(self._values)
            mean = sum(values_list) / len(values_list)
            std = (sum((v - mean) ** 2 for v in values_list) / len(values_list)) ** 0.5
            if std > 0 and abs(value - mean) > self.threshold_std * std:
                is_spike = True
                self._total_spikes += 1

        self._values.append(value)

        # Compute rolling spike score
        if len(self._values) >= self.window_size:
            # Count spikes in the window
            values_list = list(self._values)
            mean = sum(values_list) / len(values_list)
            std = (sum((v - mean) ** 2 for v in values_list) / len(values_list)) ** 0.5
            if std > 0:
                n_spikes = sum(
                    1 for v in values_list if abs(v - mean) > self.threshold_std * std
                )
                spike_score = n_spikes / len(values_list)
            else:
                spike_score = 0.0
        else:
            spike_score = 0.0

        total_spike_score = (
            self._total_spikes / self._total_steps if self._total_steps > 0 else 0.0
        )

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            trainer.record_metric("spike/SpikeScore", spike_score)
            trainer.record_metric("spike/SpikeScore (total)", total_spike_score)
