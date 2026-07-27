"""Base callback — lifecycle hooks for the training loop.

The trainer calls these hooks in order each step:
    pre_load_batch → pre_step → [train_batch] → pre_optim_step →
    [optim_step] → zero_grads → post_train_batch → post_step

Metrics are recorded via the trainer's record_metric() method, then
batched and dispatched to callbacks' log_metrics() periodically.

Reference: OLMo-core src/olmo_core/train/callbacks/base.py
"""

from typing import Any, Dict, Optional


class Callback:
    """Base class for training callbacks.

    Override any lifecycle method to add behavior at that point.
    """

    def pre_train(self, trainer: Any) -> None:
        """Called before training starts."""

    def post_train(self, trainer: Any) -> None:
        """Called after training ends."""

    def pre_load_batch(self, trainer: Any) -> None:
        """Called before loading a batch."""

    def pre_step(self, trainer: Any, batch: Any) -> None:
        """Called after batch is loaded, before forward pass."""

    def post_train_batch(self, trainer: Any) -> None:
        """Called after forward+backward, before optimizer step."""

    def pre_optim_step(self, trainer: Any) -> None:
        """Called before the optimizer step."""

    def post_step(self, trainer: Any) -> None:
        """Called after the full training step (forward+backward+optim)."""

    def pre_log_metrics(self, trainer: Any) -> None:
        """Called before metrics are dispatched to loggers."""

    def log_metrics(self, metrics: Dict[str, Any], step: int, trainer: Any) -> None:
        """Log a batch of reduced metrics.

        :param metrics: dict of metric_name -> value.
        :param step: current training step.
        :param trainer: the Trainer instance.
        """

    def post_log_metrics(self, trainer: Any) -> None:
        """Called after metrics are dispatched to loggers."""

    def on_error(self, trainer: Any, exception: BaseException) -> None:
        """Called when training encounters an error."""

    def close(self, trainer: Any) -> None:
        """Called when training ends or is interrupted."""
