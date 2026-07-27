"""WandB logger — log metrics to Weights & Biases.

Rank 0 only. Supports remote run cancellation via WandB tags.

Reference: OLMo-core src/olmo_core/train/callbacks/wandb.py
"""

import logging
from typing import Any, Dict, Optional

import torch.distributed as dist

from nanopsyche.train.callbacks.base import Callback

log = logging.getLogger(__name__)


class WandBCallback(Callback):
    """Log training metrics to Weights & Biases.

    Usage:
        WandBCallback(project="my-project", entity="my-team")

    Features:
        - Rank 0 only logging
        - Remote cancel via WandB tags
        - Automatic config upload
    """

    def __init__(
        self,
        project: str = "nanopsyche",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        cancel_check_interval: int = 300,
    ):
        self.project = project
        self.entity = entity
        self.group = group
        self.name = name
        self.tags = tags or []
        self.notes = notes
        self.cancel_check_interval = cancel_check_interval
        self._wandb: Any = None
        self._last_cancel_check: float = 0.0

    def pre_train(self, trainer: Any) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        try:
            import wandb

            self._wandb = wandb

            config = {}
            if hasattr(trainer, "config"):
                config = (
                    trainer.config.__dict__
                    if hasattr(trainer.config, "__dict__")
                    else {}
                )
            if hasattr(trainer, "model") and hasattr(trainer.model, "d_model"):
                config["d_model"] = trainer.model.d_model
                config["n_layers"] = getattr(trainer.model, "n_layers", None)
                config["num_params"] = sum(
                    p.numel() for p in trainer.model.parameters()
                )

            self._wandb.init(
                project=self.project,
                entity=self.entity,
                group=self.group,
                name=self.name,
                tags=self.tags,
                notes=self.notes,
                config=config,
                resume="allow",
            )
            log.info(f"WandB initialized: {self._wandb.run.url}")
        except ImportError:
            log.warning("wandb not installed, skipping WandB logging")
            self._wandb = None

    def log_metrics(self, metrics: Dict[str, Any], step: int, trainer: Any) -> None:
        if self._wandb is None:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        self._wandb.log(metrics, step=step)

    def post_step(self, trainer: Any) -> None:
        """Periodically check for cancel tags."""
        if self._wandb is None:
            return

        import time

        now = time.time()
        if now - self._last_cancel_check < self.cancel_check_interval:
            return
        self._last_cancel_check = now

        try:
            wandb_api = self._wandb.Api()
            run = wandb_api.run(f"{self.entity}/{self.project}/{self._wandb.run.id}")
            tags = run.tags or []
            cancel_tags = {"cancel", "canceled", "cancelled"}
            if any(t.lower() in cancel_tags for t in tags):
                log.warning("WandB cancel tag detected! Stopping training.")
                raise KeyboardInterrupt("WandB cancel tag detected")
        except Exception as e:
            if "cancel" in str(e).lower():
                raise
            # Ignore API errors

    def close(self, trainer: Any) -> None:
        if self._wandb is not None:
            self._wandb.finish()
