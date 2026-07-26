"""Trainer — production training loop with gradient accumulation.

Matches OLMo-core pattern:
    Trainer handles the loop, TrainModule handles model+optimizer.

Reference: OLMo-core src/olmo_core/train/trainer.py
           OLMo-core src/olmo_core/train/train_module/transformer/train_module.py
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    global_batch_size: int = 1024
    micro_batch_size: int = 4
    sequence_length: int = 2048
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 600000
    dtype: torch.dtype = torch.bfloat16
    loss_scale: float = 1.0
    save_interval: int = 1000
    save_dir: str = "checkpoints"
    log_interval: int = 10
    dp_world_size: int = 1

    @property
    def gradient_accumulation_steps(self) -> int:
        per_gpu = self.micro_batch_size * self.sequence_length
        return self.global_batch_size // (per_gpu * self.dp_size)

    @property
    def dp_size(self) -> int:
        return self.dp_world_size


class Trainer:
    """Production training loop.

    Handles gradient accumulation, mixed precision, gradient clipping,
    optimizer step, LR scheduling, and checkpointing.

    Reference: OLMo-core src/olmo_core/train/trainer.py
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        data_loader=None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.step = 0
        self.tokens_seen = 0
        self._setup_distributed()

    def _setup_distributed(self):
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main = self.rank == 0
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

    def fit(self):
        self.model.train()
        if self.is_main:
            print(f"Starting training: {self.config.max_steps} steps")

        for step in range(self.config.max_steps):
            step_start = time.time()
            loss = self._train_step()

            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.model.zero_grad(set_to_none=True)

            step_time = time.time() - step_start
            if self.is_main and step % self.config.log_interval == 0:
                self._log_step(step, loss, step_time)
            if self.is_main and step % self.config.save_interval == 0 and step > 0:
                self._save_checkpoint(step)

    def _train_step(self) -> float:
        self.model.train()
        total_loss = 0.0
        accum_steps = self.config.gradient_accumulation_steps

        for _ in range(accum_steps):
            batch = self._get_batch()
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.config.dtype):
                output = self.model(input_ids, labels=labels)
                loss = output["loss"] / accum_steps
                if "aux_loss" in output:
                    loss = loss + output["aux_loss"] / accum_steps

            loss.backward()
            total_loss += loss.item() * accum_steps

        return total_loss

    def _get_batch(self) -> dict[str, torch.Tensor]:
        if self.data_loader is not None:
            return next(self.data_loader)
        B = self.config.micro_batch_size
        S = self.config.sequence_length
        return {
            "input_ids": torch.randint(0, 32000, (B, S)),
            "labels": torch.randint(0, 32000, (B, S)),
        }

    def _log_step(self, step: int, loss: float, step_time: float):
        tokens_per_step = self.config.global_batch_size
        self.tokens_seen += tokens_per_step
        tokens_per_sec = tokens_per_step / step_time
        lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"step={step:6d} | loss={loss:.4f} | lr={lr:.2e} | "
            f"tokens/s={tokens_per_sec:.0f} | time={step_time:.2f}s"
        )

    def _save_checkpoint(self, step: int):
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        path = save_dir / f"step_{step:06d}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Resumed from step {self.step}")
