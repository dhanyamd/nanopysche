from __future__ import annotations

"""Learning rate schedulers — production implementations.

Common schedules for LLM training:
    1. Cosine with warmup: the default for most LLMs
    2. WSD (Warmup-Stable-Decay): used by MiniCPM, DeepSeek
    3. Exponential decay: simple baseline

OLMo-core supports: CosWithWarmup, WSD, ExponentialDecay, SteppedWSDS

Reference: Loshchilov & Hutter 2016 (SGDR)
           MiniCPM (WSD schedule)
"""

import math
import torch


def cosine_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    min_lr: float = 0.0,
    max_lr: float = 1e-3,
) -> float:
    """Cosine annealing with linear warmup.

    Phase 1 (warmup): linearly increase from 0 to max_lr
    Phase 2 (cosine): cosine anneal from max_lr to min_lr

    This is the default schedule for LLaMA-2/3, Mistral, Qwen-2.5.

    The cosine schedule is preferred over linear decay because it keeps
    the learning rate high for longer (cosine is flat at the top),
    allowing more exploration before convergence.

    Reference: Loshchilov & Hutter 2016 (SGDR)
    """
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def warmup_stable_decay(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr: float = 0.0,
    max_lr: float = 1e-3,
    decay_lr: float | None = None,
) -> float:
    """WSD (Warmup-Stable-Decay) schedule.

    Phase 1 (warmup): linear warmup to max_lr
    Phase 2 (stable): constant at max_lr
    Phase 3 (decay): cosine anneal to min_lr

    Used by MiniCPM, DeepSeek-V2/V3. The advantage over pure cosine:
    you can train the "stable" phase indefinitely, then decide when to
    anneal. This enables flexible training duration.

    The decay phase can be short (10% of total) — the model converges
    quickly because it's already been at high LR for most of training.
    """
    total = warmup_steps + stable_steps + decay_steps

    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    elif step < warmup_steps + stable_steps:
        return max_lr
    elif step < total:
        decay_progress = (step - warmup_steps - stable_steps) / decay_steps
        return min_lr + 0.5 * (max_lr - min_lr) * (
            1 + math.cos(math.pi * decay_progress)
        )
    else:
        return min_lr


def exponential_decay(
    step: int,
    warmup_steps: int,
    decay_rate: float = 0.999,
    min_lr: float = 0.0,
    max_lr: float = 1e-3,
) -> float:
    """Exponential decay with warmup.

    LR = max_lr * decay_rate^step (after warmup)
    """
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    return max(min_lr, max_lr * (decay_rate ** (step - warmup_steps)))


class CosineWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    """PyTorch-compatible cosine schedule with warmup.

    Usage:
        optimizer = AdamW(model.parameters(), lr=3e-4)
        scheduler = CosineWithWarmup(optimizer, warmup_steps=2000, max_steps=600000)
        for step in range(max_steps):
            train_step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            if step >= max_steps:
                return min_lr_ratio
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )

        super().__init__(optimizer, lr_lambda)


class WarmupStableDecay(torch.optim.lr_scheduler.LambdaLR):
    """PyTorch-compatible WSD schedule.

    Usage:
        optimizer = AdamW(model.parameters(), lr=3e-4)
        scheduler = WarmupStableDecay(optimizer, warmup_steps=2000, stable_steps=50000, decay_steps=10000)
        for step in range(max_steps):
            train_step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        total = warmup_steps + stable_steps + decay_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            elif step < warmup_steps + stable_steps:
                return 1.0
            elif step < total:
                progress = (step - warmup_steps - stable_steps) / self.decay_steps
                return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                    1 + math.cos(math.pi * progress)
                )
            else:
                return min_lr_ratio

        super().__init__(optimizer, lr_lambda)
