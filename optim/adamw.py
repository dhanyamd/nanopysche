"""AdamW optimizer — production implementation with weight decay groups.

AdamW = Adam with decoupled weight decay (separate from gradient update).

Key differences from vanilla Adam:
    - Weight decay is applied directly to weights, not through gradients
    - This prevents the L2 regularization from interacting with adaptive LR

Production patterns:
    - Separate param groups: no weight decay on embeddings, LayerNorm, biases
    - Betas: (0.9, 0.95) is standard for LLMs (LLaMA-2/3)
    - Epsilon: 1e-8 (default)
    - Fused AdamW: torch.optim.AdamW(fused=True) on CUDA for speed

Reference: Loshchilov & Hutter 2019 (Decoupled Weight Decay Regularization)
           Used by: LLaMA-2/3, Mistral, Qwen-2.5, Gemma
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Any


class AdamW(Optimizer):
    """AdamW with weight decay groups.

    Production usage:
        # Separate param groups
        decay_params = [p for n, p in model.named_parameters() if "bias" not in n and "norm" not in n]
        no_decay_params = [p for n, p in model.named_parameters() if "bias" in n or "norm" in n]

        optimizer = AdamW([
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=3e-4, betas=(0.9, 0.95))

    Why no weight decay on biases and norms:
        - Bias: doesn't contribute to model capacity in the same way
        - LayerNorm/RMSNorm: has its own scaling mechanism
        - Embeddings: regularized differently (dropout, etc.)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        AdamW update:
            1. Apply weight decay: w = w * (1 - lr * wd)
            2. Update biased moments: m = beta1 * m + (1 - beta1) * g
                                        v = beta2 * v + (1 - beta2) * g^2
            3. Bias correction: m_hat = m / (1 - beta1^t)
                                 v_hat = v / (1 - beta2^t)
            4. Update: w = w - lr * (m_hat / (sqrt(v_hat) + eps))
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Decoupled weight decay (the key difference from Adam)
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Update biased first and second moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)

                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def create_adamw(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """Create AdamW with proper param groups for a transformer model.

    Separates parameters into decay/no-decay groups:
        - Decay: all weights except biases, norms, embeddings
        - No decay: biases, LayerNorm/RMSNorm weights, embeddings
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            "bias" in name
            or "norm" in name
            or "emb" in name
            or "weight" in name.split(".")[-1] == "weight"
            and param.dim() < 2
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )
