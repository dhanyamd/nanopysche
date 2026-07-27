"""Tensor parallelism utilities — differentiable all-reduce for backward pass.

PyTorch's dist.all_reduce doesn't have an autograd kernel, so backprop
through it uses a fallthrough that can give incorrect gradients. We wrap
it in a custom autograd.Function to make it differentiable.

Megatron-LM TP backward math:
  Forward: output = all_reduce_sum(local_output)  → same on all ranks
  Backward: d(loss)/d(local_i) = d(loss)/d(output) * d(output)/d(local_i)
          = grad_output * 1  (since output = sum(local_i), d(output)/d(local_i) = 1)
  So backward of all_reduce_sum is IDENTITY — just pass grad through unchanged.

IMPORTANT: We clone the tensor before all_reduce to avoid in-place modification
of the autograd graph node. Modifying a tensor in-place after it has been used
in a computation can corrupt the backward gradient computation.

Reference: Shoeybi et al. 2019 (Megatron-LM)
           PyTorch torch.autograd.Function
"""

import torch
import torch.distributed as dist


class _AllReduce(torch.autograd.Function):
    """Differentiable all-reduce for tensor parallelism.

    Forward: all_reduce (sum) the tensor across TP ranks.
    Backward: pass gradient through unchanged (identity).
              Because output = sum(local_i), the Jacobian d(output)/d(local_i) = 1.
    """

    @staticmethod
    def forward(ctx, group: dist.ProcessGroup, tensor: torch.Tensor) -> torch.Tensor:
        ctx.group = group
        # Clone to avoid in-place modification of autograd graph node
        out = tensor.clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Identity backward — each rank keeps its gradient as-is
        return None, grad_output


def differentiable_all_reduce(
    tensor: torch.Tensor, group: dist.ProcessGroup, tp_size: int = 1
) -> torch.Tensor:
    """All-reduce (sum) that supports backpropagation.

    :param tensor: the tensor to all-reduce.
    :param group: the process group.
    :param tp_size: unused, kept for API compatibility.
    :returns: the all-reduced tensor.
    """
    return _AllReduce.apply(group, tensor)
