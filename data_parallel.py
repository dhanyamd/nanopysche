"""nanopsyche — Distributed Data Parallel (DDP) from scratch (Lesson 1.2).

DDP in one sentence:
    every GPU keeps a FULL copy of the model, trains on a DIFFERENT batch,
    then AVERAGES gradients so all copies stay identical.

The averaging is a single all_reduce — exactly the collective from Lesson 1.1.

Why averaging gradients == training on all the data at once:
    Each rank's loss is the MEAN over its own batch. Averaging the per-rank
    gradients equals the gradient of the mean loss over ALL batches combined.
    So N GPUs on N different batches behaves like 1 GPU on the concatenation.
"""

import torch.nn as nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # Make every rank START from identical weights: rank 0 broadcasts its
        # parameters to everyone. (If they started different, averaging grads
        # wouldn't keep them in sync.)
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self):
        """Average gradients across all ranks. Call AFTER loss.backward().

        NAIVE version: does ALL the all_reduces at the end, one after another.
        Correct, but slow — the network sits idle during backward, then the GPU
        sits idle during comms. This is the "comms wall".
        """
        world = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)  # sum across GPUs
                p.grad /= world                                # → average


class OverlappingDDP(nn.Module):
    """DDP that OVERLAPS communication with backward (the fast version).

    Instead of one big sync at the end, we register a hook on each parameter.
    The moment that parameter's gradient is ready during backward, the hook
    fires an ASYNC all_reduce (async_op=True returns immediately, comms runs in
    the background) — so the send happens WHILE backward keeps computing earlier
    layers. At the end we just wait for the in-flight sends to finish.

    Correctness is identical to naive DDP; the win is speed (visible on real
    multi-GPU / NCCL, where comms is hidden behind compute).
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world = dist.get_world_size()
        self._handles = []

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)                 # start identical

        for p in self.module.parameters():
            if p.requires_grad:
                # fires as soon as THIS param's grad is accumulated in backward
                p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param):
        # launch the all_reduce NOW, in the background, and keep going
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append((handle, param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_grad_sync(self):
        """Wait for all in-flight all_reduces, then divide by world_size."""
        for handle, param in self._handles:
            handle.wait()            # block until this grad's send is done
            param.grad /= self.world
        self._handles.clear()


class BucketedDDP(nn.Module):
    """DDP with gradient BUCKETING: one all_reduce per bucket, not per param.

    A real model has thousands of parameters. One all_reduce each = thousands of
    tiny latency-bound sends. Instead we pack many gradients into one big buffer
    (a "bucket", ~25 MB) and do ONE all_reduce per bucket — big, bandwidth-bound
    sends. This is what production PyTorch DDP does.

    We pack with flatten -> all_reduce -> unflatten (the same trick DDP uses).
    """

    def __init__(self, module: nn.Module, bucket_mb: int = 25):
        super().__init__()
        self.module = module
        self.world = dist.get_world_size()
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)                 # start identical
        self.buckets = self._build_buckets(bucket_mb * 1024 * 1024)

    def _build_buckets(self, cap_bytes: int):
        """Group parameters into buckets that stay under cap_bytes each."""
        buckets, cur, cur_bytes = [], [], 0
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            b = p.numel() * p.element_size()
            if cur and cur_bytes + b > cap_bytes:         # bucket full -> start new one
                buckets.append(cur)
                cur, cur_bytes = [], 0
            cur.append(p)
            cur_bytes += b
        if cur:
            buckets.append(cur)
        return buckets

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self):
        """One all_reduce per bucket instead of one per parameter."""
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        for bucket in self.buckets:
            grads = [p.grad for p in bucket]
            flat = _flatten_dense_tensors(grads)          # pack many grads -> 1 tensor
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)    # ONE send for the whole bucket
            flat /= self.world
            for p, g in zip(bucket, _unflatten_dense_tensors(flat, grads)):
                p.grad.copy_(g)                            # unpack back into each grad
