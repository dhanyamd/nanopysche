"""Benchmarks — MFU, communication volume, memory profiling.

MFU (Model FLOPs Utilization):
    MFU = actual_throughput / peak_hardware_throughput
    Target: >50% for production training

Communication volume:
    bytes_over_wire = total_bytes_sent during one training step
    Compare: DDP (all-reduce) vs FSDP (all-gather + reduce-scatter) vs compressed

Memory profiling:
    torch.cuda.memory_stats() for detailed GPU memory tracking
    torch.profiler for compute/memory timeline

Reference: PaLM paper (Chowdhery et al. 2022) for MFU definition
           OLMo-core SpeedMonitorCallback
"""

import torch
import torch.distributed as dist
import time
from typing import Any


def compute_mfu(
    model: torch.nn.Module,
    tokens_per_sec: float,
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    """Compute Model FLOPs Utilization (MFU).

    MFU = actual_flops / peak_hardware_flops

    Approximation for transformer:
        flops_per_token ≈ 6 * num_params
        (2 for forward matmul + 2 for backward matmul + 2 for gradient matmul)

    For H100 SXM:
        BF16 peak: 989 TFLOPS
        FP8 peak:  1979 TFLOPS

    Args:
        model: the transformer model
        tokens_per_sec: measured throughput
        dtype: compute dtype (determines peak flops)

    Returns:
        MFU as a fraction (0.0 to 1.0)
    """
    num_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 6 * num_params  # approximate
    flops_per_sec = flops_per_token * tokens_per_sec

    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        peak_flops = 1979e12  # H100 FP8
    elif dtype == torch.bfloat16:
        peak_flops = 989e12  # H100 BF16
    else:
        peak_flops = 989e12  # assume BF16

    return flops_per_sec / peak_flops


def estimate_bytes_communicated(
    model: torch.nn.Module,
    parallelism: str = "ddp",
    dp_size: int = 1,
    tp_size: int = 1,
    cp_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Estimate communication volume for different parallelism strategies.

    Returns bytes communicated per training step for each strategy.

    DDP:
        all-reduce gradients: 2 * N * dtype_size
        (2 because all-reduce = reduce + scatter)

    FSDP:
        all-gather forward: N * dtype_size per layer
        reduce-scatter backward: N * dtype_size per layer
        Total: 2 * N * dtype_size (same as DDP but spread across layers)

    TP:
        all-reduce per TP region (attention + FFN): 2 * activations * dtype_size
        With SequenceParallel: same total, different timing

    CP:
        Ring: P2P send/recv of KV chunks: O(S * H * D * cp_size)
        Ulysses: 2 all-to-all of (B * S/CP * H * D)
    """
    num_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = (
        2 if dtype == torch.bfloat16 else 4
    )  # fp16/bf16 = 2 bytes, fp32 = 4

    if parallelism == "ddp":
        # All-reduce: 2 * N * bytes_per_param
        return 2 * num_params * bytes_per_param
    elif parallelism == "fsdp":
        # All-gather + reduce-scatter: same as DDP
        return 2 * num_params * bytes_per_param
    elif parallelism == "tp":
        # Per-layer all-reduce in attention and FFN
        # Rough: 2 * activations * bytes_per_param
        return 2 * num_params * bytes_per_param
    elif parallelism == "cp":
        # Ring attention: P2P of KV chunks
        return num_params * bytes_per_param
    else:
        return 0


class MemoryProfiler:
    """Profile GPU memory usage during training.

    Usage:
        profiler = MemoryProfiler()
        profiler.start()
        for step in range(100):
            train_step()
            profiler.step()
        profiler.summary()
    """

    def __init__(self):
        self.snapshots = []
        self.start_time = None

    def start(self):
        """Reset memory stats and record start time."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.start_time = time.time()
        self.snapshots = []

    def step(self):
        """Record a memory snapshot."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            stats = torch.cuda.memory_stats()
            self.snapshots.append(
                {
                    "time": time.time() - self.start_time,
                    "allocated": stats["allocated_bytes.all.current"] / 1e9,
                    "reserved": stats["reserved_bytes.all.current"] / 1e9,
                    "peak_allocated": stats["allocated_bytes.all.peak"] / 1e9,
                    "peak_reserved": stats["reserved_bytes.all.peak"] / 1e9,
                }
            )

    def summary(self) -> dict[str, Any]:
        """Print and return memory usage summary."""
        if not self.snapshots:
            return {}

        peak_alloc = max(s["peak_allocated"] for s in self.snapshots)
        peak_reserved = max(s["peak_reserved"] for s in self.snapshots)
        final_alloc = self.snapshots[-1]["allocated"]

        summary = {
            "peak_allocated_GB": peak_alloc,
            "peak_reserved_GB": peak_reserved,
            "final_allocated_GB": final_alloc,
            "num_snapshots": len(self.snapshots),
        }

        print(f"Memory Profile:")
        print(f"  Peak allocated: {peak_alloc:.2f} GB")
        print(f"  Peak reserved:  {peak_reserved:.2f} GB")
        print(f"  Final allocated: {final_alloc:.2f} GB")

        return summary


def profile_throughput(
    model: torch.nn.Module,
    input_shape: tuple[int, int],
    num_steps: int = 100,
    warmup_steps: int = 10,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Profile training throughput.

    Returns:
        tokens_per_sec: throughput in tokens per second
        mfu: Model FLOPs Utilization
        step_time_ms: average step time in milliseconds
    """
    device = next(model.parameters()).device
    B, S = input_shape

    model.train()

    # Warmup
    for _ in range(warmup_steps):
        x = torch.randint(0, 32000, (B, S), device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            output = model(x, labels=x)
        output["loss"].backward()
        model.zero_grad(set_to_none=True)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_steps):
        x = torch.randint(0, 32000, (B, S), device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            output = model(x, labels=x)
        output["loss"].backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = B * S * num_steps
    tokens_per_sec = total_tokens / elapsed
    step_time_ms = (elapsed / num_steps) * 1000
    mfu = compute_mfu(model, tokens_per_sec, dtype)

    results = {
        "tokens_per_sec": tokens_per_sec,
        "mfu": mfu,
        "step_time_ms": step_time_ms,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
    }

    print(f"Throughput: {tokens_per_sec:.0f} tokens/s")
    print(f"MFU: {mfu:.2%}")
    print(f"Step time: {step_time_ms:.1f} ms")

    return results
