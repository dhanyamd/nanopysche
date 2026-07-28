"""Benchmark utilities — MFU calculation, FLOPs estimation, throughput measurement.

Formulas (standard from PaLM / Llama / OLMo):

    FLOPs per token (non-embedding):
        forward = 2 * N
        backward = 4 * N
        total = 6 * N

    FLOPs per token (including attention):
        attention per layer = 4 * d * S + 4 * d^2
        MLP per layer = 6 * d * h  (SwiGLU: w1, w2, w3)
        total non-embed = 2 * n_layers * (4*d*S + 4*d^2 + 6*d*h) / S
        simplified ≈ 6 * N  (when S >> d)

    MFU = achieved_flops / peak_flops

    achieved_flops = (6 * N_nonemb + attention_flops) * tokens_per_sec

Reference:
    - PaLM: "Scaling Language Modeling with Pathways" (Chowdhery et al. 2022)
    - Llama: "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al. 2023)
    - OLMo: "OLMo: Accelerating the Science of Language Models" (Groeneveld et al. 2024)
"""

from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Peak FLOPs per GPU (BF16/FP16 Tensor Core)
# Sourced from NVIDIA spec sheets and published MFU numbers
# ---------------------------------------------------------------------------

PEAK_FLOPS: dict[str, float] = {
    "A100": 312e12,  # 312 TFLOPS BF16
    "A10G": 125e12,  # 125 TFLOPS BF16 (GA102, 70 SMs)
    "A10": 125e12,  # same as A10G
    "H100": 989e12,  # 989 TFLOPS BF16
    "H200": 989e12,  # 989 TFLOPS BF16
    "B200": 1125e12,  # 1125 TFLOPS BF16 (4.5 PFLOPS FP8)
    "L4": 121e12,  # 121 TFLOPS BF16
    "T4": 65e12,  # 65 TFLOPS FP16
    "V100": 125e12,  # 125 TFLOPS FP16 Tensor Core
}

FP8_PEAK_FLOPS: dict[str, float] = {
    "A100": 624e12,  # 624 TFLOPS FP8 (with sparsity)
    "H100": 1979e12,  # 1979 TFLOPS FP8
    "H200": 1979e12,
    "B200": 4500e12,  # 4.5 PFLOPS FP8
}


def has_fp8_tensor_cores(device: Optional[torch.device] = None) -> bool:
    """Check if the GPU supports native FP8 tensor cores.

    Only H100/H200/B200 and later have native FP8 support.
    A100's FP8 (624 TFLOPS with sparsity) is not exposed as tensor cores in practice.

    :param device: torch device to check (None = use current device).
    :returns: True if the GPU has native FP8 tensor cores.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return False
    name = torch.cuda.get_device_name(device).upper()
    return any(key.upper() in name for key in FP8_PEAK_FLOPS if key != "A100")


def get_peak_flops(device: Optional[torch.device] = None, fp8: bool = False) -> float:
    """Return peak BF16/FP16 TFLOPS for the current GPU.

    Falls back to A10G value if GPU is not in the lookup table.
    Uses case-insensitive substring matching on the GPU name.

    :param device: torch device to check (None = use current device).
    :param fp8: if True, return FP8 peak FLOPs instead of BF16.
    :returns: peak FLOPs as float.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        return 0.0

    name = torch.cuda.get_device_name(device)
    name_upper = name.upper()
    lookup = FP8_PEAK_FLOPS if fp8 else PEAK_FLOPS

    for key, flops in lookup.items():
        if key.upper() in name_upper:
            return flops

    # A10G fallback for unrecognized Ampere+ GPUs
    print(
        f"  [bench] Unknown GPU '{name}', using A10G peak ({PEAK_FLOPS['A10G'] / 1e12:.0f} TFLOPS)"
    )
    fallback = (
        FP8_PEAK_FLOPS.get("A100", PEAK_FLOPS["A10G"]) if fp8 else PEAK_FLOPS["A10G"]
    )
    return fallback


def estimate_model_flops(
    num_params: int,
    num_non_embed_params: int,
    n_layers: int,
    d_model: int,
    ffn_hidden: int,
    n_heads: int,
    seq_len: int,
    batch_size: int,
    num_experts: int = 0,
    top_k: int = 0,
) -> float:
    """Estimate FLOPs per forward+backward pass.

    Uses the standard 6 * N formula for the decoder layers,
    plus attention FLOPs explicitly. For MoE, only top_k experts
    are activated per token.

    :returns: total FLOPs for one training step (forward+backward).
    """
    tokens = batch_size * seq_len

    # Attention FLOPs per layer (forward):
    #   QKV projection: 3 * 2 * d_model^2
    #   QK^T: 2 * S * d_model
    #   PV:   2 * S * d_model
    #   Output proj: 2 * d_model^2
    #   Total attention per layer: 8 * d_model^2 + 4 * S * d_model
    attn_flops_fwd = 8 * d_model * d_model + 4 * seq_len * d_model
    attn_flops_fwd *= n_layers

    # MLP FLOPs per layer (SwiGLU: w1, w2, w3):
    #   gate: 2 * d_model * ffn_hidden
    #   up:   2 * d_model * ffn_hidden
    #   down: 2 * ffn_hidden * d_model
    #   Total: 6 * d_model * ffn_hidden
    if num_experts > 0 and top_k > 0:
        # MoE: only top_k experts active per token
        mlp_flops_fwd = 6 * d_model * ffn_hidden * top_k / num_experts
    else:
        mlp_flops_fwd = 6 * d_model * ffn_hidden
    mlp_flops_fwd *= n_layers

    # Embedding + LM head (negligible but include roughly)
    embed_flops_fwd = 2 * d_model * tokens  # embedding lookup + LM head

    total_fwd = (attn_flops_fwd + mlp_flops_fwd) * tokens + embed_flops_fwd
    total_bwd = 2 * total_fwd  # backward ≈ 2x forward

    return total_fwd + total_bwd


def compute_mfu(
    step_time_seconds: float,
    tokens_per_step: int,
    num_params: int,
    n_layers: int,
    d_model: int,
    ffn_hidden: int,
    n_heads: int,
    seq_len: int,
    batch_size: int,
    device: Optional[torch.device] = None,
    fp8: bool = False,
    num_gpus: int = 1,
    num_experts: int = 0,
    top_k: int = 0,
) -> float:
    """Compute Model FLOPs Utilization (MFU) as a percentage.

    Standard formula:
        MFU = (estimated FLOPs per step) / (peak FLOPs per GPU * num_gpus * step time)

    Reference: PaLM §4.2, OLMo §5.1

    :param step_time_seconds: wall-clock time for one training step.
    :param tokens_per_step: total tokens processed (global_batch_size * seq_len).
    :param num_params: total model parameters.
    :param device: GPU device.
    :param fp8: whether using FP8 computation.
    :param num_gpus: number of GPUs.
    :returns: MFU as a fraction (0.0 to 1.0).
    """
    # Auto-detect: if FP8 was requested but the GPU doesn't have native FP8
    # tensor cores (e.g. A10G), fall back to BF16 peak FLOPs.
    # This prevents inflated MFU on GPUs that simulate FP8 in BF16.
    actual_fp8 = fp8 and has_fp8_tensor_cores(device)
    if fp8 and not actual_fp8:
        import logging

        logging.getLogger("bench_utils").warning(
            "FP8 requested but GPU lacks native FP8 tensor cores; using BF16 peak FLOPs"
        )

    peak = get_peak_flops(device, fp8=actual_fp8) * num_gpus
    if peak <= 0:
        return 0.0

    step_flops = estimate_model_flops(
        num_params=num_params,
        num_non_embed_params=num_params,  # rough: all params counted
        n_layers=n_layers,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        n_heads=n_heads,
        seq_len=seq_len,
        batch_size=batch_size,
        num_experts=num_experts,
        top_k=top_k,
    )

    achieved = step_flops / step_time_seconds
    return achieved / peak
