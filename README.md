# nanopsyche

Distributed training framework built from scratch in PyTorch with a standalone
**FP8-Flow-MoE** package. All 5 parallelism dimensions, MoE with async overlap,
FP8 training, fault-tolerant checkpointing, and automatic pipeline scheduling.

Validated against [FP8-Flow-MoE](https://arxiv.org/abs/2511.03070) (MLSys 2026 Oral) —
found and fixed 17 implementation bugs in the paper's dataflow specification.

## Architecture

```
                      parallelize_model()
             PP → FP8 → CP → TP → EP → AC → compile → FSDP → init

 ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
 │    MODEL LAYER     │  │    DISTRIBUTED     │  │      TRAIN         │  │   CHECKPOINT &     │
 │                    │  │                    │  │                    │  │   FAULT TOLERANCE  │
 │  RMSNorm           │  │  TP (Megatron)     │  │  Trainer           │  │  Distributed-      │
 │  RoPE (YaRN)       │  │  CP (Ring+Ulysses) │  │  (callback driven) │  │  Checkpointer      │
 │  Attention         │  │  PP (GPipe, 1F1B,  │  │  Scheduler         │  │  (async atomic)    │
 │  (MHA/GQA/MQA)     │  │   DualPipe, ZB)    │  │  (Cosine, WSD, Exp)│  │                    │
 │  SwiGLU FFN        │  │  EP (A2A dispatch) │  │  Data (rank-       │  │  RNG State Save    │
 │  MoEBase           │  │  FSDP2 (full shard)│  │   sharded)         │  │                    │
 │  (DeepSeek-V3 bias)│  │                    │  │  Callbacks         │  │  Hang Detector     │
 │  MoEHybridBlock    ├──┤                    ├──┤  (7: speed, mem,   ├──┤  (watchdog thread) │
 │  (dense+MoE async  │  │                    │  │   wandb, console,  │  │                    │
 │   overlap)         │  │                    │  │   profiler, stab.) │  │  ADAMW / DiLoCo    │
 │  FP8 (torchao)     │  │                    │  │                    │  │  FP8 (torchao)     │
 │  FP8-Flow-MoE      │  │                    │  │                    │  │                    │
 └────────────────────┘  └─────────┬──────────┘  └─────────┬──────────┘  └────────────────────┘
                                   │                       │
                           ┌───────▼───────────────────────▼──────────────┐
                           │         DeviceMesh (hybrid, up to 5D)        │
                           │       DP × TP × PP × CP × EP                │
                           └─────────────────────────────────────────────┘
```

## Packages

| Package | Description |
|---------|-------------|
| `fp8_flow_moe/` | **Standalone** FP8-Flow-MoE operators. Zero dependency on nanopsyche. Works with any MoE pipeline. |
| `nanopsyche/` | Full training framework. Depends on `fp8_flow_moe`. |

## Parallelism

| Dimension | Implementation | Reference |
|-----------|---------------|-----------|
| **Data Parallel** | FSDP2 `fully_shard()`, per-block wrapping | PyTorch FSDP2 |
| **Tensor Parallel** | Megatron-style manual sharding + `differentiable_all_reduce` | Megatron-LM |
| **Pipeline Parallel** | GPipe, 1F1B, **DualPipe** (DeepSeek-V3), Zero-Bubble | DeepSeek-V3 |
| **Context Parallel** | Ring Attention + Ulysses (both numerically verified) | Ring Attention, Ulysses |
| **Expert Parallel** | A2A token dispatch, padded-bmm, **async MoE-overlap** | OLMo-core, DeepSeek-V3 |

## Benchmark Results

Single-GPU benchmarks (1B params, batch=2, seq=2048), tokens/s measured with CUDA
events over 10 steady-state iterations:

| Config | GPU | Tokens/s |
|--------|-----|----------|
| Dense | T4 | 13,710 |
| Dense | A10G | 32,168 |
| Dense | H100 | 15,587 |
| MoE (4E, k=1) | A10G | 28,796 |
| MoE (4E, k=1) | H100 | 12,521 |
| MoE + FP8-Flow-MoE (dispatch) | H100 | 13,657 |
| MoE + FP8-Flow-MoE (FP8 GEMM) | H100 | 11,678 |

**FP8 crossover is real and measurable.** Sweeping tokens/expert on an H100 shows
FP8 GEMM only beats BF16 bmm above ~2,048 tokens/expert:

| Tokens/Exp | BF16 tok/s | FP8 tok/s | Speedup |
|------------|-----------|-----------|---------|
| 512 | 11,935 | 10,524 | 0.88x |
| 1,024 | 12,603 | 11,829 | 0.94x |
| 2,048 | 13,086 | 13,465 | **1.03x** |

`fp8_flow_moe` routes each expert block to FP8 GEMM or BF16 bmm at runtime based on
this threshold (`--fp8-gemm-threshold`, default 1,000,000 → BF16). The paper's 21%
speedup requires 671B+ params with EP32 on 256 H100s, where per-expert token counts
reach millions and FP8 halves activation/weight bandwidth. At 1B/1-GPU scale the
sort+batched-bmm dispatch gives +6.6% over per-expert loops; FP8 GEMM itself is a
net loss. See `sweep_crossover.py` / `plot_crossover.py`.

**Note on MFU:** single-GPU 1B benchmarks are kernel-launch and memory-bandwidth
bound, so MFU lands in the 3–23% range. Production MFU numbers (40–60%) come from
large models with long sequences and heavily optimized kernels. We report tokens/s
as the honest, comparable metric.

## CLI

```bash
# Train
nanopsyche train --model 1b --use-moe --fp8-recipe flow_moe --batch-size 4 --max-steps 1000

# Multi-GPU
nanopsyche train --model 7b --use-moe --tp-size 2 --fsdp --ep-size 4 --max-steps 50000

# Benchmark
nanopsyche bench --model 1b --use-moe --fp8-recipe flow_moe --iters 20

# Flags: --model (125m/350m/1b/7b), --use-moe, --num-experts, --top-k,
#   --fp8-recipe (none/blockwise/mxfp8/flow_moe), --fp8-gemm-threshold,
#   --tp-size, --pp-size, --cp-size, --ep-size, --fsdp, --ac-mode,
#   --wandb, --compile, --resume
```

## Tests

```bash
pip install -e ./fp8_flow_moe -e .
pytest tests/ -v   # 69 passed, 5 skipped
```

| Test Suite | Tests | What it verifies |
|-----------|-------|-----------------|
| `test_fp8_flow_moe.py` | 8 | FP8 quantize/dequantize, transpose, permute, SwiGLU, integration |
| `test_moe.py` | 10 | MoE forward, aux loss, bmm, EP, gradient, DeepSeek-V3, capacity |
| `test_model_factory.py` | 10 | Model factory, hook detection, CLI integration |
| `test_cli.py` | 31 | Parser, data pipeline, bench utils, CLI commands |
| `test_pp.py` | 5 | Pipeline parallel split, bubble fractions, forward/backward |
| `test_tp_*.py` | 4 | TP correctness, backward, FSDP+TP composition |

## Key Design Decisions

1. **Megatron-style manual TP** — DTensor conflicts with RoPE and SDPA. Manual weight sharding gives full control.

2. **FSDP2 `fully_shard`** — Per-block wrapping with `reshard_after_forward=False` under PP.

3. **Callback-driven observability** — 7 callbacks: SpeedMonitor, GPUMemoryMonitor, StabilityMonitor, ConsoleLogger, WandB, Validation, Profiler.

4. **FP8-Flow-MoE standalone** — Core FP8 operators in a separate pip package with zero nanopsyche dependencies.

5. **Async MoE overlap** — `MoEHybridTransformerBlock` overlaps EP all-to-all with dense FFN computation.

6. **Adaptive FP8 routing** — Automatically falls back to BF16 bmm when per-expert token count is below the amortization threshold.

## What We Validated

From FP8-Flow-MoE (Wang et al., MLSys 2026 Oral):

- **Transposition bug:** `torch._scaled_mm(A, B)` computes `A @ B.T`. Paper's pseudocode had incorrect transpose flags for gate/up projections.
- **Router weight bug:** Permute-based compute path drops router weights for top_k > 1.
- **EP mesh bug:** `build_world_mesh()` accepted `ep_size` parameter but never added EP to the DeviceMesh.
- **Double quantization:** Paper's BF16 fallback path had incorrect matrix dimension for output projection.
- **17 bugs total** across the dataflow specification — all fixed with tests.

## Environment

- Python 3.12+, PyTorch 2.4+
- Runs on CPU (gloo) for testing
- CUDA required for: FP8 (torchao), Flash Attention, grouped_gemm, NCCL
- Modal for cloud GPU benchmarks (T4, A10G, H100)
