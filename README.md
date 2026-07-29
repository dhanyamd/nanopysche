# nanopsyche

Production-grade distributed training framework built from scratch in PyTorch, with
a **standalone FP8-Flow-MoE** pip package (MLSys 2026 Oral). Replicates patterns from
[OLMo-core](https://github.com/allenai/olmo-core) and
[DeepSeek-V3](https://arxiv.org/abs/2503.19046).

Covers **all 5 parallelism dimensions** (TP, PP, CP, DP/FSDP, EP), MoE with async overlap,
DualPipe scheduling, FP8 training (torchao + FP8-Flow-MoE), fault-tolerant checkpointing,
and automatic pipeline schedule generation.

## Architecture

```
                      parallelize_model()
             PP → FP8 → CP → TP → EP → AC → compile → FSDP → init

 ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
 │    MODEL LAYER     │  │    DISTRIBUTED     │  │      TRAIN         │  │   CHECKPOINT &     │
 │                    │  │                    │  │                    │  │   FAULT TOLERANCE  │
 │  RMSNorm           │  │  TP (Megatron)     │  │  Trainer           │  │                    │
 │  RoPE (YaRN)       │  │  CP (Ring+Ulysses) │  │  (callback driven) │  │  Distributed-      │
 │  Attention         │  │  PP (GPipe, 1F1B,  │  │  Scheduler         │  │  Checkpointer      │
 │  (MHA/GQA/MQA)     │  │   DualPipe, ZB)    │  │  (Cosine, WSD, Exp)│  │  (async atomic)    │
 │  SwiGLU FFN        │  │  EP (A2A dispatch) │  │  Data (rank-       │  │                    │
 │  MoEBase           │  │  FSDP2 (full shard)│  │   sharded)         │  │  RNG State Save    │
 │  (DeepSeek-V3 bias)│  │                    │  │  Callbacks         │  │                    │
 │  MoEHybridBlock    ├──┤                    ├──┤  (7: speed, mem,   ├──┤  Hang Detector     │
 │  (dense+MoE async  │  │                    │  │   wandb, console,  │  │  (watchdog thread) │
 │   overlap)         │  │                    │  │   profiler, stab.) │  │                    │
 │  AutoScheduler     │  │                    │  │  Optimizer         │  │  ADAMW             │
 │  (auto PP schedule)│  │                    │  │  (AdamW, DiLoCo)   │  │  DiLoCo (gradient  │
 │  FP8 (torchao)     │  │                    │  │                    │  │   compression)     │
 │  FP8-Flow-MoE      │  │                    │  │                    │  │  FP8 (torchao      │
 │  Config Builder    │  │                    │  │                    │  │   wrapper)         │
 └────────────────────┘  └─────────┬──────────┘  └─────────┬──────────┘  └────────────────────┘
                                   │                      │
                           ┌───────▼──────────────────────▼──────────────┐
                           │         DeviceMesh (hybrid, up to 5D)        │
                           │       DP × TP × PP × CP × EP                │
                           └─────────────────────────────────────────────┘
```

## Packages

| Package | pip install | Description |
|---------|-------------|-------------|
| `fp8_flow_moe/` | `pip install fp8-flow-moe` | **Standalone** FP8-Flow-MoE operators. Zero dependency on nanopsyche. Works with any MoE training pipeline. |
| `nanopsyche/` | `pip install nanopsyche` | Full training framework. Depends on `fp8_flow_moe`. |

### fp8_flow_moe — Standalone FP8-Flow-MoE

Reference: Wang et al. "FP8-Flow-MoE: A Casting-Free Dataflow for FP8 MoE Training" (MLSys 2026 Oral).

Reduces quantize/dequantize operations from **12 → 2** per MoE layer:
- Standard FP8 (DeepSeek-V3): cast BF16↔FP8 for every sub-op (w1, w3, w2 × fprop/bprop)
- FP8-Flow-MoE: cast BF16→FP8 **once**, do all expert math in FP8, cast FP8→BF16 **once**

Framework-agnostic — use it with nanopsyche, TorchTitan, Megatron, or any other framework:

```python
from fp8_flow_moe import (
    FP8FlowMoEConfig, FP8FlowMoECompute,
    quantize_to_fp8, dequantize_from_fp8,
    scaling_aware_transpose, fused_swiglu_quantize,
)

compute = FP8FlowMoECompute(config, d_model, hidden_size, num_experts)
output = compute(x, weights, indices, expert_start, expert_end)
```

## File Map

| Directory | Files | Description |
|-----------|-------|-------------|
| `model/` | 8 files | RMSNorm, RoPE, MHA/GQA/MQA, SwiGLU, Transformer, MoE (DeepSeek-V3), MoEHybridBlock |
| `distributed/` | 6 files | TP, CP (Ring+Ulysses), PP (GPipe/1F1B/DualPipe/ZB), PP runtime, FSDP, AutoSchedule |
| `train/` | 11 files | Trainer, scheduler, data loader, 6 callbacks (speed, mem, wandb, console, profiler, stability) |
| `fault_tolerance/` | 2 files | Async checkpoint, hang detection via watchdog |
| `checkpoint/` | 1 file | DistributedCheckpointer with atomic saves, RNG state, async Futures |
| `optim/` | 2 files | AdamW from scratch, DiLoCo gradient compression |
| `fp8/` | 2 files | FP8-Flow-MoE re-exports + torchao FP8 training for dense layers |
| Root | 5 files | CLI (train + bench), config builder, DeviceMesh, parallelize_model(), bench_utils |

## CLI Usage

```bash
# Train a dense model (CPU)
nanopsyche train --model 125m

# Train with MoE + FP8-Flow-MoE
nanopsyche train --model 1b --use-moe --fp8-recipe flow_moe --batch-size 4 --max-steps 1000

# Train with parallelism (multi-GPU)
nanopsyche train --model 7b --use-moe --tp-size 2 --fsdp --ep-size 4 \
  --batch-size 8 --max-steps 50000 --save-dir ./ckpts --wandb

# Resume from checkpoint
nanopsyche train --model 350m --resume checkpoints/step_000100.pt

# Benchmark throughput
nanopsyche bench --model 1b --use-moe --fp8-recipe flow_moe --iters 20

# CLI flags
#   --model           Model size preset: 125m, 350m, 1b, 7b
#   --use-moe         Enable MoE layers
#   --num-experts     Number of experts (default: 8)
#   --top-k           MoE top-k routing (default: 2)
#   --fp8-recipe      FP8 recipe: none, blockwise, mxfp8, flow_moe
#   --batch-size      Micro batch size per GPU
#   --seq-len         Sequence length (default: 2048)
#   --dataset         HF dataset or local .npy file
#   --lr              Learning rate (default: 3e-4)
#   --max-steps       Training steps (default: 100)

# Parallelism flags (multi-GPU):
#   --tp-size         Tensor parallelism degree
#   --pp-size         Pipeline parallelism degree
#   --cp-size         Context parallelism degree
#   --ep-size         Expert parallelism degree
#   --fsdp            Enable FSDP (Fully Sharded Data Parallelism)
#   --ac-mode         Activation checkpointing: full, selected_blocks

# Observability:
#   --wandb           Enable Weights & Biases
#   --wandb-project   WandB project name
#   --compile         Enable torch.compile
#   --save-dir        Checkpoint save directory
#   --save-interval   Checkpoint interval (steps)
#   --resume          Resume from checkpoint path
#   --val-interval    Validation interval
```

## Parallelism Dimensions

| Dimension | Implementation | Reference |
|-----------|---------------|-----------|
| **Data Parallel (DP)** | FSDP2 via `fully_shard()`, per-block wrapping, `reshard_after_forward` config | PyTorch FSDP2 |
| **Tensor Parallel (TP)** | Megatron-style manual sharding + `differentiable_all_reduce` (clone-before-reduce) | Megatron-LM |
| **Pipeline Parallel (PP)** | GPipe, 1F1B, **DualPipe** (DeepSeek-V3), Zero-Bubble schedules + runtime via `torch.distributed.pipelining` | DeepSeek-V3, PyTorch PP |
| **Context Parallel (CP)** | Ring Attention (P2P KV ring) + Ulysses (QKV all-to-all reshape) — both numerically verified | Ring Attention, Ulysses |
| **Expert Parallel (EP)** | A2A token dispatch, padded-bmm/grouped_gemm, **async MoE-overlap via hybrid blocks** | OLMo-core, DeepSeek-V3 |

## Key Design Decisions

1. **Megatron-style manual TP** (not DTensor parallelize_module): DTensor conflicts with RoPE and SDPA. Manual weight sharding + `differentiable_all_reduce` gives full control.

2. **FSDP2 `fully_shard`** (not FSDP1): Per-block wrapping with `reshard_after_forward=False` under PP to avoid redundant all-gathers during microbatch pipelining. **Router FP32** policy for MoE layers.

3. **Callback-driven observability**: Matches OLMo-core's architecture. `record_metric()` batches metrics, reduces across ranks, dispatches to all callbacks. 6 callbacks pre-wired: SpeedMonitor, GPUMemoryMonitor, StabilityMonitor, ConsoleLogger, WandB, Validation.

4. **Parallelize_model() composition**: Single entry point applies PP → FP8 → CP → TP → EP → AC → compile → FSDP in order. All parallelism degrees default to 1 (no-op for single GPU).

5. **FP8-Flow-MoE standalone**: Core FP8 operators extracted into a separate pip package (`fp8_flow_moe/`) with zero nanopsyche dependencies. Works with any framework.

6. **Async MoE overlap**: `MoEHybridTransformerBlock` overlaps all-to-all EP dispatch with dense FFN computation — hides EP communication behind compute.

7. **Automatic pipeline schedule generation** (novel): `AutoScheduleGenerator` computes optimal pipeline schedules from stage profiles, supporting both 1F1B and DualPipe modes.

## Running Tests

```bash
# Install in dev mode
pip install -e ./fp8_flow_moe -e .

# Run all tests (36 total)
pytest tests/test_cli.py tests/test_fp8_flow_moe.py -v

# Single-GPU tests
PYTHONPATH=. python3 tests/test_moe.py                          # 10 MoE tests
PYTHONPATH=. python3 tests/test_hybrid_moe.py                   # 11 hybrid MoE tests

# Multi-GPU tests (need torchrun)
PYTHONPATH=. python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29500 tests/test_tp_correctness.py
PYTHONPATH=. python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29501 tests/test_fsdp.py
PYTHONPATH=. python3 -m torch.distributed.run --nproc_per_node=4 --master_port=29503 tests/test_tp_fsdp.py
PYTHONPATH=. python3 -m torch.distributed.run --nproc_per_node=1 --master_port=29504 tests/test_trainer.py
PYTHONPATH=. python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29505 tests/test_cp.py
PYTHONPATH=. python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29506 tests/test_pp.py
```

## Test Status — All Passing

| Test | Ranks | What it verifies | Status |
|------|-------|-----------------|--------|
| `test_cli.py` | 1 | 31 tests: parser, data pipeline, bench utils, CLI commands | ✅ |
| `test_fp8_flow_moe.py` | 1 | 8 tests: quantize→dequantize, transpose, permute, SwiGLU, recipes, integration, fallback | ✅ |
| `test_tp_correctness.py` | 2 | TP matches non-TP numerically (max_diff=0.000001) | ✅ |
| `test_fsdp.py` | 2 | FSDP forward+backward matches non-FSDP (loss_diff=0.000000) | ✅ |
| `test_tp_fsdp.py` | 4 | TP=2 + DP=2 composition — 28/28 params grad flow | ✅ |
| `test_trainer.py` | 1 | 20-step training with callback lifecycle verification | ✅ |
| `test_cp.py` | 2 | Ulysses CP matches standard attention (max_diff=0.000000) | ✅ |
| `test_pp.py` | 2 | split_model, bubble fractions, stage forward/backward | ✅ |
| `test_moe.py` | 1 | 10 MoE tests: forward, aux loss, bmm, EP, gradient, DeepSeekV3, capacity, shared expert, sigmoid | ✅ |
| `test_hybrid_moe.py` | 1 | 11 hybrid tests: forward, capacity, shared expert, sigmoid, replace_block, aux loss, bias, auto-schedule, no-dense, TP, dense-only | ✅ |

## Novel Contributions

| Contribution | File | Description |
|-------------|------|-------------|
| **FP8-Flow-MoE** | `fp8_flow_moe/` | Standalone pip package. Only open-source implementation of the MLSys 2026 Oral FP8-Flow-MoE recipe. |
| **Auto-Pipeline-Schedule** | `distributed/auto_schedule.py` | Generates optimal PP schedules from stage profiles using DAG-based solver. Supports 1F1B and DualPipe. |
| **MoE Hybrid Block** | `model/transformer.py:172` | Overlaps EP all-to-all with dense FFN computation. |
| **DualPipe Schedule** | `distributed/pipeline_parallel.py:250` | DeepSeek-V3 bidirectional PP: microbatches enter from both ends. |
| **DeepSeek-V3 Routing** | `model/moe.py:82` | Auxiliary-loss-free routing via learnable score bias. |
| **Distributed Checkpointer** | `checkpoint/distributed.py` | Atomic saves, async via ThreadPoolExecutor, RNG state per rank. |

## Environment

- Python 3.9+, PyTorch 2.4+
- Runs on CPU (gloo backend) for testing
- CUDA required for: FP8 (torchao), Flash Attention, grouped_gemm, NCCL collectives
