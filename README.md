# nanopsyche

A minimal, hackable, **from-scratch distributed-training library** — a rebuild of
[Picotron](https://github.com/huggingface/picotron)'s 4D parallelism, extended with
**communication-efficient optimization** and **fault tolerance** that the original lacks.

## The problem it solves

Distributed training today assumes a **pristine, expensive, homogeneous GPU cluster**
with fast interconnect (NVLink / InfiniBand). `nanopsyche` makes distributed training
work on **cheap, slow-networked, heterogeneous, or unreliable hardware** — by cutting
the bytes sent over the wire (compressed optimizer) and surviving node failures
(checkpoint + kill-and-resume) on top of standard 4D parallelism.

This is the [Nous Psyche / DisTrO](https://github.com/NousResearch/DisTrO) thesis:
**democratize *who* can train large models**, beyond the big labs. It doubles as a
readable testbed for comms-efficient-training experiments — something the production
libraries (too complex) and Picotron (too minimal) don't provide.

## Design

Every file small and readable (Picotron philosophy, <300 lines each). Everything built
from scratch and **verified numerically** against PyTorch (`torch.testing.assert_close`)
and against single-GPU results. Nothing is trusted until it matches a reference.

## What it implements

**Core 4D parallelism (replicate Picotron, understand every line):**
- [x] `process_group.py` — ranks, world size, collectives
- [x] `data_parallel.py` — DDP (bucketing, overlap) + FSDP
- [ ] `tensor_parallel.py` — Megatron column/row + sequence parallel
- [ ] `pipeline_parallel.py` — 1F1B (+ optional DualPipe)
- [ ] `context_parallel.py` — Ring Attention
- [ ] `model.py` + `train.py` — small Llama-like + training loop

**Extensions (the differentiators — what Picotron does NOT have):**
- [ ] `optim/` — communication-efficient optimizer (DeMo/DiLoCo-style compression)
- [ ] `fault_tolerance/` — distributed checkpoint + kill-and-resume + hang detection
- [ ] FP8 training (fwd + bwd, loss parity vs bf16)
- [ ] config-driven composition of parallelism + compression + precision
- [ ] hooks to connect to RL (the trainer↔inference seam → nanoatropos)

## Benchmarks (the numbers that make it a portfolio piece)
- MFU (target: production-comparable, like Picotron's ~50%)
- bytes-over-wire: compressed optimizer vs standard all-reduce
- resume time after a killed node
- comms/compute ratio before vs after overlap

## Verify protocol
Every component: (1) grounded in a primary source (paper / PyTorch docs),
(2) built from scratch, (3) verified numerically vs a reference, (4) benchmarked.

## Run (locally, free — CPU/Gloo, 4 fake ranks)
```bash
torchrun --nproc_per_node=4 -m nanopsyche.lessons.lesson_1_1_collectives
torchrun --nproc_per_node=4 -m nanopsyche.lessons.lesson_1_2_ddp
torchrun --nproc_per_node=4 -m nanopsyche.lessons.lesson_1_4_fsdp
```
