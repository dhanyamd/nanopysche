"""nanopsyche CLI — train and benchmark MoE models.

Usage:
    nanopsyche train --model 125m --use-moe --fp8-recipe flow_moe --dataset c4
    nanopsyche bench --model 1b --use-moe --validate
"""

import argparse
import os
import sys
import time as time_module
from pathlib import Path
from typing import Optional

import torch

from nanopsyche.bench_utils import compute_mfu


MODEL_PRESETS = {
    "125m": dict(d_model=768, n_layers=12, n_heads=12, ffn_hidden=2048, n_kv_heads=12),
    "350m": dict(d_model=1024, n_layers=24, n_heads=16, ffn_hidden=2816, n_kv_heads=16),
    "1b": dict(d_model=2048, n_layers=24, n_heads=32, ffn_hidden=5632, n_kv_heads=32),
    "7b": dict(d_model=4096, n_layers=32, n_heads=32, ffn_hidden=11008, n_kv_heads=32),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanopsyche",
        description="Production-grade distributed training framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model",
        "-m",
        default="125m",
        choices=list(MODEL_PRESETS),
        help="Model size preset",
    )
    common.add_argument("--use-moe", action="store_true", help="Enable MoE layers")
    common.add_argument(
        "--num-experts", type=int, default=8, help="Number of MoE experts"
    )
    common.add_argument("--top-k", type=int, default=2, help="MoE top-k routing")
    common.add_argument(
        "--fp8-recipe",
        default="none",
        choices=["none", "blockwise", "mxfp8", "flow_moe"],
        help="FP8 training recipe",
    )
    common.add_argument(
        "--fp8-gemm-threshold",
        type=int,
        default=1000000,
        help=(
            "Min tokens/expert before FP8 GEMM beats BF16 bmm "
            "(FP8-Flow-MoE adaptive routing). 0 forces FP8 GEMM always."
        ),
    )
    common.add_argument(
        "--batch-size", type=int, default=4, help="Micro batch size per GPU"
    )
    common.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    common.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HF dataset name or .npy/.bin file for training data",
    )
    common.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    common.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    common.add_argument("--max-steps", type=int, default=100, help="Training steps")
    common.add_argument(
        "--model-factory",
        type=str,
        default=None,
        metavar="MODULE:CALLABLE",
        help=(
            "Custom model factory (e.g. my_pkg.models:build_model). "
            "Receives ModelBuildContext; returns nn.Module or (model, opt, sched). "
            "When omitted, uses the built-in Transformer preset from --model."
        ),
    )

    # train
    train_p = sub.add_parser("train", parents=[common], help="Train a model")
    train_p.add_argument("--compile", action="store_true", help="Enable torch.compile")
    train_p.add_argument(
        "--val-interval",
        type=int,
        default=10,
        help="Validation every N steps (0 = no validation)",
    )
    train_p.add_argument(
        "--val-steps",
        type=int,
        default=5,
        help="Validation steps (samples to evaluate)",
    )
    train_p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    train_p.add_argument(
        "--wandb-project",
        type=str,
        default="nanopsyche",
        help="WandB project name",
    )
    train_p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    train_p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Checkpoint save directory",
    )
    train_p.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Checkpoint save interval (steps)",
    )
    train_p.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism degree (shard weights along hidden dim)",
    )
    train_p.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallelism degree (split layers across stages)",
    )
    train_p.add_argument(
        "--cp-size",
        type=int,
        default=1,
        help="Context parallelism degree (shard sequence across ranks)",
    )
    train_p.add_argument(
        "--ep-size",
        type=int,
        default=1,
        help="Expert parallelism degree (shard experts across ranks)",
    )
    train_p.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP (Fully Sharded Data Parallelism)",
    )
    train_p.add_argument(
        "--ac-mode",
        type=str,
        default=None,
        choices=["full", "selected_blocks"],
        help="Activation checkpointing mode",
    )

    # bench
    bench_p = sub.add_parser(
        "bench", parents=[common], help="Benchmark model throughput"
    )
    bench_p.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    bench_p.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    bench_p.add_argument(
        "--validate",
        action="store_true",
        help="Compute validation perplexity after benchmark",
    )

    return parser


def load_dataset(
    dataset_spec: str,
    batch_size: int,
    seq_len: int,
    split: str = "train",
    val_split: str = "validation",
    device: str = "cpu",
):
    """Load a dataset from HuggingFace or a local tokenized file.

    :param dataset_spec: HF dataset name (e.g. "c4", "HuggingFaceFW/fineweb")
                         or path to local .npy file.
    :param batch_size: micro batch size.
    :param seq_len: sequence length.
    :param split: dataset split name.
    :param val_split: validation split name.
    :param device: target device.
    :returns: (train_loader, val_loader, vocab_size) or (None, None, vocab_size).
    """
    from nanopsyche.train.data import create_distributed_loader

    if dataset_spec is None:
        return None, None, 32000

    if dataset_spec.endswith(".npy") or dataset_spec.endswith(".bin"):
        import numpy as np

        data = np.memmap(dataset_spec, dtype=np.uint16, mode="r")
        # Simple 90/10 split
        split_idx = int(len(data) * 0.9)
        train_data, val_data = data[:split_idx], data[split_idx:]
        train_loader = create_distributed_loader(train_data, batch_size, seq_len)
        val_loader = create_distributed_loader(val_data, batch_size, seq_len)
        return train_loader, val_loader, 32000

    # HuggingFace datasets
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print("  [data] Install 'datasets' for HuggingFace data: pip install datasets")
        return None, None, 32000

    print(f"  Loading dataset '{dataset_spec}' ({split} split)...")
    ds = hf_load(dataset_spec, split=split, streaming=True)
    print(f"  Tokenizing with GPT-2 tokenizer...")

    # Try fast tokenizers first, fall back to transformers
    tok = None
    vocab_size = 32000
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained("gpt2")
        vocab_size = tok.get_vocab_size()
    except Exception:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tok.vocab_size

    def _encode(text: str):
        """Encode text, handling both tokenizers.Encoding and list returns."""
        if tok is None:
            return []
        ids = tok.encode(text)
        # tokenizers.Encoding has .ids; transformers.AutoTokenizer returns list
        return ids.ids if hasattr(ids, "ids") else ids

    def tokenize_fn(example):
        """Tokenize a single example (not a batch)."""
        text = example.get("text") or example.get("content") or ""
        if isinstance(text, list):
            text = text[0] if text else ""
        return _encode(text)

    # Tokenize into flat array
    tokens = []
    for i, example in enumerate(ds):
        encoded = tokenize_fn(example)
        if isinstance(encoded, list):
            tokens.extend(encoded)
        if len(tokens) >= 1_000_000:
            break

    import numpy as np

    tokens_arr = np.array(tokens[: (len(tokens) // seq_len) * seq_len], dtype=np.int64)
    split_idx = int(len(tokens_arr) * 0.95)
    train_data, val_data = tokens_arr[:split_idx], tokens_arr[split_idx:]

    train_loader = create_distributed_loader(train_data, batch_size, seq_len)
    val_loader = create_distributed_loader(val_data, batch_size, seq_len)
    print(
        f"  Train: {len(train_data) // seq_len} sequences | Val: {len(val_data) // seq_len} sequences"
    )
    return train_loader, val_loader, vocab_size


def compute_val_perplexity(model, val_loader, max_batches: int, device: str) -> float:
    """Compute validation perplexity.

    perplexity = exp(cross_entropy_loss)

    :param model: the model.
    :param val_loader: validation data loader (None = return inf).
    :param max_batches: maximum batches to evaluate.
    :param device: device to evaluate on.
    :returns: perplexity as a float.
    """
    if val_loader is None:
        return float("inf")
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            output = model(input_ids, labels=labels)
            total_loss += output["loss"].item()
            n_batches += 1
    model.train()
    if n_batches == 0:
        return float("inf")
    avg_loss = total_loss / n_batches
    return float(torch.exp(torch.tensor(avg_loss)).item())


def _model_build_context(args: argparse.Namespace, vocab_size: int):
    from nanopsyche.model_factory import ModelBuildContext

    factory_spec = getattr(args, "model_factory", None)
    preset = None if factory_spec else MODEL_PRESETS[args.model]
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    return ModelBuildContext(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        device=args.device,
        model_preset=args.model,
        preset=preset,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        moe_top_k=args.top_k,
        fp8_recipe=args.fp8_recipe,
        fp8_gemm_threshold=args.fp8_gemm_threshold,
        dtype=dtype,
    )


def _build_optimizer_and_scheduler(
    model, args: argparse.Namespace, factory_opt=None, factory_sched=None
):
    import torch.optim as optim
    from nanopsyche.train.scheduler import CosineWithWarmup

    if factory_opt is not None:
        optimizer = factory_opt
        scheduler = factory_sched or CosineWithWarmup(
            optimizer,
            warmup_steps=min(10, args.max_steps // 10),
            max_steps=args.max_steps,
        )
        return optimizer, scheduler

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    scheduler = CosineWithWarmup(
        optimizer,
        warmup_steps=min(10, args.max_steps // 10),
        max_steps=args.max_steps,
    )
    return optimizer, scheduler


def _maybe_parallelize_model(model, args: argparse.Namespace):
    tp_size = getattr(args, "tp_size", 1)
    pp_size = getattr(args, "pp_size", 1)
    cp_size = getattr(args, "cp_size", 1)
    ep_size = getattr(args, "ep_size", 1)
    fsdp = getattr(args, "fsdp", False)
    ac_mode = getattr(args, "ac_mode", None)
    compile_model = getattr(args, "compile", False)

    has_parallelism = any(
        [
            tp_size > 1,
            pp_size > 1,
            cp_size > 1,
            ep_size > 1,
            fsdp,
            ac_mode is not None,
        ]
    )

    if not has_parallelism:
        if compile_model:
            if hasattr(model, "apply_compile"):
                model.apply_compile()
            else:
                model = torch.compile(model)
        return model

    import torch.distributed as dist

    if dist.is_available() and "RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        args.device = torch.device(
            f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(args.device)
        model.to(args.device)

    from nanopsyche.parallel import build_world_mesh
    from nanopsyche.parallelize import parallelize_model
    from nanopsyche.distributed.tensor_parallel import TensorParallelConfig
    from nanopsyche.distributed.context_parallel import ContextParallelConfig
    from nanopsyche.distributed.pipeline_parallel import PipelineParallelConfig
    from nanopsyche.distributed.fsdp import DataParallelConfig

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    world_mesh = build_world_mesh(
        world_size=world_size,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        device_type=device_type,
    )

    tp_config = TensorParallelConfig(degree=tp_size) if tp_size > 1 else None
    cp_config = ContextParallelConfig(degree=cp_size) if cp_size > 1 else None
    pp_config = PipelineParallelConfig(degree=pp_size) if pp_size > 1 else None
    dp_config = DataParallelConfig(name="fsdp") if fsdp else None

    return parallelize_model(
        model,
        world_mesh=world_mesh,
        device=args.device,
        tp_config=tp_config,
        cp_config=cp_config,
        pp_config=pp_config,
        dp_config=dp_config,
        moe_fp8_recipe=args.fp8_recipe,
        moe_fp8_gemm_threshold=args.fp8_gemm_threshold,
        compile_model=compile_model,
        ac_mode=ac_mode,
    )


def cmd_train(args: argparse.Namespace):
    from nanopsyche.config import TrainingConfig, ExperimentConfig
    from nanopsyche.model_factory import build_model
    from nanopsyche.model.protocol import count_parameters, model_display_name

    factory_spec = getattr(args, "model_factory", None)

    # Load data first to get actual vocab_size
    train_loader, val_loader, vocab_size = load_dataset(
        args.dataset,
        args.batch_size,
        args.seq_len,
        device=args.device,
    )
    if vocab_size is None:
        vocab_size = 32000

    config = ExperimentConfig(
        name=f"train-{args.model}" if not factory_spec else f"train-custom",
        model_factory=factory_spec,
        training=TrainingConfig(
            micro_batch_size=args.batch_size,
            sequence_length=args.seq_len,
            learning_rate=args.lr,
            max_steps=args.max_steps,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
        ),
    )

    ctx = _model_build_context(args, vocab_size)
    model, factory_opt, factory_sched = build_model(ctx, factory_spec=factory_spec)
    model.to(args.device)
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model, args, factory_opt, factory_sched
    )
    model = _maybe_parallelize_model(model, args)

    model_config = count_parameters(model)
    display = model_display_name(model, args.model if not factory_spec else None)

    print(f"\nModel: {display} ({model_config['num_params'] // 1000**2}M params)")
    print(f"MoE: {args.use_moe} | Experts: {args.num_experts} | Top-k: {args.top_k}")
    print(f"FP8: {args.fp8_recipe} | Batch: {args.batch_size} | Seq: {args.seq_len}")
    print(f"Steps: {args.max_steps} | LR: {args.lr} | Device: {args.device}")

    if train_loader is not None:
        print(f"Data: {args.dataset}")

    from nanopsyche.train.trainer import TrainingConfig as TCT, Trainer
    from nanopsyche.checkpoint.distributed import (
        DistributedCheckpointer,
        CheckpointConfig,
    )

    save_dir = args.save_dir or f"checkpoints/{args.model}"

    tc = TCT(
        global_batch_size=args.batch_size,
        micro_batch_size=args.batch_size,
        sequence_length=args.seq_len,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        save_dir=save_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    checkpointer = DistributedCheckpointer(
        CheckpointConfig(
            save_dir=save_dir,
            save_interval=args.save_interval,
        )
    )
    trainer = Trainer(
        tc,
        model,
        optimizer,
        scheduler,
        data_loader=train_loader,
        checkpointer=checkpointer,
    )

    # Wire production callbacks
    from nanopsyche.train.callbacks import (
        ConsoleLoggerCallback,
        SpeedMonitorCallback,
        GPUMemoryMonitorCallback,
        StabilityMonitorCallback,
        WandBCallback,
    )

    trainer.callbacks.append(
        SpeedMonitorCallback(
            device=torch.device(args.device), batch_size=args.batch_size * args.seq_len
        )
    )
    if args.device == "cuda":
        trainer.callbacks.append(
            GPUMemoryMonitorCallback(device=torch.device(args.device))
        )
    trainer.callbacks.append(StabilityMonitorCallback())
    trainer.callbacks.append(ConsoleLoggerCallback())

    if args.wandb:
        trainer.callbacks.append(
            WandBCallback(project=args.wandb_project, name=config.name)
        )

    # Attach validation callback
    if val_loader is not None and args.val_interval > 0:
        from nanopsyche.train.callbacks.base import Callback

        class ValidationCallback(Callback):
            def post_step(self, trainer):
                step = trainer.step
                if step % args.val_interval == 0 and step > 0:
                    ppl = compute_val_perplexity(
                        model, val_loader, args.val_steps, args.device
                    )
                    trainer.record_metric("eval/perplexity", ppl)
                    if trainer.is_main:
                        print(f"  step={step:6d} | val perplexity={ppl:.4f}")

            def log_metrics(self, metrics, step, trainer):
                if "eval/perplexity" in metrics:
                    print(
                        f"  step={step:6d} | val perplexity={metrics['eval/perplexity']:.4f}"
                    )

        trainer.callbacks.append(ValidationCallback())

    # Resume from checkpoint if requested
    if args.resume:
        resume_path = Path(args.resume)
        if checkpointer is not None and resume_path.is_dir():
            # Load from distributed checkpoint directory
            metadata = checkpointer.load(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            trainer.step = metadata.get("step", 0)
            print(f"Resumed from checkpoint step {trainer.step}")
        else:
            trainer.load_checkpoint(args.resume)

    trainer.fit()


def cmd_bench(args: argparse.Namespace):
    from nanopsyche.bench_utils import get_peak_flops
    from nanopsyche.model_factory import build_model
    from nanopsyche.model.protocol import count_parameters, model_display_name

    torch.manual_seed(42)
    factory_spec = getattr(args, "model_factory", None)

    # Load data first to get actual vocab_size (needed for model construction)
    train_loader, val_loader, vocab_size = load_dataset(
        args.dataset,
        args.batch_size,
        args.seq_len,
        device=args.device,
    )
    if vocab_size is None:
        vocab_size = 32000

    ctx = _model_build_context(args, vocab_size)
    model, _, _ = build_model(ctx, factory_spec=factory_spec)
    model.to(args.device)
    model = _maybe_parallelize_model(model, args)
    model.train()

    model_config = count_parameters(model)
    display = model_display_name(model, args.model if not factory_spec else None)
    num_params = model_config["num_params"]
    num_non_embed = model_config["num_non_embed_params"]

    B, S = args.batch_size, args.seq_len
    x = torch.randint(0, vocab_size, (B, S), device=args.device)

    # Warmup
    for _ in range(args.warmup):
        model(x)

    if args.device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            model(x)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
    else:
        t0 = time_module.time()
        for _ in range(args.iters):
            model(x)
        t1 = time_module.time()
        elapsed = t1 - t0

    step_time = elapsed / args.iters
    tokens_per_step = B * S
    tokens_per_sec = (B * S * args.iters) / elapsed
    tokens_per_sec_per_gpu = tokens_per_sec  # single GPU

    device = torch.device(args.device)
    peak = get_peak_flops(device, fp8=(args.fp8_recipe != "none"))

    if factory_spec:
        bench_preset = {
            "n_layers": getattr(model, "n_layers", 1),
            "d_model": getattr(model, "d_model", 768),
            "ffn_hidden": getattr(model, "ffn_hidden", 2048),
            "n_heads": getattr(model, "n_heads", 12),
        }
    else:
        bench_preset = MODEL_PRESETS[args.model]

    mfu = compute_mfu(
        step_time_seconds=step_time,
        tokens_per_step=tokens_per_step,
        num_params=num_params,
        n_layers=bench_preset["n_layers"],
        d_model=bench_preset["d_model"],
        ffn_hidden=bench_preset.get("ffn_hidden")
        or int(8 / 3 * bench_preset["d_model"]),
        n_heads=bench_preset["n_heads"],
        seq_len=S,
        batch_size=B,
        device=device,
        fp8=(args.fp8_recipe != "none"),
        num_gpus=1,
        num_experts=args.num_experts if args.use_moe else 0,
        top_k=args.top_k if args.use_moe else 0,
    )

    print(f"\n{'=' * 65}")
    print(
        f"  Model:      {display} ({num_params // 1000**2}M total, {num_non_embed // 1000**2}M non-embed)"
    )
    print(
        f"  MoE:        {args.use_moe} | Experts: {args.num_experts} | Top-k: {args.top_k}"
    )
    print(f"  FP8:        {args.fp8_recipe}")
    print(
        f"  Device:     {torch.cuda.get_device_name(device) if args.device == 'cuda' else 'CPU'}"
    )
    print(f"  Peak FLOPs: {peak / 1e12:.0f} TFLOPS")
    print(f"  {'─' * 65}")
    print(f"  Batch:      {B} x {S}")
    print(f"  Tokens/s:   {tokens_per_sec:>10,.0f}")
    print(f"  Tokens/s/GPU: {tokens_per_sec_per_gpu:>7,.0f}")
    print(f"  ms/step:    {step_time * 1000:>8.1f}")
    print(f"  MFU:        {mfu * 100:>6.2f}%")
    print(f"{'=' * 65}")

    # Validation perplexity with real data if available
    if args.validate and val_loader is not None:
        ppl = compute_val_perplexity(model, val_loader, args.iters, args.device)
        print(f"  Val perplexity: {ppl:.4f}")
        print(f"{'=' * 65}")


def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        cmd_train(args)
    elif args.command == "bench":
        cmd_bench(args)


if __name__ == "__main__":
    main()
