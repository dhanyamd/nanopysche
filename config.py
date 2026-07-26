"""Config system — dataclass-based with build() pattern.

Matches OLMo-core Config/ModuleConfig pattern:
    1. Define configs as dataclasses
    2. Config.build() constructs the actual object
    3. CLI overrides via dot-notation
    4. Serialization to/from JSON

Reference: OLMo-core src/olmo_core/config.py
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class Config:
    """Base config class with serialization and merge support."""

    def build(self, **kwargs):
        raise NotImplementedError

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def merge(self, overrides: dict[str, Any]) -> "Config":
        import dataclasses

        fields_dict = {f.name: f for f in dataclasses.fields(self)}
        new_values = {}
        for key, value in overrides.items():
            if key in fields_dict:
                new_values[key] = value
        return dataclasses.replace(self, **new_values)

    @classmethod
    def from_cli(cls, args: list[str]) -> "Config":
        overrides = {}
        i = 0
        while i < len(args):
            if args[i].startswith("--"):
                key = args[i][2:]
                value = args[i + 1]
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() in ("true", "false"):
                            value = value.lower() == "true"
                overrides[key] = value
                i += 2
            else:
                i += 1
        return cls().merge(overrides)


@dataclass
class TransformerConfig(Config):
    """Transformer model configuration."""

    vocab_size: int = 32000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rope_base: float = 500000.0
    max_seq_len: int = 8192
    qk_norm: bool = True
    ffn_hidden: Optional[int] = None
    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    tie_word_embeddings: bool = False
    dropout: float = 0.0

    # Parallelism
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    dp_size: int = 1

    # FSDP
    fsdp_wrapping: str = "full"
    fsdp_mixed_precision: Optional[str] = "bf16"

    # Activation checkpointing
    activation_checkpointing: Optional[str] = None
    checkpoint_interval: Optional[int] = None

    # torch.compile
    compile: bool = False

    def build(self, **kwargs):
        from nanopsyche.model.transformer import Transformer

        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        ffn_hidden = self.ffn_hidden or int(8 / 3 * self.d_model)
        ffn_hidden = ((ffn_hidden + 255) // 256) * 256

        block_config = dict(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=self.head_dim,
            rope_base=self.rope_base,
            max_seq_len=self.max_seq_len,
            qk_norm=self.qk_norm,
            ffn_hidden=ffn_hidden,
            dropout=self.dropout,
        )
        block_config.update(kwargs.pop("block", {}))

        return Transformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            block=block_config,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )


@dataclass
class TrainingConfig(Config):
    """Training configuration."""

    global_batch_size: int = 1024
    micro_batch_size: int = 4
    sequence_length: int = 2048
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 600000
    schedule: str = "cosine"
    stable_steps: int = 0
    decay_steps: int = 0
    dtype: str = "bfloat16"
    save_interval: int = 1000
    save_dir: str = "checkpoints"
    log_interval: int = 10

    def build(self, **kwargs):
        import torch
        from nanopsyche.train.trainer import TrainingConfig as TC

        return TC(
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            sequence_length=self.sequence_length,
            learning_rate=self.learning_rate,
            min_lr=self.min_lr,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            dtype=getattr(torch, self.dtype),
            save_interval=self.save_interval,
            save_dir=self.save_dir,
            **kwargs,
        )


@dataclass
class ExperimentConfig(Config):
    """Full experiment configuration."""

    name: str = "experiment"
    model: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42

    def build(self, **kwargs):
        model = self.model.build()
        import torch.optim as optim
        from nanopsyche.train.scheduler import CosineWithWarmup

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
            betas=(0.9, 0.95),
        )
        scheduler = CosineWithWarmup(
            optimizer,
            warmup_steps=self.training.warmup_steps,
            max_steps=self.training.max_steps,
        )
        config = self.training.build()
        return model, optimizer, scheduler, config
