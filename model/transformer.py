"""Transformer — full model assembly with composable parallelism.

Matches OLMo-core nn.transformer pattern:
  - TransformerBlock with apply_tp/cp/fsdp methods
  - Transformer with apply_tp/cp/pp/fsdp/compile/activation_checkpointing
  - ResidualStream with alpha scaling (simplified here as x = x + ...)
  - Config-driven build pattern

Reference: OLMo-core src/olmo_core/nn/transformer/model.py
           OLMo-core src/olmo_core/nn/transformer/block.py
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from nanopsyche.model.norm import RMSNorm
from nanopsyche.model.attention import Attention
from nanopsyche.model.feed_forward import FeedForward
from nanopsyche.model.moe import MoEBase

if TYPE_CHECKING:
    pass


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm residual connections.

    :param d_model: model dimensionality.
    :param block_idx: block index (0-based).
    :param n_layers: total number of blocks (for scaling).
    :param attention: pre-built attention module.
    :param feed_forward: pre-built feed-forward module (or MoE).
    :param attention_norm: pre-built norm before attention.
    :param feed_forward_norm: pre-built norm before feed-forward.
    :param dropout: dropout on residual connections.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int = 0,
        n_layers: int = 1,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rope_base: float = 10000.0,
        max_seq_len: int = 8192,
        qk_norm: bool = False,
        ffn_hidden: Optional[int] = None,
        attention: Optional[Attention] = None,
        feed_forward: Optional[nn.Module] = None,
        attention_norm: Optional[RMSNorm] = None,
        feed_forward_norm: Optional[RMSNorm] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_idx = block_idx

        self.attention_norm = attention_norm or RMSNorm(d_model)
        if attention is not None:
            self.attention = attention
        else:
            self.attention = Attention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
                qk_norm=qk_norm,
            )

        self.feed_forward_norm = feed_forward_norm or RMSNorm(d_model)
        if feed_forward is not None:
            self.feed_forward = feed_forward
        else:
            from nanopsyche.model.feed_forward import FeedForward

            self.feed_forward = FeedForward(d_model=d_model, hidden_size=ffn_hidden)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        causal: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Pre-norm residual block.

        x = x + dropout(Attn(attention_norm(x)))
        x = x + dropout(FFN(feed_forward_norm(x)))
        """
        h = x + self.dropout(
            self.attention(
                self.attention_norm(x), causal=causal, attn_mask=attn_mask, **kwargs
            )
        )
        x = h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))
        return x

    def apply_tp(self, tp_mesh: DeviceMesh):
        """Apply Megatron-style tensor parallelism to this block.

        Activations stay full d_model. Only weight matrices are sharded.

        :param tp_mesh: tensor parallel DeviceMesh.
        """
        self.attention.apply_tp(tp_mesh)
        self.feed_forward.apply_tp(tp_mesh)

    def apply_cp(self, cp_mesh: DeviceMesh, *, ring=None, uly=None):
        """Apply context parallelism via attention backend."""
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        *,
        wrapping_strategy: str = "full",
        reshard_after_forward: bool = True,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        prefetch_factor: int = 0,
        **fsdp_kwargs,
    ):
        """Apply FSDP to this block.

        :param dp_mesh: data parallel DeviceMesh.
        :param wrapping_strategy: "full" or "fine_grained".
        :param reshard_after_forward: whether to reshard after forward.
        :param mp_policy: mixed precision policy.
        :param prefetch_factor: number of blocks to prefetch.
        """
        if dp_mesh is None:
            return

        fsdp_kwargs.setdefault("mesh", dp_mesh)
        if mp_policy is not None:
            fsdp_kwargs["mp_policy"] = mp_policy
        fsdp_kwargs["reshard_after_forward"] = reshard_after_forward

        if wrapping_strategy == "fine_grained":
            # Fine-grained: shard attention, FFN, and block root separately
            fsdp_att = fully_shard(self.attention, **fsdp_kwargs)
            fsdp_ffn = fully_shard(self.feed_forward, **fsdp_kwargs)
            fsdp_root = fully_shard(self, **fsdp_kwargs)
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_ffn])
        else:
            # Full: shard entire block
            fully_shard(self, **fsdp_kwargs)


class Transformer(nn.Module):
    """Full transformer with composable parallelism.

    :param d_model: model dimensionality.
    :param vocab_size: vocabulary size.
    :param n_layers: number of transformer blocks.
    :param block: pre-built block config or dict of block configs.
    :param lm_head: optional pre-built LM head.
    :param embedding_norm: optional norm after embeddings.
    :param dtype: datatype for parameters.
    :param init_device: device for initialization.
    :param tie_word_embeddings: tie embedding and LM head weights.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: Optional[Dict[str, object]] = None,
        lm_head: Optional[nn.Linear] = None,
        embedding_norm: Optional[RMSNorm] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        tie_word_embeddings: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dtype = dtype

        # Embeddings
        self.embeddings = nn.Embedding(
            vocab_size, d_model, dtype=dtype, device=init_device
        )
        self.embedding_norm = embedding_norm

        # Transformer blocks
        self.blocks = nn.ModuleDict()
        for block_idx in range(n_layers):
            block_config = block or {}
            self.blocks[str(block_idx)] = TransformerBlock(
                d_model=d_model,
                block_idx=block_idx,
                n_layers=n_layers,
                **block_config,
            )

        # LM head
        self.lm_head = lm_head or nn.Linear(
            d_model, vocab_size, bias=False, dtype=dtype, device=init_device
        )

        self.tie_word_embeddings = tie_word_embeddings
        if tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

        # State flags
        self._pp_enabled = False
        self._tp_enabled = False
        self._fsdp_enabled = False
        self._compile_enabled = False

    @property
    def num_params(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_non_embedding_params(self) -> int:
        """Number of parameters excluding embeddings."""
        return self.num_params - self.embeddings.weight.numel()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the full transformer.

        :param input_ids: (B, S) — token indices.
        :param labels: (B, S) — target indices for loss (optional).
        :returns: dict with logits, loss, and optional aux_loss.
        """
        B, S = input_ids.shape

        # Embeddings
        x = self.embeddings(input_ids)
        if self.embedding_norm is not None:
            x = self.embedding_norm(x)

        # Transformer blocks
        for block in self.blocks.values():
            x = block(x, **kwargs)

        # LM head
        logits = self.lm_head(x)

        output: dict[str, torch.Tensor] = {"logits": logits}

        # Loss
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        # MoE aux losses
        aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks.values():
            if hasattr(block.feed_forward, "aux_loss"):
                al = block.feed_forward.aux_loss
                if al is not None:
                    aux_loss = aux_loss + al
        if aux_loss.item() > 0:
            output["aux_loss"] = aux_loss

        return output

    def apply_tp(self, tp_mesh: DeviceMesh):
        """Apply Megatron-style tensor parallelism to the entire model.

        In Megatron TP, activations stay full d_model on each rank.
        Only weight matrices are sharded:
          - Q/K/V: ColwiseParallel (split output dim → each rank computes subset of heads)
          - Output proj: RowwiseParallel (split input dim → all-reduce after)
          - FFN: same pattern
        Embedding and LM head are replicated (not sharded).

        :param tp_mesh: tensor parallel DeviceMesh.
        """
        for block in self.blocks.values():
            block.apply_tp(tp_mesh)

        self._tp_size = tp_mesh.size()
        self._tp_enabled = True

    def apply_cp(self, cp_mesh: DeviceMesh, *, ring=None, uly=None):
        """Apply context parallelism.

        :param cp_mesh: context parallel DeviceMesh.
        :param ring: ring context parallel style.
        :param uly: ulysses context parallel style.
        """
        for block in self.blocks.values():
            block.apply_cp(cp_mesh, ring=ring, uly=uly)

    def apply_pp(self, pp_mesh: DeviceMesh):
        """Apply pipeline parallelism (marks model as PP-enabled).

        :param pp_mesh: pipeline parallel DeviceMesh.
        """
        self._pp_enabled = True

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        *,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        wrapping_strategy: str = "full",
        reshard_after_forward: bool = True,
        prefetch_factor: int = 0,
    ):
        """Apply FSDP to the entire model.

        :param dp_mesh: data parallel DeviceMesh.
        :param param_dtype: parameter dtype for mixed precision.
        :param reduce_dtype: gradient reduction dtype.
        :param wrapping_strategy: "full" or "fine_grained".
        :param reshard_after_forward: whether to reshard after forward.
        :param prefetch_factor: number of blocks to prefetch.
        """
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype,
            reduce_dtype=reduce_dtype,
        )

        for block in self.blocks.values():
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                wrapping_strategy=wrapping_strategy,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
                prefetch_factor=prefetch_factor,
            )

        # Embeddings
        if not self.tie_word_embeddings:
            fully_shard(
                self.embeddings,
                mesh=dp_mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )

        # LM head
        if not self.tie_word_embeddings:
            fully_shard(
                self.lm_head,
                mesh=dp_mesh,
                reshard_after_forward=False,
                mp_policy=mp_policy,
            )

        # Root FSDP wrapper
        fully_shard(
            self,
            mesh=dp_mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )
        self._fsdp_enabled = True

    def apply_compile(self):
        """Apply torch.compile to each block."""
        for block in self.blocks.values():
            block = torch.compile(block)
        self._compile_enabled = True

    def apply_activation_checkpointing(
        self,
        mode: str = "full",
        block_interval: Optional[int] = None,
    ):
        """Apply activation checkpointing.

        :param mode: "full" (every block), "selected_blocks" (every N-th).
        :param block_interval: for "selected_blocks", checkpoint every N-th.
        """
        from torch.utils.checkpoint import checkpoint_wrapper

        for i, (name, block) in enumerate(self.blocks.items()):
            if mode == "full" or (
                mode == "selected_blocks" and block_interval and i % block_interval == 0
            ):
                self.blocks[name] = checkpoint_wrapper(block)
