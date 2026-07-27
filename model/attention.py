"""Attention — Multi-head, Grouped-query, and Multi-query attention.

Matches OLMo-core nn.attention.Attention pattern:
  - w_q, w_k, w_v, w_out naming
  - Optional w_g gate, q_norm, k_norm
  - Backend abstraction for attention computation
  - apply_tp() shards weights across TP ranks, forward uses local compute + all-reduce
  - apply_cp() for context parallelism
  - sdpa() method for the core attention computation

TP approach (Megatron-LM style):
  - Colwise sharding: split output dim of Q/K/V projections across TP ranks
  - Rowwise sharding: split input dim of output projection, all-reduce after
  - Each rank holds n_heads/tp heads, local compute, one all-reduce at the end

Reference: Shoeybi et al. 2019 (Megatron-LM)
           Ainslie et al. 2023 (GQA)
           OLMo-core src/olmo_core/nn/attention/__init__.py
"""

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh

from nanopsyche.model.rope import RotaryEmbedding, RoPEScalingConfig, apply_rotary_emb
from nanopsyche.model.tp_utils import differentiable_all_reduce

if TYPE_CHECKING:
    pass


class Attention(nn.Module):
    """Multi-head self-attention with GQA/MQA support.

    :param d_model: model dimensionality.
    :param n_heads: number of attention heads.
    :param n_kv_heads: number of KV heads (for GQA). None = n_heads (MHA).
    :param bias: whether to use bias in projections.
    :param rope_base: RoPE theta parameter.
    :param max_seq_len: maximum sequence length for RoPE precomputation.
    :param head_dim: head dimension. None = d_model // n_heads.
    :param qk_norm: apply LayerNorm to Q and K before attention.
    :param rope_head_first: RoPE layout (True = (B,S,H,D), False = (B,H,S,D)).
    :param rope_scaling: optional RoPE scaling config for context extension.
    :param partial_rotary_factor: fraction of head_dim to rotate.
    :param output_dropout: dropout on attention output.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        bias: bool = False,
        rope_base: float = 10000.0,
        max_seq_len: int = 8192,
        head_dim: Optional[int] = None,
        qk_norm: bool = False,
        rope_head_first: bool = True,
        rope_scaling: Optional[RoPEScalingConfig] = None,
        partial_rotary_factor: float = 1.0,
        output_dropout: float = 0.0,
    ):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim or d_model // n_heads
        self.d_model = d_model
        self.rope_head_first = rope_head_first
        self._tp_size = 1
        self._tp_group = None

        total_q_dim = n_heads * self.head_dim
        total_kv_dim = n_kv_heads * self.head_dim

        self.w_q = nn.Linear(d_model, total_q_dim, bias=bias)
        self.w_k = nn.Linear(d_model, total_kv_dim, bias=bias)
        self.w_v = nn.Linear(d_model, total_kv_dim, bias=bias)
        self.w_out = nn.Linear(total_q_dim, d_model, bias=bias)

        self.rope = RotaryEmbedding(
            self.head_dim,
            base=rope_base,
            max_seq_len=max_seq_len,
            head_first=rope_head_first,
            partial_rotary_factor=partial_rotary_factor,
            scaling=rope_scaling,
        )

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.w_g: Optional[nn.Linear] = None
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout > 0 else None

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        causal: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        tp = self._tp_size

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        n_heads_local = self.n_heads // tp
        n_kv_heads_local = self.n_kv_heads // tp

        q = q.view(B, S, n_heads_local, self.head_dim)
        k = k.view(B, S, n_kv_heads_local, self.head_dim)
        v = v.view(B, S, n_kv_heads_local, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        cos, sin = self.rope(S, offset=start_pos, device=x.device, dtype=q.dtype)
        q = apply_rotary_emb(q, cos, sin, head_first=self.rope_head_first)
        k = apply_rotary_emb(k, cos, sin, head_first=self.rope_head_first)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        n_rep_local = n_heads_local // n_kv_heads_local
        if n_rep_local > 1:
            k = k.repeat_interleave(n_rep_local, dim=2)
            v = v.repeat_interleave(n_rep_local, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if hasattr(self, "_cp_module") and self._cp_module is not None:
            # Context Parallelism: input is (B, S_local, H, D) shaped
            # CP module handles the sequence-partitioned attention
            att = self._cp_module(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            att = att.transpose(1, 2)
        else:
            att = self.sdpa(q, k, v, causal=causal, attn_mask=attn_mask)
        att = att.transpose(1, 2).reshape(B, S, n_heads_local * self.head_dim)

        if self.output_dropout is not None:
            att = self.output_dropout(att)

        out = self.w_out(att)

        if tp > 1:
            out = differentiable_all_reduce(out, self._tp_group, tp_size=tp)

        return out

    def apply_tp(self, tp_mesh: DeviceMesh):
        """Apply Megatron-style tensor parallelism to this attention module.

        Shards Q/K/V projections colwise (split output dim) and output projection
        rowwise (split input dim). Forward pass does local compute + all-reduce.

        :param tp_mesh: the tensor parallel DeviceMesh.
        """
        tp_size = tp_mesh.size()
        tp_group = tp_mesh.get_group()
        self._tp_size = tp_size
        self._tp_group = tp_group

        rank = dist.get_rank(tp_group)

        # Shard Q projection colwise: split output dim (n_heads * head_dim)
        w_q_weight = self.w_q.weight.view(tp_size, -1, self.w_q.weight.shape[-1])
        self.w_q.weight = nn.Parameter(w_q_weight[rank].contiguous())
        if self.w_q.bias is not None:
            w_q_bias = self.w_q.bias.view(tp_size, -1)
            self.w_q.bias = nn.Parameter(w_q_bias[rank].contiguous())

        # Shard K projection colwise
        w_k_weight = self.w_k.weight.view(tp_size, -1, self.w_k.weight.shape[-1])
        self.w_k.weight = nn.Parameter(w_k_weight[rank].contiguous())
        if self.w_k.bias is not None:
            w_k_bias = self.w_k.bias.view(tp_size, -1)
            self.w_k.bias = nn.Parameter(w_k_bias[rank].contiguous())

        # Shard V projection colwise
        w_v_weight = self.w_v.weight.view(tp_size, -1, self.w_v.weight.shape[-1])
        self.w_v.weight = nn.Parameter(w_v_weight[rank].contiguous())
        if self.w_v.bias is not None:
            w_v_bias = self.w_v.bias.view(tp_size, -1)
            self.w_v.bias = nn.Parameter(w_v_bias[rank].contiguous())

        # Shard output projection rowwise: split input dim
        w_out_weight = self.w_out.weight
        d_out = w_out_weight.shape[0]
        d_out_local = d_out // tp_size
        self.w_out.weight = nn.Parameter(
            w_out_weight[:, rank * d_out_local : (rank + 1) * d_out_local].contiguous()
        )

    def apply_cp(self, cp_mesh: DeviceMesh, *, ring=None, uly=None):
        """Apply context parallelism via Ring Attention or Ulysses.

        Sets up the attention module to use sequence-partitioned KV with
        the specified CP strategy. The actual CP logic runs inside forward()
        when self._cp_module is set.

        :param cp_mesh: context parallel DeviceMesh.
        :param ring: if True, use Ring Attention (P2P KV exchange).
        :param uly: if True, use Ulysses (all-to-all reshape).
        """
        from nanopsyche.distributed.context_parallel import RingAttention

        cp_size = cp_mesh.size()
        cp_group = cp_mesh.get_group()

        use_ulysses = uly if uly is not None else False
        use_zigzag = not use_ulysses  # zig-zag only for ring

        self._cp_size = cp_size
        self._cp_group = cp_group
        self._cp_module = RingAttention(
            group=cp_group,
            use_ulysses=use_ulysses,
            use_zigzag=use_zigzag,
        )
