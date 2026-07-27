"""Context Parallelism — Ring Attention and Ulysses for long sequences.

Context Parallelism splits the SEQUENCE dimension across CP ranks.
Each rank holds (B, S/CP, H, D) — a slice of the full sequence.

Two strategies (OLMo-core supports both):

1. Ring Attention with zig-zag load balancing:
    - Each rank holds a contiguous chunk of Q
    - KV chunks arrive from neighbors via P2P in a ring
    - Causal masking is applied based on relative chunk positions
    - Zig-zag ordering improves load balance with causal masking
    - In production, calls ring_flash_attn.zigzag_ring_flash_attn_func

2. Ulysses (All-to-All):
    - Convert from sequence-partitioned to head-partitioned via all-to-all
    - Standard flash attention on full sequence with fewer heads
    - Convert back via all-to-all
    - Communication: 2 all-to-alls per attention layer

Causal masking for Ring Attention:
    With CP=4 and ranks [R0, R1, R2, R3], sequence chunks [C0, C1, C2, C3]:
    - R0 has Q positions [0, S/4)
    - When R0 receives KV from R1 (chunk C1 positions [S/4, 2S/4)):
      ALL KV positions are in the FUTURE of R0's Q → output is zero
    - When R0 receives KV from R0 (chunk C0 positions [0, S/4)):
      ALL KV positions are in the PAST → full attention, no masking
    - When R1 receives KV from R0 (chunk C0):
      ALL KV positions are in the PAST → full attention

Zig-zag load balancing (OLMo-core pattern):
    Instead of chunks [0, 1, 2, 3], use interleaved order [0, 3, 1, 2].
    This ensures each rank sees a mix of "mostly unmasked" and "fully masked"
    chunks, rather than some ranks doing all the work and others idle.

Reference: Liu et al. 2023 (Ring Attention)
           OLMo-core src/olmo_core/distributed/context_parallel.py
           Ainslie et al. 2023 (Ulysses)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


@dataclass
class ContextParallelConfig:
    """Configuration for context parallelism.

    :param degree: the CP degree.
    """

    degree: int


def _zigzag_order(cp_size: int, cp_rank: int) -> list[int]:
    """Compute zig-zag chunk order for a given rank.

    The zig-zag pattern ensures each rank processes chunks in an order
    that mixes "mostly unmasked" (local/past) and "fully masked" (future)
    chunks, improving load balance.

    For CP=4, rank 0 gets order [0, 3, 1, 2]:
        Step 0: local chunk (all unmasked)
        Step 1: far future chunk (all masked — fast)
        Step 2: near future chunk (partially masked)
        Step 3: near past chunk (partially unmasked)

    This is better than [0, 1, 2, 3] where step 0 is fast but steps 1-3
    get progressively slower as more attention is computed.

    :param cp_size: total CP degree.
    :param cp_rank: this rank's index.
    :returns: list of chunk indices in zig-zag order.
    """
    order = []
    # First half: alternating from start and end
    for i in range(cp_size):
        if i % 2 == 0:
            order.append(i // 2)
        else:
            order.append(cp_size - 1 - i // 2)
    return order


def _compute_causal_mask(
    q_chunk_idx: int,
    kv_chunk_idx: int,
    cp_size: int,
    S_local: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Compute causal mask for a (Q_chunk, KV_chunk) pair.

    With zig-zag ordering, the relative position between chunks determines
    whether masking is needed:

    If kv_chunk_idx > q_chunk_idx: all KV positions are in the future →
        return a mask of -inf (fully masked).

    If kv_chunk_idx < q_chunk_idx: all KV positions are in the past →
        return None (no masking needed, full attention).

    If kv_chunk_idx == q_chunk_idx: local chunk → apply standard causal mask.

    :param q_chunk_idx: absolute index of the Q chunk in the sequence.
    :param kv_chunk_idx: absolute index of the KV chunk.
    :param cp_size: total CP degree.
    :param S_local: sequence length per chunk.
    :param dtype: output dtype.
    :param device: output device.
    :returns: (S_local, S_local) mask tensor or None.
    """
    if kv_chunk_idx > q_chunk_idx:
        # Future chunk: fully masked
        return torch.full((S_local, S_local), float("-inf"), dtype=dtype, device=device)
    elif kv_chunk_idx < q_chunk_idx:
        # Past chunk: no masking
        return None
    else:
        # Local chunk: standard causal mask
        mask = torch.triu(
            torch.ones(S_local, S_local, dtype=dtype, device=device), diagonal=1
        )
        return mask.masked_fill(mask == 1, float("-inf"))


def _all_to_all_single_cp2hp(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Convert from context-parallel (sequence-partitioned) to head-partitioned.

    Ulysses CP pattern:
        Input:  (B, S/CP, H, D) — sequence-partitioned, full heads
        Output: (B, S, H/CP, D) — full sequence, head-partitioned

    Implementation matches OLMo-core:
        1. Split H into (CP, H_local) and permute CP to dim 0 for all-to-all
        2. All-to-all exchanges head chunks across ranks
        3. Reassemble: CP ranks contribute contiguous sequence chunks

    Reference: OLMo-core src/olmo_core/distributed/context_parallel.py
    """
    cp_size = dist.get_world_size(group)
    B, S_local, H, D = x.shape
    H_local = H // cp_size

    # (B, S_local, H, D) → (B, S_local, CP, H_local, D) → (CP, B, S_local, H_local, D)
    x = x.view(B, S_local, cp_size, H_local, D)
    x = x.permute(2, 0, 1, 3, 4).contiguous()

    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=group)

    # Reassemble: (CP, B, S_local, H_local, D) → (B, CP, S_local, H_local, D)
    # → (B, S, H_local, D)
    # permute(1,0) puts B first, then CP → sequential sequence ordering:
    # [rank0_pos0..posN, rank1_pos0..posN]
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    return output.reshape(B, cp_size * S_local, H_local, D)


def _all_to_all_single_hp2cp(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Convert from head-partitioned back to context-parallel.

    Input:  (B, S, H/CP, D) — full sequence, head-partitioned
    Output: (B, S/CP, H, D) — sequence-partitioned, full heads

    Implementation matches OLMo-core:
        1. Split S into (CP, S_local) and permute CP to dim 0 for all-to-all
        2. All-to-all exchanges sequence chunks across ranks
        3. Reassemble: CP ranks contribute head chunks for each position

    Reference: OLMo-core src/olmo_core/distributed/context_parallel.py
    """
    cp_size = dist.get_world_size(group)
    B, S, H_local, D = x.shape
    S_local = S // cp_size
    H = H_local * cp_size

    # (B, S, H_local, D) → (B, CP, S_local, H_local, D) → (CP, B, S_local, H_local, D)
    x = x.view(B, cp_size, S_local, H_local, D)
    x = x.permute(1, 0, 2, 3, 4).contiguous()

    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=group)

    # Reassemble: (CP, B, S_local, H_local, D) → (B, S_local, CP, H_local, D)
    # → (B, S_local, H, D)
    # permute(1,2,0) puts B first, then S_local, then CP → heads concatenated:
    # [pos0_rank0_h.., pos0_rank1_h.., pos1_rank0_h.., ...]
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    return output.reshape(B, S_local, cp_size * H_local, D)


class RingAttention(nn.Module):
    """Ring Attention for context parallelism.

    Splits the sequence across CP ranks. Each rank computes attention
    on its local Q slice with KV chunks received from neighbors in a ring.

    Causal masking is applied based on the relative positions of Q and KV chunks:
        - KV from future ranks: fully masked (output = 0)
        - KV from past ranks: no masking (full attention)
        - KV from same rank: standard causal mask

    Zig-zag load balancing improves throughput by interleaving chunk order.

    In production, this would call ring_flash_attn.zigzag_ring_flash_attn_func
    which fuses P2P communication with Flash Attention kernels.

    :param group: process group for CP communication.
    :param use_ulysses: use Ulysses (all-to-all) instead of ring P2P.
    :param use_zigzag: use zig-zag chunk ordering for load balancing.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        use_ulysses: bool = False,
        use_zigzag: bool = True,
    ):
        super().__init__()
        self.group = group or dist.group.WORLD
        self.cp_size = dist.get_world_size(self.group)
        self.cp_rank = dist.get_rank(self.group)
        self.use_ulysses = use_ulysses
        self.use_zigzag = use_zigzag

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ring attention.

        :param q: (B, S/CP, H, D) — query, sequence-partitioned.
        :param k: (B, S/CP, H, D) — key, sequence-partitioned.
        :param v: (B, S/CP, H, D) — value, sequence-partitioned.
        :param cu_seqlens: optional cumulative sequence lengths for varlen.
        :returns: (B, S/CP, H, D) — attention output, sequence-partitioned.
        """
        if self.use_ulysses:
            return self._ulysses_forward(q, k, v)
        return self._ring_forward(q, k, v)

    def _ring_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Ring attention with zig-zag ordering and causal masking.

        Each rank starts with its local KV, then receives KV from neighbors
        in a ring pattern. For each KV chunk, compute partial attention with
        appropriate causal masking, then accumulate.
        """
        B, S_local, H, D = q.shape
        output = torch.zeros_like(q)
        scale = D**-0.5

        # Compute zig-zag order for this rank
        if self.use_zigzag:
            chunk_order = _zigzag_order(self.cp_size, self.cp_rank)
        else:
            chunk_order = list(range(self.cp_size))

        # We need to track the actual chunk index for causal masking
        # chunk_order[i] tells us which absolute chunk we process at step i
        # We also need to know which chunk each neighbor sends us
        # At step i, we receive the chunk at index chunk_order[i] from some neighbor

        # Pre-build the chunk assignments per step
        # At step 0: we have our local chunk (chunk_order[0])
        # At step i > 0: we receive chunk chunk_order[i]

        # Current KV we hold
        cur_k = k
        cur_v = v

        for step in range(self.cp_size):
            # Which absolute chunk are we processing this step?
            abs_chunk_idx = chunk_order[step]

            # Compute causal mask for this (Q_chunk, KV_chunk) pair
            mask = _compute_causal_mask(
                q_chunk_idx=self.cp_rank,
                kv_chunk_idx=abs_chunk_idx,
                cp_size=self.cp_size,
                S_local=S_local,
                dtype=q.dtype,
                device=q.device,
            )

            # Compute attention with this KV chunk
            # q: (B, S_local, H, D), k/v: (B, S_local, H, D)
            q_t = q.transpose(1, 2)  # (B, H, S_local, D)
            k_t = cur_k.transpose(1, 2)  # (B, H, S_local, D)
            v_t = cur_v.transpose(1, 2)  # (B, H, S_local, D)

            att = (
                torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
            )  # (B, H, S_local, S_local)

            if mask is not None:
                att = att + mask  # broadcast over (B, H, ...)

            att = torch.softmax(att, dim=-1)
            partial = torch.matmul(att, v_t).transpose(1, 2)  # (B, S_local, H, D)

            output = output + partial

            # Ring communication: send current KV to next rank, receive from prev
            if step < self.cp_size - 1:
                src = (self.cp_rank - 1) % self.cp_size
                dst = (self.cp_rank + 1) % self.cp_size

                new_k = torch.empty_like(k)
                new_v = torch.empty_like(v)
                dist.send(cur_k, dst=dst)
                dist.recv(new_k, src=src)
                dist.send(cur_v, dst=dst)
                dist.recv(new_v, src=src)
                cur_k, cur_v = new_k, new_v

        return output

    def _ulysses_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Ulysses CP: all-to-all to convert sequence→head partitioning.

        Step 1: all-to-all to convert (B, S/CP, H, D) -> (B, S, H/CP, D)
        Step 2: standard attention with full sequence, fewer heads
        Step 3: all-to-all to convert back (B, S, H/CP, D) -> (B, S/CP, H, D)

        Communication volume: 2 * all-to-all of (B * S/CP * H * D)
        """
        B, S_local, H, D = q.shape

        # Step 1: sequence-partitioned -> head-partitioned
        q_hp = _all_to_all_single_cp2hp(q, self.group)
        k_hp = _all_to_all_single_cp2hp(k, self.group)
        v_hp = _all_to_all_single_cp2hp(v, self.group)

        # Step 2: standard causal attention on full sequence, fewer heads
        # In production: flash_attn_func(q_hp, k_hp, v_hp, causal=True)
        att = torch.matmul(
            q_hp.transpose(1, 2), k_hp.transpose(1, 2).transpose(-2, -1)
        ) * (D**-0.5)

        S = S_local * self.cp_size
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, dtype=q.dtype, device=q.device), diagonal=1
        ).masked_fill_(
            torch.triu(torch.ones(S, S, dtype=q.dtype, device=q.device), diagonal=1)
            == 1,
            float("-inf"),
        )
        att = att + causal_mask
        att = torch.softmax(att, dim=-1)
        out_hp = torch.matmul(att, v_hp.transpose(1, 2)).transpose(1, 2)

        # Step 3: head-partitioned -> sequence-partitioned
        return _all_to_all_single_hp2cp(out_hp, self.group)
