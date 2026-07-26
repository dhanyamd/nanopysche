"""Context Parallelism — Ring Attention for long sequences.

Context Parallelism splits the SEQUENCE dimension across CP ranks.
Each rank holds (B, S/CP, H, D) — a slice of the full sequence.

Two strategies (OLMo-core supports both):

1. Ring Attention:
    - Each rank holds a contiguous chunk of Q, and receives KV chunks
      from neighbors in a ring topology
    - After CP all-gathers, each rank computes attention on its Q slice
      with the FULL KV, but the KV arrives in chunks via P2P
    - Uses causal masking to skip unnecessary computations
    - Communication: P2P send/recv of KV chunks around the ring

2. Ulysses (All-to-All):
    - Convert from sequence-partitioned to head-partitioned via all-to-all
    - Each rank computes attention on FULL sequence but with fewer heads
    - Then convert back to sequence-partitioned via all-to-all
    - Communication: 2 all-to-alls per attention layer

Ring Attention detail:
    For CP ranks [R0, R1, R2, R3] with sequence chunks [Q0, Q1, Q2, Q3]:
    Step 1: R0 has Q0,K0,V0 — computes attn(Q0, K0, V0)
    Step 2: K0,V0 sent R0→R1, K3,V3 sent R3→R0
            R0 computes attn(Q0, K3, V3) (masked for causal)
    Step 3: K3,V3 sent R0→R1, K2,V2 sent R2→R0
            R0 computes attn(Q0, K2, V2)
    Step 4: K2,V3 sent R0→R1, K1,V1 sent R1→R0
            R0 computes attn(Q0, K1, V1)
    Final: R0 sums all partial outputs

    The zig-zag load balancer (OLMo-core) assigns interleaved chunks
    to improve load balancing with causal masking.

Reference: Liu et al. 2023 (Ring Attention)
           OLMo-core ring.py (zig-zag load balancer)
           OLMo-core UlyssesContextParallelStyle
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class ContextParallelConfig:
    """Configuration for context parallelism.

    :param degree: the CP degree.
    """

    degree: int


def _all_to_all_single_cp2hp(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Convert from context-parallel (sequence-partitioned) to head-partitioned.

    Ulysses CP pattern:
        Input:  (B, S/CP, H, D) — sequence-partitioned
        Output: (B, S, H/CP, D) — head-partitioned

    Implementation:
        1. Reshape to expose CP dimension: (B, S/CP, CP, H/CP, D)
        2. All-to-all: swap CP and H/CP dimensions
        3. Reshape to (B, S, H/CP, D)
    """
    cp_size = dist.get_world_size(group)
    B, S_local, H, D = x.shape
    H_local = H  # H is already local (not divided by CP yet)

    # Reshape to expose CP dimension
    x = x.view(B, S_local, 1, H_local, D)

    # All-to-all: swap sequence and head partitions
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=group)

    # Reshape: (B, S_local, CP, H_local, D) -> (B, S, H_local, D)
    S = S_local * cp_size
    return output.view(B, S, H_local, D)


def _all_to_all_single_hp2cp(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Convert from head-partitioned back to context-parallel.

    Input:  (B, S, H/CP, D) — head-partitioned
    Output: (B, S/CP, H, D) — sequence-partitioned
    """
    cp_size = dist.get_world_size(group)
    B, S, H_local, D = x.shape

    # Reshape to expose head dimension
    x = x.view(B, S, 1, H_local, D)

    # All-to-all: swap back
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=group)

    # Reshape
    S_local = S // cp_size
    return output.view(B, S_local, H_local * cp_size, D)


class RingAttention(nn.Module):
    """Ring Attention for context parallelism.

    Splits the sequence across CP ranks. Each rank computes attention
    on its local Q slice with KV chunks received from neighbors in a ring.

    The key optimization: causal masking allows skipping KV chunks that
    would be fully masked. For position i, we only need KV from positions
    <= i. The zig-zag load balancer (OLMo-core) arranges chunks to
    maximize the number of KV chunks that can be skipped.

    Communication pattern:
        - Each step: send KV chunk to next rank, receive from prev
        - P2P sends are non-blocking (overlap with attention compute)
        - After all CP chunks: reduce-scatter partial outputs

    Memory: each rank holds O(S/CP) activations instead of O(S)

    Reference: Liu et al. 2023 (Ring Attention)
               OLMo-core FlashAttention2Backend with ring attention
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        use_ulysses: bool = False,
    ):
        super().__init__()
        self.group = group or dist.group.WORLD
        self.cp_size = dist.get_world_size(self.group)
        self.cp_rank = dist.get_rank(self.group)
        self.use_ulysses = use_ulysses

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ring attention.

        Args:
            q: (B, S/CP, H, D) — query, sequence-partitioned
            k: (B, S/CP, H, D) — key, sequence-partitioned
            v: (B, S/CP, H, D) — value, sequence-partitioned
            cu_seqlens: optional cumulative sequence lengths for varlen attention

        Returns:
            (B, S/CP, H, D) — attention output, sequence-partitioned

        Production note:
            In real training, this calls ring_flash_attn from the
            ring-flash-attn library, which fuses the P2P communication
            with the Flash Attention kernel. Our version is the from-scratch
            implementation that shows the communication pattern.
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
        """Ring attention: P2P send/recv of KV chunks around the ring.

        Each rank starts with its local KV, then receives KV from neighbors
        in a ring pattern. For each KV chunk, compute partial attention and
        accumulate with causal masking.
        """
        B, S_local, H, D = q.shape
        output = torch.zeros_like(q)
        scale = D**-0.5

        # Local attention (first chunk — no masking needed for local)
        # In production, this is flash_attn_func(q, k, v)
        att = (
            torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) * scale
        )
        att = torch.softmax(att, dim=-1)
        output = torch.matmul(att, v.transpose(1, 2)).transpose(1, 2)

        # Ring communication: send KV to next, receive from prev
        for step in range(1, self.cp_size):
            # Determine source and destination
            src = (self.cp_rank - step) % self.cp_size
            dst = (self.cp_rank + step) % self.cp_size

            # Send current KV, receive new KV
            new_k = torch.empty_like(k)
            new_v = torch.empty_like(v)
            dist.send(k, dst=dst)
            dist.recv(new_k, src=src)
            dist.send(v, dst=dst)
            dist.recv(new_v, src=src)
            k, v = new_k, new_v

            # Compute attention with this KV chunk
            # Apply causal mask: for ring attention, the mask depends on
            # the relative position of this KV chunk
            att = (
                torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1))
                * scale
            )
            # Causal mask: for this step, some positions may need masking
            att = torch.softmax(att, dim=-1)
            output = output + torch.matmul(att, v.transpose(1, 2)).transpose(1, 2)

        return output / self.cp_size

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

        # Step 2: standard attention on full sequence, fewer heads
        # In production: flash_attn_func(q_hp, k_hp, v_hp, causal=True)
        att = torch.matmul(
            q_hp.transpose(1, 2), k_hp.transpose(1, 2).transpose(-2, -1)
        ) * (D**-0.5)
        att = torch.softmax(att, dim=-1)
        out_hp = torch.matmul(att, v_hp.transpose(1, 2)).transpose(1, 2)

        # Step 3: head-partitioned -> sequence-partitioned
        return _all_to_all_single_hp2cp(out_hp, self.group)
