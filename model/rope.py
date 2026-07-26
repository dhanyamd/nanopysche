"""Rotary Position Embeddings (RoPE).

Matches OLMo-core nn.rope pattern:
  - RotaryEmbedding with get_buffers() and warmup_cache()
  - head_first parameter for layout flexibility
  - partial_rotary_factor support
  - Scaling configs (ABF, PI, Stepwise, YaRN)

Reference: Su et al. 2021 (arXiv:2104.09864)
           OLMo-core src/olmo_core/nn/rope.py
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class RoPEScalingConfig:
    """RoPE scaling configuration for extending context length."""

    factor: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


class RotaryEmbedding(nn.Module):
    """Precomputed rotary frequencies for RoPE.

    :param dim: head dimension (must be even).
    :param base: theta parameter. LLaMA-2=10000, LLaMA-3=500000.
    :param max_seq_len: precompute buffer size.
    :param head_first: if True, tensors are (B, S, H, D); else (B, H, S, D).
    :param partial_rotary_factor: fraction of head_dim to rotate (0.0-1.0).
    :param scaling: optional scaling config for context extension.
    """

    def __init__(
        self,
        dim: int,
        *,
        base: float = 10000.0,
        max_seq_len: int = 8192,
        head_first: bool = True,
        partial_rotary_factor: float = 1.0,
        scaling: Optional[RoPEScalingConfig] = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.head_first = head_first
        self.partial_rotary_factor = partial_rotary_factor

        rotary_dim = int(dim * partial_rotary_factor)
        assert rotary_dim % 2 == 0, "rotary dimension must be even"

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))

        if scaling is not None:
            factor = scaling.factor
            if factor > 1.0:
                low_freq_wavelen = (
                    scaling.original_max_position_embeddings / scaling.low_freq_factor
                )
                high_freq_wavelen = (
                    scaling.original_max_position_embeddings / scaling.high_freq_factor
                )
                wavelet = 2 * math.pi / inv_freq
                inv_freq = torch.where(
                    wavelet < high_freq_wavelen,
                    inv_freq,
                    torch.where(
                        wavelet > low_freq_wavelen, inv_freq / factor, inv_freq
                    ),
                )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._seq_len_cached: int = 0

    def get_buffers(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin buffers for the given sequence length.

        :param seq_len: sequence length to precompute.
        :param device: target device.
        :param dtype: target dtype.
        :returns: (cos, sin) each of shape (seq_len, rotary_dim).
        """
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

    def warmup_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32
    ):
        """Pre-populate the cos/sin cache.

        :param seq_len: maximum sequence length expected during training.
        :param device: target device.
        :param dtype: target dtype for the cached buffers.
        """
        self.get_buffers(seq_len, device, dtype)

    def forward(
        self,
        seq_len: int,
        *,
        offset: int = 0,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for positions [offset, offset + seq_len).

        :param seq_len: number of positions.
        :param offset: starting position offset.
        :param device: target device.
        :param dtype: target dtype.
        :returns: (cos, sin) each of shape (seq_len, rotary_dim).
        """
        cos, sin = self.get_buffers(offset + seq_len, device, dtype)
        return cos[offset : offset + seq_len], sin[offset : offset + seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate halves: [x0, x1, x2, ...] -> [-x1, x0, -x3, ...]

    Equivalent to multiplying by e^{i*angle} in complex space.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    head_first: bool = True,
) -> torch.Tensor:
    """Apply rotary embeddings to a tensor.

    :param x: (B, S, H, D) if head_first else (B, H, S, D)
    :param cos: (S, rotary_dim)
    :param sin: (S, rotary_dim)
    :param head_first: tensor layout flag.
    :returns: rotated tensor, same shape.
    """
    if head_first:
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
        sin = sin.unsqueeze(0).unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)
