"""nanopsyche.model — Production-grade transformer building blocks.

RMSNorm, RoPE, Attention, FeedForward, MoE, Transformer — each in its own file.
Modeled after OLMo-core patterns, using PyTorch distributed primitives.

Reference: OLMo-core src/olmo_core/nn/
"""

from nanopsyche.model.norm import RMSNorm
from nanopsyche.model.rope import RotaryEmbedding, RoPEScalingConfig, apply_rotary_emb
from nanopsyche.model.attention import Attention
from nanopsyche.model.feed_forward import FeedForward
from nanopsyche.model.moe import MoEBase, MoERouter
from nanopsyche.model.transformer import TransformerBlock, Transformer

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "RoPEScalingConfig",
    "apply_rotary_emb",
    "Attention",
    "FeedForward",
    "MoEBase",
    "MoERouter",
    "TransformerBlock",
    "Transformer",
]
