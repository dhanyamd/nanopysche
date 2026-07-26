"""FSDP — uses PyTorch native fully_shard (FSDP2).

FSDP shards model parameters, gradients, and optimizer states across
data-parallel ranks. During forward/backward, parameters are temporarily
gathered (all-gather), used, then resharded.

OLMo-core uses PyTorch FSDP2 (DTensor-based):
    fully_shard(module, mesh=dp_mesh, mp_policy=MixedPrecisionPolicy(...))

The actual FSDP application happens inside each module's apply_fsdp() method.
This file provides wrapping strategy selection and activation checkpointing.

Reference: Zhao et al. 2023 (PyTorch FSDP)
           OLMo-core src/olmo_core/nn/transformer/block.py
           Korthikanti et al. 2022 (activation checkpointing)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

log = logging.getLogger(__name__)


class DataParallelType(Enum):
    """Data parallelism type."""

    fsdp = "fsdp"
    hsdp = "hsdp"
    ddp = "ddp"


class TransformerDataParallelWrappingStrategy(Enum):
    """FSDP wrapping strategies for transformer blocks."""

    full = "full"
    fine_grained = "fine_grained"


@dataclass
class DataParallelConfig:
    """Configuration for data parallelism.

    :param name: data parallel type.
    :param param_dtype: parameter dtype for mixed precision.
    :param reduce_dtype: gradient reduction dtype.
    """

    name: DataParallelType = DataParallelType.fsdp
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: torch.dtype = torch.float32


class ActivationCheckpointing(nn.Module):
    """Activation checkpointing wrapper.

    Reference: Chen et al. 2016
               PyTorch torch.utils.checkpoint
    """

    def __init__(self, module: nn.Module, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        return checkpoint(
            self.module,
            *args,
            use_reentrant=self.use_reentrant,
            **kwargs,
        )


def apply_activation_checkpointing(
    model: nn.Module,
    mode: str = "full",
    block_interval: Optional[int] = None,
) -> nn.Module:
    """Apply activation checkpointing to a Transformer model.

    :param model: the transformer model (must have .blocks as ModuleDict or ModuleList).
    :param mode: "full" (every block), "selected_blocks" (every N-th).
    :param block_interval: for "selected_blocks" mode.
    """
    if mode == "full":
        for name, block in model.named_children():
            if name == "blocks":
                for i, (blk_name, blk) in enumerate(block.named_children()):
                    block[blk_name] = ActivationCheckpointing(blk)
    elif mode == "selected_blocks" and block_interval is not None:
        for name, child in model.named_children():
            if name == "blocks":
                for i, (blk_name, blk) in enumerate(child.named_children()):
                    if i % block_interval == 0:
                        child[blk_name] = ActivationCheckpointing(blk)
    return model
