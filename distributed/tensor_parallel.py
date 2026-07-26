"""Tensor Parallelism — uses PyTorch native parallelize_module.

TP is NOT implemented with custom autograd functions. Instead we use:
  - ColwiseParallel: splits weight output dim, identity fwd, all-reduce bwd
  - RowwiseParallel: splits weight input dim, all-reduce fwd, identity bwd
  - SequenceParallel: all-reduce around norms/dropout between TP regions

The TP patterns are applied inside each module's apply_tp() method.

Reference: Shoeybi et al. 2019 (Megatron-LM)
           Korthikanti et al. 2022 (Reducing Activation Recomputation)
           OLMo-core src/olmo_core/distributed/parallel/tensor_parallel.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement, Shard, distribute_module
from torch.distributed.tensor.parallel import SequenceParallel as _SequenceParallel

log = logging.getLogger(__name__)


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism.

    :param degree: the TP degree.
    :param enable_async: enable experimental async tensor parallelism.
    """

    degree: int
    enable_async: bool = False

    def maybe_enable_async_tp(self, tp_mesh: DeviceMesh):
        """Enable async TP if configured."""
        if self.enable_async:
            log.info("Enabling async tensor parallel")
            try:
                from torch.distributed._symmetric_memory import (
                    enable_symm_mem_for_group,
                )

                torch._inductor.config._micro_pipeline_tp = True  # type: ignore
                enable_symm_mem_for_group(tp_mesh.get_group().group_name)
            except (ImportError, AttributeError):
                log.warning("Async TP not available in this PyTorch version")


class SequenceParallel(_SequenceParallel):
    """Custom SequenceParallel with configurable output layouts.

    Reference: OLMo-core src/olmo_core/distributed/parallel/tensor_parallel.py
    """

    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        use_local_output: bool = False,
        output_layouts: Optional[Placement] = None,
    ):
        super().__init__(sequence_dim=sequence_dim, use_local_output=use_local_output)
        self._output_layouts = (output_layouts or Shard(sequence_dim),)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        def _prepare_output(outputs, device_mesh):
            if outputs.placements != self._output_layouts:
                outputs = outputs.redistribute(
                    placements=self._output_layouts, async_op=True
                )
            return outputs.to_local() if self.use_local_output else outputs

        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.sequence_sharding),  # type: ignore
            _prepare_output,
        )
