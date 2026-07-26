"""NN utilities — TP wrapper selection.

Matches OLMo-core nn.utils pattern:
  - get_tp_wrappers() selects between standard and Float8-aware parallel styles
  - Returns (RowwiseParallel, ColwiseParallel, PrepareModuleInput)

Reference: OLMo-core src/olmo_core/nn/utils.py
"""

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)


def get_tp_wrappers(float8_enabled: bool = False):
    """Get the appropriate TP wrapper classes.

    :param float8_enabled: if True, use Float8-aware parallel styles.
    :returns: (rowwise_parallel, colwise_parallel, prepare_module_input)
    """
    if not float8_enabled:
        return RowwiseParallel, ColwiseParallel, PrepareModuleInput
    else:
        try:
            from torchao.float8.float8_tensor_parallel import (
                Float8ColwiseParallel,
                Float8RowwiseParallel,
                PrepareFloat8ModuleInput,
            )

            return (
                Float8RowwiseParallel,
                Float8ColwiseParallel,
                PrepareFloat8ModuleInput,
            )
        except ImportError:
            return RowwiseParallel, ColwiseParallel, PrepareModuleInput
