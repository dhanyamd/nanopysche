"""nanopsyche.nn — Neural network utilities.

TP wrapper selection (standard vs Float8-aware).
"""

from nanopsyche.nn.utils import get_tp_wrappers

__all__ = ["get_tp_wrappers"]
