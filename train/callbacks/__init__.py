"""Training callbacks — production observability and monitoring.

OLMo-core pattern: callback-driven architecture where the trainer invokes
lifecycle hooks and callbacks record/dispatch metrics.

Reference: OLMo-core src/olmo_core/train/callbacks/
"""

from nanopsyche.train.callbacks.base import Callback
from nanopsyche.train.callbacks.speed_monitor import SpeedMonitorCallback
from nanopsyche.train.callbacks.gpu_memory_monitor import GPUMemoryMonitorCallback
from nanopsyche.train.callbacks.wandb_logger import WandBCallback
from nanopsyche.train.callbacks.console_logger import ConsoleLoggerCallback
from nanopsyche.train.callbacks.profiler import ProfilerCallback
from nanopsyche.train.callbacks.stability_monitor import StabilityMonitorCallback

__all__ = [
    "Callback",
    "SpeedMonitorCallback",
    "GPUMemoryMonitorCallback",
    "WandBCallback",
    "ConsoleLoggerCallback",
    "ProfilerCallback",
    "StabilityMonitorCallback",
]
