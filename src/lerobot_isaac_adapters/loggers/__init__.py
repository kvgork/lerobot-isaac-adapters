"""
loggers — metrics logger adapters

Soft-import discipline: the W&B adapter is optional. If `wandb` is not
installed, importing this package still succeeds, but `WandbLogger` raises
ImportError when instantiated.
"""

from .base import LoggerBase
from .stdout import StdoutLogger

__all__ = ["LoggerBase", "StdoutLogger", "WandbLogger"]


def __getattr__(name):
    # Lazy import for the optional W&B logger so the import chain doesn't
    # crash on systems without the SDK.
    if name == "WandbLogger":
        from .wandb_logger import WandbLogger as _WL
        return _WL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
