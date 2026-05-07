"""
lerobot-isaac-adapters
======================

Modular training adapter for the LeRobot + Isaac Lab workspace.

Public API
----------
- ``train``              — entrypoint module (main() callable)
- ``metric_extractor``   — canonical stdout metric emitter

Quick start
-----------
>>> from lerobot_isaac_adapters.metric_extractor import emit
>>> emit("pc_success", 0.73)  # prints: pc_success=0.73
"""

from lerobot_isaac_adapters import metric_extractor  # noqa: F401
from lerobot_isaac_adapters import train  # noqa: F401

__all__ = ["train", "metric_extractor"]
__version__ = "0.1.0"
