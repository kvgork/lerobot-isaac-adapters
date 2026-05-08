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

# NOTE: ``train`` is intentionally NOT imported eagerly here.
# Importing it would shadow ``python -m lerobot_isaac_adapters.train`` and
# trigger ``RuntimeWarning: 'lerobot_isaac_adapters.train' found in sys.modules``
# whenever the module is invoked as a script. Callers that need it should
# do ``from lerobot_isaac_adapters import train`` themselves.

__all__ = ["metric_extractor"]
__version__ = "0.1.0"
