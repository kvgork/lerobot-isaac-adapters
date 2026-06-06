"""
LoggerBase — common protocol for training metrics sinks.

A logger adapter is a small object with these methods:

- ``log(step: int, metrics: dict[str, float])`` — record a step
- ``log_image(step: int, key: str, image: np.ndarray)`` — record an image
- ``log_artifact(step: int, name: str, path: Path)`` — record a file artefact
- ``finish()`` — flush + close

Adapters MUST NOT raise on missing SDKs; they should soft-import in __init__
and degrade to a no-op if the backend is unavailable. The default ``StdoutLogger``
in ``stdout.py`` is always available and serves as the fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

try:
    import numpy as np  # type: ignore
    NDArray = "np.ndarray"
except ImportError:  # pragma: no cover - numpy is a hard dep in practice
    NDArray = "object"


@runtime_checkable
class LoggerBase(Protocol):
    """Protocol for training metric loggers."""

    name: str

    def log(self, step: int, metrics: dict[str, float]) -> None: ...

    def log_image(self, step: int, key: str, image) -> None: ...

    def log_artifact(self, step: int, name: str, path: Path) -> None: ...

    def finish(self) -> None: ...
