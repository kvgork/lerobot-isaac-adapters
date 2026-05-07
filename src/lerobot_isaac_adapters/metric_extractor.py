"""
metric_extractor
================

Canonical stdout metric emitter for lerobot-isaac-adapters.

All training targets use this module to emit eval metrics so that
``autoresearch-ml-executor-worker`` can parse them via the regex::

    (\\w+)[=:\\s]+([0-9.eE+-]+)

Usage
-----
>>> from lerobot_isaac_adapters.metric_extractor import emit, MetricEmitter
>>> emit("pc_success", 0.73)
pc_success=0.73
>>> emit("recon_loss", 3.17e-2, step=1000)
recon_loss=3.17e-02  # step=1000

Context manager (flushes on exit)::

    with MetricEmitter(prefix="eval") as me:
        me.emit("pred_loss", 0.021)

Notes
-----
- Always prints to ``sys.stdout`` so the value is captured by subprocess pipes.
- ``step`` is appended as a comment after ``#`` to preserve parse-compatibility.
- Format: ``<name>=<value>`` where ``value`` uses ``:.6g`` (6 significant digits)
  which keeps output stable across Python versions and avoids repr()-dependent
  surprises (e.g. ``0.73`` vs ``0.7300000000000001``).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Generator, Optional


def emit(name: str, value: float, step: Optional[int] = None) -> None:
    """Print a single metric line to stdout in the autoresearch-compatible format.

    The output format is::

        <name>=<value>

    If ``step`` is provided a comment is appended::

        <name>=<value>  # step=<step>

    Parameters
    ----------
    name:
        Metric name — must match ``\\w+`` (letters, digits, underscores).
    value:
        Numeric metric value (float).
    step:
        Optional training step counter; appended as a comment so the regex
        parser can ignore it and focus on the ``name=value`` portion.

    Raises
    ------
    ValueError
        If ``name`` contains characters outside ``[A-Za-z0-9_]``.
    """
    if not name.replace("_", "").isalnum():
        raise ValueError(
            f"Metric name {name!r} is not regex-safe. "
            "Use only letters, digits, and underscores."
        )

    line = f"{name}={value:.6g}"
    if step is not None:
        line = f"{line}  # step={step}"
    print(line, flush=True)


class MetricEmitter:
    """Context manager that collects metrics and flushes ``sys.stdout`` on exit.

    Useful when a training loop buffers output and you want a guaranteed flush
    at the end of each eval step.

    Parameters
    ----------
    prefix:
        Optional string prepended to metric names with ``_`` separator.

    Example
    -------
    >>> with MetricEmitter(prefix="eval") as me:
    ...     me.emit("pred_loss", 0.021, step=500)
    eval_pred_loss=0.021  # step=500
    """

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix

    def emit(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Emit a metric, optionally prepending the configured prefix."""
        full_name = f"{self._prefix}_{name}" if self._prefix else name
        emit(full_name, value, step=step)

    def __enter__(self) -> "MetricEmitter":
        return self

    def __exit__(self, *_exc) -> None:
        sys.stdout.flush()


@contextmanager
def metric_scope(prefix: str = "") -> Generator[MetricEmitter, None, None]:
    """Convenience alias for ``MetricEmitter`` as a ``contextmanager``.

    >>> with metric_scope("train") as m:
    ...     m.emit("loss", 0.5)
    train_loss=0.5
    """
    emitter = MetricEmitter(prefix=prefix)
    with emitter:
        yield emitter
