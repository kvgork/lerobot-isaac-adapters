"""
get_logger — factory + selection by name. Used by train.py CLI:

    args.logger ∈ {"stdout", "wandb"}
    logger = get_logger(args.logger, run_name=args.run_name, config=cfg.to_dict())

Falls back to StdoutLogger if the requested backend cannot be loaded.
"""

from __future__ import annotations

import logging
from typing import Any

from .stdout import StdoutLogger

logger = logging.getLogger(__name__)


def get_logger(name: str, **kwargs: Any):
    """Return a logger adapter by name. Falls back to stdout on failure."""
    name = (name or "stdout").lower()
    if name == "stdout":
        return StdoutLogger(run_name=kwargs.get("run_name"))
    if name == "wandb":
        try:
            from .wandb_logger import WandbLogger

            return WandbLogger(**kwargs)
        except ImportError as e:
            logger.warning("wandb unavailable, falling back to stdout: %s", e)
            return StdoutLogger(run_name=kwargs.get("run_name"))
    raise ValueError(f"Unknown logger backend: {name!r}. Pick one of: stdout, wandb")
