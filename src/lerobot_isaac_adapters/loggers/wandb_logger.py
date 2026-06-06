"""
WandbLogger — Weights & Biases backend (Bundle E scaffold, 2026-05-21).

Soft-imports wandb. If wandb is not installed, constructor raises ImportError
with the install hint. Configuration via constructor args + env vars:

- WANDB_PROJECT (default: ``lerobot-isaac-training``)
- WANDB_ENTITY (default: unset; uses W&B user default)
- WANDB_RUN_GROUP (default: unset)
- WANDB_MODE (default: ``online``; set to ``disabled`` for CI / unit tests)
- WANDB_ANONYMOUS (default: unset)

The adapter intentionally exposes only the minimal LoggerBase surface +
``log_table`` for the eval-lake compatibility used by `lerobot-isaac-dashboard`.

Status: scaffold only. Real training run + dashboard validation pending.
See ``01-Projects/lerobot-isaac-deferred-bundles-plan.md`` §"Bundle E".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WandbLogger:
    name = "wandb"

    def __init__(
        self,
        run_name: str | None = None,
        project: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
    ):
        try:
            import wandb  # type: ignore
        except ImportError as e:
            raise ImportError(
                "wandb is required for WandbLogger but is not installed.\n"
                "Install via: pip install wandb>=0.16\n"
                "Or fall back to StdoutLogger for offline runs."
            ) from e

        self._wandb = wandb
        self.run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", "lerobot-isaac-training"),
            entity=entity or os.environ.get("WANDB_ENTITY"),
            group=group or os.environ.get("WANDB_RUN_GROUP"),
            name=run_name,
            config=config or {},
            tags=tags or [],
            mode=mode or os.environ.get("WANDB_MODE", "online"),
            reinit=True,
        )
        logger.info("W&B run started: %s/%s", self.run.entity, self.run.id)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        self._wandb.log(metrics, step=step)

    def log_image(self, step: int, key: str, image) -> None:
        self._wandb.log({key: self._wandb.Image(image)}, step=step)

    def log_artifact(self, step: int, name: str, path: Path) -> None:
        artifact = self._wandb.Artifact(name=name, type="model")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def log_table(self, step: int, name: str, columns: list[str], rows: list[list[Any]]) -> None:
        table = self._wandb.Table(columns=columns, data=rows)
        self._wandb.log({name: table}, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            logger.info("W&B run finished")
