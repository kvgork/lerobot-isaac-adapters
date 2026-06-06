"""
StdoutLogger — minimal logger that prints to stdout. Always available.

Used as default when no other logger is selected, and as fallback if W&B
or other backends fail to initialize.
"""

from __future__ import annotations

import json
from pathlib import Path


class StdoutLogger:
    name = "stdout"

    def __init__(self, run_name: str | None = None):
        self.run_name = run_name or "stdout-run"
        print(f"[stdout-logger] start run={self.run_name}")

    def log(self, step: int, metrics: dict[str, float]) -> None:
        print(f"[stdout-logger] step={step} {json.dumps(metrics)}")

    def log_image(self, step: int, key: str, image) -> None:
        # Shape only; do not actually serialize images.
        shape = getattr(image, "shape", "unknown")
        print(f"[stdout-logger] step={step} image={key} shape={shape}")

    def log_artifact(self, step: int, name: str, path: Path) -> None:
        print(f"[stdout-logger] step={step} artifact={name} path={path}")

    def finish(self) -> None:
        print(f"[stdout-logger] end run={self.run_name}")
