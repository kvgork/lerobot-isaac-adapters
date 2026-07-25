"""quality.py — dataset quality filtering for lerobot-isaac-adapters.

Thin facade over the in-package quality module
``lerobot_isaac_adapters.data.dataset_quality`` (SAL + TED). The SAL/TED math used to live in
the ``lerobot_dataset_quality`` skill, reached from here via a ``CLAUDE_CODE_ROOT`` sys.path
injection + a ``python -c`` subprocess fallback. That code was moved into the package
(2026-06-28), so this is now a direct in-process call — no path bridging, no subprocess.

Plan reference: §13.1 Bundle A, deliverable A2
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# SAL/TED filtering now lives in the package (moved out of the skill). The legacy skill path
# is kept only as a fallback for older checkouts where it hasn't been refactored yet.
try:
    from lerobot_isaac_adapters.data.dataset_quality import (
        OperationResult,
        filter_dataset as _filter_dataset,
    )
except ImportError:  # pragma: no cover
    try:
        from skills.lerobot_dataset_quality.operations import (  # type: ignore[import]
            OperationResult,
            filter_dataset as _filter_dataset,
        )
    except ImportError as exc:
        raise ImportError(
            "Cannot import dataset-quality filtering. Expected "
            "lerobot_isaac_adapters.data.dataset_quality (this package) or the "
            "lerobot_dataset_quality skill on PYTHONPATH."
        ) from exc


def apply_quality_filter(
    dataset_path: str | Path,
    sal_threshold: float = 0.2,
    ted_threshold: float = 2.0,
    min_episode_length: int = 50,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> "OperationResult":
    """Filter a LeRobotDataset using SAL + TED quality metrics.

    Delegates to ``lerobot_isaac_adapters.data.dataset_quality.filter_dataset`` (in-process).

    Parameters
    ----------
    dataset_path:
        Path to a LeRobotDataset root directory.
    sal_threshold:
        Fraction of worst-ranked episodes to remove (0.2 → remove bottom 20%); passed as
        ``filter_percentile=int(sal_threshold * 100)`` with the composite SAL+TED strategy.
    ted_threshold, min_episode_length:
        Retained for API compatibility — the composite ranking already folds in TED, and a
        minimum-length guard is advisory.
    output_path:
        Destination for the filtered dataset (default ``<dataset_path>_filtered``).
    dry_run:
        If True, report what would be filtered without writing any files.

    Returns
    -------
    OperationResult
        ``success=True`` with kept/removed counts in ``data`` on success; ``success=False``
        with ``error`` + ``suggestions`` otherwise.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return OperationResult(
            success=False,
            error=f"Dataset not found: {dataset_path}",
            suggestions=["Check --dataset path.", "Run lerobot-record first."],
        )

    if output_path is None:
        output_path = Path(str(dataset_path) + "_filtered")
    output_path = Path(output_path)

    logger.info(
        "apply_quality_filter: dataset=%s output=%s sal=%.2f ted=%.2f minlen=%d dry_run=%s",
        dataset_path, output_path, sal_threshold, ted_threshold, min_episode_length, dry_run,
    )

    result = _filter_dataset(
        dataset_path=str(dataset_path),
        output_path=str(output_path),
        filter_percentile=int(sal_threshold * 100),
        strategy="composite",
        dry_run=dry_run,
    )
    return OperationResult(
        success=getattr(result, "success", False),
        data=getattr(result, "data", None),
        error=getattr(result, "error", None),
        suggestions=getattr(result, "suggestions", None),
    )
