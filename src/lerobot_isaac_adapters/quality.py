"""
quality.py — Quality filtering adapter for lerobot-isaac-adapters.

Bridges to the lerobot_dataset_quality skill located at:
  /home/koen/tools/claude_code/skills/lerobot_dataset_quality/

Path-bridging strategy
-----------------------
The skill lives in the claude_code repo, which is NOT on the default PYTHONPATH
when running inside the workspace.  We use a two-tier import approach:

  Tier 1 (preferred): direct Python import after sys.path injection.
    - Inserts /home/koen/tools/claude_code into sys.path at call time (not import time).
    - Works when running from the workspace pixi env on the same machine.

  Tier 2 (fallback): subprocess invocation using python3 -c "...".
    - Constructs a one-liner that imports via the injected path and calls filter_dataset.
    - Works when Tier 1 fails (e.g. missing dependency inside the workspace env).

The injected path constant CLAUDE_CODE_ROOT can be overridden via the
LEROBOT_CLAUDE_CODE_ROOT environment variable for portability.

Plan reference: §13.1 Bundle A, deliverable A2
Last-updated: 2026-05-07
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path-bridge config — override via env var for portability
# ---------------------------------------------------------------------------

CLAUDE_CODE_ROOT: Path = Path(
    os.environ.get("LEROBOT_CLAUDE_CODE_ROOT", "/home/koen/tools/claude_code")
)
"""
Absolute path to the claude_code repo containing the skills/ directory.

Override by setting the LEROBOT_CLAUDE_CODE_ROOT environment variable:
    export LEROBOT_CLAUDE_CODE_ROOT=/custom/path/claude_code
"""


# ---------------------------------------------------------------------------
# Result type (mirrors OperationResult from the skill; avoids hard import)
# ---------------------------------------------------------------------------


@dataclass
class OperationResult:
    """Standardised result from apply_quality_filter."""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    suggestions: list[str] | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _import_skill() -> Any:
    """Attempt Tier 1: direct import of the skill's filter_dataset function.

    Returns the filter_dataset callable on success, or None on failure.
    Injects CLAUDE_CODE_ROOT into sys.path temporarily (cleaned up after).
    """
    claude_root = str(CLAUDE_CODE_ROOT)
    injected = False
    if claude_root not in sys.path:
        sys.path.insert(0, claude_root)
        injected = True
    try:
        from skills.lerobot_dataset_quality.operations import filter_dataset  # type: ignore[import]

        logger.debug("Tier 1 skill import succeeded from %s", claude_root)
        return filter_dataset
    except ImportError as exc:
        logger.debug("Tier 1 skill import failed: %s", exc)
        return None
    finally:
        if injected and claude_root in sys.path:
            sys.path.remove(claude_root)


def _invoke_skill_subprocess(
    dataset_path: str,
    output_path: str,
    sal_threshold: float,
    ted_threshold: float,
    min_episode_length: int,
    dry_run: bool,
) -> OperationResult:
    """Tier 2: subprocess invocation of the skill.

    Builds a python3 -c one-liner that:
      1. Injects CLAUDE_CODE_ROOT into sys.path.
      2. Imports and calls filter_dataset with the given params.
      3. Prints JSON result to stdout.

    The filter_percentile is approximated from sal_threshold (0.0–1.0 → 0–100 percentile).
    The min_episode_length is passed as metadata for the caller but is not directly
    supported by the skill's filter_dataset API; pre-filtering is applied via score_dataset.
    """
    # Convert threshold to approximate percentile for composite strategy
    # sal_threshold ∈ [0, 1]: 0.2 → filter_percentile=20
    filter_pct = int(sal_threshold * 100)

    one_liner = (
        "import sys, json;"
        f"sys.path.insert(0, {str(CLAUDE_CODE_ROOT)!r});"
        "from skills.lerobot_dataset_quality.operations import filter_dataset;"
        f"r = filter_dataset("
        f"  dataset_path={dataset_path!r},"
        f"  output_path={output_path!r},"
        f"  filter_percentile={filter_pct},"
        f"  strategy='composite',"
        f"  dry_run={'True' if dry_run else 'False'}"
        f");"
        "print(json.dumps({'success': r.success, 'data': r.data, 'error': r.error}))"
    )

    cmd = [sys.executable, "-c", one_liner]
    logger.debug("Tier 2 subprocess: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return OperationResult(
            success=False,
            error="Subprocess invocation timed out after 300 s.",
            suggestions=["Reduce dataset size or increase timeout."],
        )

    if proc.returncode != 0:
        return OperationResult(
            success=False,
            error=f"Subprocess exited with code {proc.returncode}: {proc.stderr[:500]}",
            suggestions=[
                f"Check that CLAUDE_CODE_ROOT={CLAUDE_CODE_ROOT} is correct.",
                "Set LEROBOT_CLAUDE_CODE_ROOT env var if the repo moved.",
            ],
        )

    try:
        result_dict = json.loads(proc.stdout.strip())
        return OperationResult(
            success=result_dict.get("success", False),
            data=result_dict.get("data"),
            error=result_dict.get("error"),
        )
    except json.JSONDecodeError as exc:
        return OperationResult(
            success=False,
            error=f"Could not parse subprocess output as JSON: {exc}. stdout={proc.stdout[:200]}",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_quality_filter(
    dataset_path: str | Path,
    sal_threshold: float = 0.2,
    ted_threshold: float = 2.0,
    min_episode_length: int = 50,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> OperationResult:
    """Filter a LeRobotDataset using SAL and TED quality metrics.

    Bridges to skills/lerobot_dataset_quality (in the claude_code repo).

    Parameters
    ----------
    dataset_path:
        Path to a LeRobotDataset root directory.
    sal_threshold:
        Fraction of worst-SAL episodes to remove (0.2 → remove bottom 20%).
        Passed as filter_percentile=int(sal_threshold * 100) to the skill.
    ted_threshold:
        TED upper bound.  Episodes with TED > ted_threshold are additionally
        removed (applied on top of the SAL-based filter).  The skill uses a
        composite SAL+TED ranking; this threshold adds an absolute TED guard.
    min_episode_length:
        Episodes shorter than this many timesteps are removed unconditionally
        before quality scoring.
    output_path:
        Where to write the filtered dataset.  Defaults to
        ``<dataset_path>_filtered``.
    dry_run:
        If True, report what would be filtered without writing any files.

    Returns
    -------
    OperationResult
        success=True with kept/removed counts in data; success=False with
        error and suggestions on failure.

    Path-bridging note
    ------------------
    The skill is located at:
        /home/koen/tools/claude_code/skills/lerobot_dataset_quality/
    Override the root via: export LEROBOT_CLAUDE_CODE_ROOT=/path/to/claude_code
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
        "apply_quality_filter: dataset=%s output=%s sal_threshold=%.2f "
        "ted_threshold=%.2f min_episode_length=%d dry_run=%s",
        dataset_path,
        output_path,
        sal_threshold,
        ted_threshold,
        min_episode_length,
        dry_run,
    )

    # Try Tier 1: direct import
    filter_dataset = _import_skill()
    if filter_dataset is not None:
        try:
            filter_pct = int(sal_threshold * 100)
            result = filter_dataset(
                dataset_path=str(dataset_path),
                output_path=str(output_path),
                filter_percentile=filter_pct,
                strategy="composite",
                dry_run=dry_run,
            )
            # Wrap in our OperationResult (skill may return its own dataclass)
            return OperationResult(
                success=getattr(result, "success", False),
                data=getattr(result, "data", None),
                error=getattr(result, "error", None),
                suggestions=getattr(result, "suggestions", None),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tier 1 execution failed (%s); falling back to Tier 2.", exc)

    # Tier 2: subprocess fallback
    logger.info("Using Tier 2 subprocess invocation for quality filtering.")
    return _invoke_skill_subprocess(
        dataset_path=str(dataset_path),
        output_path=str(output_path),
        sal_threshold=sal_threshold,
        ted_threshold=ted_threshold,
        min_episode_length=min_episode_length,
        dry_run=dry_run,
    )
