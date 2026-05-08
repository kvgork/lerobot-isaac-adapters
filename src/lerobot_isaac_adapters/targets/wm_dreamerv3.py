"""
wm_dreamerv3
============

Training dispatch for DreamerV3 world models.

Step 1: Convert LeRobotDataset (Parquet+MP4) to DreamerV3 HDF5 (64x64) using
        the ``lerobot_world_model_bridge`` skill Python API.
        Skips conversion if the HDF5 cache already exists (idempotent).

Step 2: Invoke sheeprl DreamerV3 training via subprocess.

The HDF5 is cached at ``<output_dir>/dreamerv3_data.hdf5`` so repeated runs
with the same dataset do not re-convert.

Metric output
-------------
Parses ``recon_loss=<float>`` from sheeprl stdout and re-emits via
``metric_extractor.emit("recon_loss", ...)`` for autoresearch regex compatibility.

Soft-import contract
--------------------
Do NOT import sheeprl or dreamerv3 at module level.  Use try/except so argparse
and tests work without the backend installed.

RTX 3080 10 GB notes
--------------------
- image_size (64, 64) per DreamerV3 convention.
- batch_size <= 16 initially; increase if VRAM allows.
- Enable AMP (automatic mixed precision) if sheeprl supports it.
- num_envs=1 for data collection replay.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

_RECON_LOSS_RE = re.compile(r"recon_loss[=:\s]+([0-9.eE+\-]+)")


def _convert_dataset(args: argparse.Namespace) -> Path:
    """Convert LeRobotDataset to DreamerV3 HDF5 format.

    Uses the ``lerobot_world_model_bridge`` skill Python API (imported lazily).
    Skips conversion if the cache file already exists.

    Returns
    -------
    Path
        Path to the HDF5 file.
    """
    hdf5_path = Path(args.output_dir) / "dreamerv3_data.hdf5"

    # Skip if pre-converted HDF5 path was passed directly
    if args.dataset and args.dataset.endswith((".h5", ".hdf5")):
        return Path(args.dataset)

    # Skip if cache already exists
    if hdf5_path.exists():
        print(f"[wm_dreamerv3] Conversion cache found: {hdf5_path} — skipping.")
        return hdf5_path

    # Import bridge skill Python API (soft import)
    try:
        from skills.lerobot_world_model_bridge.operations import lerobot_to_worldmodel
    except ImportError:
        # Skill may not be on PYTHONPATH; provide helpful guidance
        raise ImportError(
            "Cannot import lerobot_world_model_bridge skill. "
            "Add /home/koen/tools/claude_code to PYTHONPATH:\n"
            "  export PYTHONPATH=/home/koen/tools/claude_code:$PYTHONPATH"
        )

    print(
        f"[wm_dreamerv3] Converting dataset {args.dataset!r} "
        f"-> {hdf5_path} (64x64, HDF5)..."
    )
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    result = lerobot_to_worldmodel(
        dataset_path=args.dataset or "",
        output_path=str(hdf5_path),
        output_format="hdf5",
        image_size=(64, 64),
        window_size=16,
        stride=8,
        normalize_actions=True,
    )
    if not result.success:
        raise RuntimeError(
            f"[wm_dreamerv3] Dataset conversion failed: {result.error}"
        )

    print(f"[wm_dreamerv3] Conversion complete: {result.data}")
    return hdf5_path


def run(args: argparse.Namespace) -> int:
    """Dispatch a DreamerV3 world-model training run.

    Parameters
    ----------
    args:
        Parsed CLI namespace from ``lerobot_isaac_adapters.train``.
        Expected attributes:
          - ``dataset``    (str | None) — Parquet dir OR pre-converted HDF5 path
          - ``config``     (str | None) — path to ``wm_dreamerv3.yaml``
          - ``output_dir`` (str)
          - ``steps``      (int)
          - ``batch_size`` (int)
          - ``lr``         (float)
          - ``seed``       (int)
          - ``dry_run``    (bool)
          - ``remainder``  (list[str])

    Returns
    -------
    int
        0 on success, non-zero on failure.

    Notes
    -----
    Primary metric: ``recon_loss`` (minimize).
    Secondary metric: ``pred_loss`` (minimize).
    Both emitted via ``metric_extractor.emit()``.

    sheeprl requires a custom env registered for HDF5 replay.  Users must
    register ``env=custom_hdf5`` before invoking this target.  See the
    ``sheeprl`` documentation for custom env registration.
    """
    hdf5_path = Path(args.output_dir) / "dreamerv3_data.hdf5"
    if args.dataset and args.dataset.endswith((".h5", ".hdf5")):
        hdf5_path = Path(args.dataset)

    def _build_train_cmd(resolved_hdf5: Path) -> list[str]:
        cmd = [
            "python", "-m", "sheeprl.cli",
            "exp=dreamer_v3",
            "env=custom_hdf5",  # user must register this env; see sheeprl docs
            f"env.dataset_path={resolved_hdf5}",
            f"algo.batch_size={args.batch_size}",
            f"algo.lr={args.lr}",
            f"total_steps={args.steps}",
            f"seed={args.seed}",
            f"checkpoint.save_dir={args.output_dir}",
        ]
        if getattr(args, "remainder", None):
            cmd.extend(a for a in args.remainder if a != "--")
        return cmd

    if args.dry_run:
        train_cmd = _build_train_cmd(hdf5_path)
        if not (args.dataset and args.dataset.endswith((".h5", ".hdf5"))):
            print(
                f"[wm_dreamerv3] Step 1 — convert dataset (via lerobot_world_model_bridge Python API):\n"
                f"  dataset={args.dataset!r} -> {hdf5_path} (64x64 HDF5)"
            )
        else:
            print(f"[wm_dreamerv3] Step 1 — pre-converted HDF5: {hdf5_path}")
        print(f"[wm_dreamerv3] Step 2 — train:\n  {shlex.join(train_cmd)}")
        return 0

    # Step 1: convert dataset
    try:
        hdf5_path = _convert_dataset(args)
    except (ImportError, RuntimeError) as exc:
        print(f"[wm_dreamerv3] Conversion error: {exc}", file=sys.stderr)
        return 1

    train_cmd = _build_train_cmd(hdf5_path)

    # Step 2: run sheeprl
    try:
        proc = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print(
            "[wm_dreamerv3] ERROR: 'sheeprl' not found. "
            "Install: pip install sheeprl",
            file=sys.stderr,
        )
        return 127

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        m = _RECON_LOSS_RE.search(line)
        if m:
            from lerobot_isaac_adapters.metric_extractor import emit
            emit("recon_loss", float(m.group(1)))

    proc.wait()
    if proc.returncode != 0:
        print(
            f"\033[31m[wm_dreamerv3] Training failed (exit={proc.returncode}) "
            f"— see stdout above\033[0m",
            file=sys.stderr,
        )
    return proc.returncode
