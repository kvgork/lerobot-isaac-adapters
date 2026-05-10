"""
wm_leworldmodel
===============

Training dispatch for HF LeWorldModel (next-embedding prediction world model).

Step 1: Convert LeRobotDataset (Parquet+MP4) to LeWorldModel HDF5 (96x96,
        window=16) using the ``lerobot_world_model_bridge`` skill Python API.
        Skips conversion if the HDF5 cache already exists (idempotent).

Step 2: Invoke LeWorldModel training via
        ``python -m lerobot.scripts.train_world_model``.

        Note: the exact CLI entrypoint for HF LeWorldModel is subject to change
        as the upstream package evolves.  If ``lerobot.scripts.train_world_model``
        does not exist in your installed version, override with a custom command
        via ``--`` remainder args or adapt this module.

The HDF5 is cached at ``<output_dir>/leworldmodel_data.hdf5``.

Metric output
-------------
Parses ``pred_loss=<float>`` from training stdout and re-emits via
``metric_extractor.emit("pred_loss", ...)`` for autoresearch regex compatibility.

LeWorldModel is a next-embedding predictor, not a pixel decoder — there is
no reconstruction loss.  ``pred_loss`` is the sole primary metric.

Soft-import contract
--------------------
Do NOT import ``transformers`` or ``lerobot`` at module level. Use try/except so
argparse and tests work without these packages installed.

RTX 3080 10 GB notes
--------------------
- Image size 96x96 with 16-step windows is more VRAM-intensive than DreamerV3.
- Keep batch_size <= 8 initially.
- Enable gradient checkpointing if OOM.
- Use fp16 / AMP throughout.

HDF5 schema reference
---------------------
The LeWorldModel HDF5 schema is not fully documented in the paper. The bridge
skill uses ``quentinll/lewm-pusht`` (HF Hub) as the reference for group layout
and key naming.  The ``lerobot_world_model_bridge`` skill handles this via
``lerobot_to_worldmodel(..., image_size=(96,96), window_size=16)``.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

from lerobot_isaac_adapters.targets._subprocess import stream_training_subprocess

_PRED_LOSS_RE = re.compile(r"pred_loss[=:\s]+([0-9.eE+\-]+)")


def _convert_dataset(args: argparse.Namespace) -> Path:
    """Convert LeRobotDataset to LeWorldModel HDF5 format.

    Uses the ``lerobot_world_model_bridge`` skill Python API (imported lazily).
    Skips conversion if the cache file already exists.

    Returns
    -------
    Path
        Path to the HDF5 file.
    """
    hdf5_path = Path(args.output_dir) / "leworldmodel_data.hdf5"

    # Skip if pre-converted HDF5 path was passed directly
    if args.dataset and args.dataset.endswith((".h5", ".hdf5")):
        return Path(args.dataset)

    # Skip if cache already exists
    if hdf5_path.exists():
        print(f"[wm_leworldmodel] Conversion cache found: {hdf5_path} — skipping.")
        return hdf5_path

    # Import bridge skill Python API (soft import)
    try:
        from skills.lerobot_world_model_bridge.operations import lerobot_to_worldmodel
    except ImportError:
        raise ImportError(
            "Cannot import lerobot_world_model_bridge skill. "
            "Add ${CLAUDE_CODE_ROOT} to PYTHONPATH:\n"
            "  export PYTHONPATH=${CLAUDE_CODE_ROOT}:$PYTHONPATH"
        )

    print(
        f"[wm_leworldmodel] Converting dataset {args.dataset!r} "
        f"-> {hdf5_path} (96x96, window=16, HDF5)..."
    )
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    result = lerobot_to_worldmodel(
        dataset_path=args.dataset or "",
        output_path=str(hdf5_path),
        output_format="hdf5",
        image_size=(96, 96),
        window_size=16,
        stride=8,
        normalize_actions=True,
    )
    if not result.success:
        raise RuntimeError(
            f"[wm_leworldmodel] Dataset conversion failed: {result.error}"
        )

    print(f"[wm_leworldmodel] Conversion complete: {result.data}")
    return hdf5_path


def run(args: argparse.Namespace) -> int:
    """Dispatch a HF LeWorldModel training run.

    Parameters
    ----------
    args:
        Parsed CLI namespace from ``lerobot_isaac_adapters.train``.
        Expected attributes:
          - ``dataset``    (str | None) — Parquet dir OR pre-converted HDF5 path
          - ``config``     (str | None) — path to ``wm_leworldmodel.yaml``
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
    Primary metric: ``pred_loss`` (minimize).
    No reconstruction loss — LeWorldModel is a next-embedding predictor, not
    a pixel decoder.

    The CLI ``python -m lerobot.scripts.train_world_model`` is the expected
    entrypoint.  If your installed version of lerobot does not expose this
    module, pass an alternative command via ``--`` remainder args.
    """
    hdf5_path = Path(args.output_dir) / "leworldmodel_data.hdf5"
    if args.dataset and args.dataset.endswith((".h5", ".hdf5")):
        hdf5_path = Path(args.dataset)

    def _build_train_cmd(resolved_hdf5: Path) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.train_world_model",
            f"--dataset_path={resolved_hdf5}",
            f"--batch_size={args.batch_size}",
            f"--lr={args.lr}",
            f"--num_steps={args.steps}",
            f"--seed={args.seed}",
            f"--output_dir={args.output_dir}",
        ]
        if args.config:
            cmd.append(f"--config={args.config}")
        if getattr(args, "remainder", None):
            cmd.extend(a for a in args.remainder if a != "--")
        return cmd

    if args.dry_run:
        train_cmd = _build_train_cmd(hdf5_path)
        if not (args.dataset and args.dataset.endswith((".h5", ".hdf5"))):
            print(
                f"[wm_leworldmodel] Step 1 — convert dataset (via lerobot_world_model_bridge Python API):\n"
                f"  dataset={args.dataset!r} -> {hdf5_path} (96x96, window=16, HDF5)"
            )
        else:
            print(f"[wm_leworldmodel] Step 1 — pre-converted HDF5: {hdf5_path}")
        print(f"[wm_leworldmodel] Step 2 — train:\n  {shlex.join(train_cmd)}")
        return 0

    # Step 1: convert dataset
    try:
        hdf5_path = _convert_dataset(args)
    except (ImportError, RuntimeError) as exc:
        print(f"[wm_leworldmodel] Conversion error: {exc}", file=sys.stderr)
        return 1

    train_cmd = _build_train_cmd(hdf5_path)

    # Step 2: run LeWorldModel training
    return stream_training_subprocess(
        train_cmd,
        metric_re=_PRED_LOSS_RE,
        metric_name="pred_loss",
        label="wm_leworldmodel",
        install_hint=(
            "lerobot.scripts.train_world_model not available; "
            "install LeRobot with world-model support: pip install lerobot"
        ),
    )
