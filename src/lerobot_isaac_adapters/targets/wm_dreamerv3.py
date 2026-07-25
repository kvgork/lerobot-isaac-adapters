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

Plan2Explore (p2e_dv3) support
-------------------------------
Pass ``--exp p2e_dv3_exploration`` (reward-free intrinsic-reward pre-training)
or ``--exp p2e_dv3_finetuning`` (resume exploration ckpt with extrinsic rewards)
via ``args.exp`` (or env var ``LEROBOT_ISAAC_EXP``).  Both variants share the
same env wiring and monkeypatches as ``dreamer_v3``.  The adapter resolves the
exp name via: args.exp → LEROBOT_ISAAC_EXP → "dreamer_v3" (default).

Double-exp guard: if a ``exp=`` token already exists in ``args.remainder`` the
adapter suppresses emitting its own ``exp=`` to avoid hydra
ConfigCompositionException on duplicate overrides.

Finetuning ckpt guard: when ``exp_name`` ends with ``_finetuning`` the adapter
requires a checkpoint path via ``args.exploration_ckpt`` or the env var
``LEROBOT_ISAAC_EXPLORATION_CKPT``, unless ``checkpoint.exploration_ckpt_path=``
is already present in ``args.remainder``.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

from lerobot_isaac_adapters.targets._subprocess import stream_training_subprocess

_RECON_LOSS_RE = re.compile(r"recon_loss[=:\s]+([0-9.eE+\-]+)")

# Path to the bundled sheeprl plugin configs directory.
# Computed from __file__ to avoid importing sheeprl_plugin (which eagerly imports
# gymnasium at module level and would break tests in environments without gymnasium).
# Layout: targets/wm_dreamerv3.py → ../sheeprl_plugin/configs/
_PLUGIN_CONFIGS_DIR = str(
    Path(__file__).resolve().parent.parent / "sheeprl_plugin" / "configs"
)


def _resolve_exp_name(args: argparse.Namespace) -> str:
    """Resolve the sheeprl exp= config name.

    Priority order:
    1. ``args.exp`` (CLI ``--exp`` flag)
    2. ``LEROBOT_ISAAC_EXP`` environment variable
    3. ``"dreamer_v3"`` (default — preserves full back-compat)
    """
    return (
        getattr(args, "exp", None)
        or os.environ.get("LEROBOT_ISAAC_EXP")
        or "dreamer_v3"
    )


def _remainder_has_exp(remainder: list[str]) -> bool:
    """Return True if any token in ``remainder`` starts with ``exp=``."""
    return any(tok.startswith("exp=") for tok in (remainder or []))


def _remainder_has_exploration_ckpt(remainder: list[str]) -> bool:
    """Return True if ``checkpoint.exploration_ckpt_path=`` is in remainder."""
    return any(
        tok.startswith("checkpoint.exploration_ckpt_path=")
        for tok in (remainder or [])
    )


def _remainder_has_resume_from(remainder: list[str]) -> bool:
    """Return True if ``checkpoint.resume_from=`` is in remainder."""
    return any(
        tok.startswith("checkpoint.resume_from=") for tok in (remainder or [])
    )


def _parse_image_size(raw: str | None) -> tuple[int, int]:
    """Parse the ``--image_size`` flag into an ``(H, W)`` tuple.

    Accepts a single int ``"64"`` -> ``(64, 64)`` or ``"H,W"`` -> ``(H, W)``.
    Returns the DreamerV3 default ``(64, 64)`` when ``raw`` is None/empty.
    """
    if not raw:
        return (64, 64)
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise ValueError(
        f"--image_size must be 'N' or 'H,W'; got {raw!r}"
    )


def _resolve_bridge_kwargs(args: argparse.Namespace) -> dict:
    """Build the optional image/state-key + image_size kwargs for the bridge.

    Purely additive: when none of ``--camera_key`` / ``--state_keys`` /
    ``--image_size`` are set the bridge sees ``image_keys=None`` (auto-detect),
    ``state_keys=None`` (auto-detect) and ``image_size=(64, 64)`` — identical
    to the prior hard-coded behaviour.
    """
    camera_key = getattr(args, "camera_key", None)
    state_keys_raw = getattr(args, "state_keys", None)
    state_keys = (
        [s.strip() for s in str(state_keys_raw).split(",") if s.strip()]
        if state_keys_raw
        else None
    )
    return {
        "image_size": _parse_image_size(getattr(args, "image_size", None)),
        "image_keys": [camera_key] if camera_key else None,
        "state_keys": state_keys,
    }


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

    # World-model bridge API. Primary home is now this package
    # (lerobot_isaac_adapters.data.world_model_bridge); the legacy skill path is a
    # fallback for older checkouts where the skill hasn't been refactored yet.
    try:
        from lerobot_isaac_adapters.data.world_model_bridge import lerobot_to_worldmodel
    except ImportError:
        try:
            from skills.lerobot_world_model_bridge.operations import lerobot_to_worldmodel
        except ImportError:
            raise ImportError(
                "Cannot import the world-model bridge. Expected "
                "lerobot_isaac_adapters.data.world_model_bridge (this package) or, as a "
                "fallback, the lerobot_world_model_bridge skill on PYTHONPATH."
            )

    bridge_kwargs = _resolve_bridge_kwargs(args)
    print(
        f"[wm_dreamerv3] Converting dataset {args.dataset!r} "
        f"-> {hdf5_path} ({bridge_kwargs['image_size']}, HDF5)..."
    )
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    result = lerobot_to_worldmodel(
        dataset_path=args.dataset or "",
        output_path=str(hdf5_path),
        output_format="hdf5",
        window_size=16,
        stride=8,
        normalize_actions=True,
        **bridge_kwargs,
    )
    if not result.success:
        raise RuntimeError(f"[wm_dreamerv3] Dataset conversion failed: {result.error}")

    print(f"[wm_dreamerv3] Conversion complete: {result.data}")
    return hdf5_path


def run(args: argparse.Namespace) -> int:
    """Dispatch a DreamerV3 world-model training run.

    Parameters
    ----------
    args:
        Parsed CLI namespace from ``lerobot_isaac_adapters.train``.
        Expected attributes:
          - ``dataset``          (str | None) — Parquet dir OR pre-converted HDF5 path
          - ``config``           (str | None) — path to ``wm_dreamerv3.yaml``
          - ``output_dir``       (str)
          - ``steps``            (int)
          - ``batch_size``       (int)
          - ``lr``               (float)
          - ``seed``             (int)
          - ``dry_run``          (bool)
          - ``exp``              (str | None) — sheeprl exp name; None → "dreamer_v3"
          - ``exploration_ckpt`` (str | None) — path for p2e finetuning ckpt
          - ``remainder``        (list[str])

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
    # Two env paths:
    #   * env=custom_hdf5  → replay env, needs an HDF5 bridge step first.
    #   * env=isaac_so101  → real Isaac Lab env, NO bridge needed (the env
    #                       is live — actions affect physics, reward is
    #                       emitted by the env's RewardManager).
    # The caller picks via `--env <name>` (passed through the wrapper's
    # remainder OR set explicitly on args.env). Default = custom_hdf5 for
    # backwards compat with the existing autoresearch sweep.
    env_name = getattr(args, "env", None) or "custom_hdf5"
    if "--env" in (getattr(args, "remainder", []) or []):
        rem = args.remainder
        for i, tok in enumerate(rem):
            if tok == "--env" and i + 1 < len(rem):
                env_name = rem[i + 1]
                break

    use_isaac_env = env_name in ("isaac_so101", "isaac")
    hdf5_path = Path(args.output_dir) / "dreamerv3_data.hdf5"
    if args.dataset and args.dataset.endswith((".h5", ".hdf5")):
        hdf5_path = Path(args.dataset)

    # Resolve exp name: --exp flag > LEROBOT_ISAAC_EXP env var > "dreamer_v3"
    exp_name = _resolve_exp_name(args)

    # Finetuning ckpt guard: p2e_dv3_finetuning requires a ckpt path.
    remainder = list(getattr(args, "remainder", []) or [])
    if exp_name.endswith("_finetuning") and not _remainder_has_exploration_ckpt(
        remainder
    ):
        ckpt_path = getattr(args, "exploration_ckpt", None) or os.environ.get(
            "LEROBOT_ISAAC_EXPLORATION_CKPT"
        )
        if ckpt_path:
            remainder.append(
                f"checkpoint.exploration_ckpt_path={Path(ckpt_path).resolve()}"
            )
        else:
            msg = (
                f"[wm_dreamerv3] ERROR: exp={exp_name!r} requires a checkpoint path.\n"
                "  Provide it via one of:\n"
                "    --exploration_ckpt /path/to/exploration/ckpt\n"
                "    LEROBOT_ISAAC_EXPLORATION_CKPT=/path/to/exploration/ckpt\n"
                "    -- checkpoint.exploration_ckpt_path=/path/to/exploration/ckpt"
            )
            print(msg, file=sys.stderr)
            return 1

    # Native sheeprl resume (any exp): append checkpoint.resume_from=<path>.
    # Distinct from the _finetuning exploration_ckpt_path branch above —
    # resume_from rehydrates a full sheeprl run state (sheeprl/cli.py:362).
    if not _remainder_has_resume_from(remainder):
        resume_from = getattr(args, "resume_from", None) or os.environ.get(
            "LEROBOT_ISAAC_RESUME_FROM"
        )
        if resume_from:
            remainder.append(
                f"checkpoint.resume_from={Path(resume_from).resolve()}"
            )

    def _build_train_cmd(resolved_hdf5: Path) -> list[str]:
        # sheeprl entrypoint: `python -m sheeprl` (-> sheeprl/__main__.py).
        # `python -m sheeprl.cli` runs the module body but does NOT dispatch the
        # @hydra.main-decorated `run()` function. Use `-m sheeprl` instead.
        #
        # `env=custom_hdf5` resolves against our bundled config dir
        # `lerobot_isaac_adapters/sheeprl_plugin/configs/env/custom_hdf5.yaml`,
        # which wraps `HDF5ReplayEnv` and feeds the bridge-produced HDF5
        # to sheeprl's dreamer_v3 directly. Override via remainder if you
        # have a different sheeprl env registered (`-- env=dmc`, etc.).
        #
        # _PLUGIN_CONFIGS_DIR is pre-computed from __file__ to avoid importing
        # sheeprl_plugin (which has an eager `gymnasium` import that breaks tests
        # in the default pixi env where gymnasium is not installed).

        if use_isaac_env:
            # Isaac Lab needs SimulationApp booted BEFORE sheeprl imports —
            # `python -m sheeprl` loses libgobject to hydra+lightning first
            # and Isaac Sim's gpu_foundation plugin then fails to load.
            # Our `_wm_isaac_entry.py` claims libgobject via AppLauncher
            # before delegating to sheeprl.cli.run.
            entry = (
                Path(__file__).resolve().parents[3].parents[1]
                / "scripts"
                / "_wm_isaac_entry.py"
            )
            # Fallback: resolve from workspace root in case file lives in
            # an installed site-packages copy.
            if not entry.is_file():
                from os import environ

                ws = Path(environ.get("LEROBOT_ISAAC_WORKSPACE", Path.cwd()))
                entry = ws / "scripts" / "_wm_isaac_entry.py"
            cmd = [sys.executable, str(entry)]
        else:
            cmd = [sys.executable, "-m", "sheeprl"]
        cmd += [
            f"--config-dir={_PLUGIN_CONFIGS_DIR}",
        ]

        # Double-exp guard: only emit our own exp= when the remainder does NOT
        # already contain an exp= override (hydra raises ConfigCompositionException
        # on duplicate overrides).
        if not _remainder_has_exp(remainder):
            cmd.append(f"exp={exp_name}")

        cmd += [
            f"env={env_name}",
        ]
        if not use_isaac_env:
            # HDF5 replay env needs the dataset path injected.
            cmd.append(f"+env.dataset_path={resolved_hdf5}")
        cmd += [
            f"algo.per_rank_batch_size={args.batch_size}",
            f"algo.world_model.optimizer.lr={args.lr}",
            f"algo.total_steps={args.steps}",
            f"seed={args.seed}",
            f"hydra.run.dir={args.output_dir}",
        ]
        # Append remainder (already has exploration_ckpt_path injected if needed),
        # stripping the synthetic `--env <name>` tokens we consumed above
        # so they don't reach sheeprl as garbage.
        if remainder:
            skip = 0
            for tok in remainder:
                if skip:
                    skip -= 1
                    continue
                if tok == "--":
                    continue
                if tok == "--env":
                    skip = 1
                    continue
                cmd.append(tok)
        return cmd

    if args.dry_run:
        train_cmd = _build_train_cmd(hdf5_path)
        if use_isaac_env:
            print(
                f"[wm_dreamerv3] env={env_name} (Isaac Lab) — bridge step skipped, "
                f"actor will learn against live physics + RewardManager."
            )
        elif not (args.dataset and args.dataset.endswith((".h5", ".hdf5"))):
            _bk = _resolve_bridge_kwargs(args)
            _extra = []
            if _bk["image_keys"]:
                _extra.append(f"image_keys={_bk['image_keys']}")
            if _bk["state_keys"]:
                _extra.append(f"state_keys={_bk['state_keys']}")
            _extra_s = (" " + " ".join(_extra)) if _extra else ""
            print(
                f"[wm_dreamerv3] Step 1 — convert dataset (via lerobot_world_model_bridge Python API):\n"
                f"  dataset={args.dataset!r} -> {hdf5_path} "
                f"({_bk['image_size']} HDF5){_extra_s}"
            )
        else:
            print(f"[wm_dreamerv3] Step 1 — pre-converted HDF5: {hdf5_path}")
        print(f"[wm_dreamerv3] Step 2 — train:\n  {shlex.join(train_cmd)}")
        return 0

    # Step 1: convert dataset (skip when running against a live env).
    if not use_isaac_env:
        try:
            hdf5_path = _convert_dataset(args)
        except (ImportError, RuntimeError) as exc:
            print(f"[wm_dreamerv3] Conversion error: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"[wm_dreamerv3] env={env_name} — skipping HDF5 bridge step.")

    train_cmd = _build_train_cmd(hdf5_path)

    # Step 2: run sheeprl
    return stream_training_subprocess(
        train_cmd,
        metric_re=_RECON_LOSS_RE,
        metric_name="recon_loss",
        label="wm_dreamerv3",
        install_hint="Install: pip install sheeprl",
    )
