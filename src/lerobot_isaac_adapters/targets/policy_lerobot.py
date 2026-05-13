"""
policy_lerobot
==============

Training dispatch for LeRobot policy architectures:
  - smolvla
  - act
  - diffusion

Invokes the ``lerobot-train`` CLI via subprocess, streams stdout line-by-line,
and re-emits ``eval/pc_success`` metrics via ``metric_extractor.emit()`` so
that ``autoresearch-ml-executor-worker`` can parse them.

The LeRobot ``lerobot-train`` CLI prints lines of the form::

    eval/pc_success=0.73

This module strips the ``eval/`` prefix and re-emits the value to satisfy the
simpler regex ``(\\w+)[=:\\s]+([0-9.eE+-]+)`` used by the autoresearch executor.

Soft-import contract
--------------------
Do NOT import lerobot at module level.  Use a try/except block so that the
adapter's argparse layer and tests work even when lerobot is not installed.
"""

from __future__ import annotations

import argparse
import re
import shlex

from lerobot_isaac_adapters.targets._subprocess import stream_training_subprocess

# `eval/pc_success=` -> re-emitted as `pc_success=` for the executor regex.
_PC_SUCCESS_RE = re.compile(r"eval/pc_success[=:\s]+([0-9.eE+\-]+)")


def _split_dataset_arg(dataset: str | None) -> tuple[str, str | None]:
    """Split ``--dataset`` into (repo_id, optional_local_root).

    Heuristic: if the value looks like an on-disk LeRobotDataset directory
    (contains a path separator AND exists on disk), treat it as a local
    dataset — the lerobot CLI still wants a `--dataset.repo_id` so we
    derive one from the trailing two path components (`org/name`).

    Otherwise treat the value as a HuggingFace repo id verbatim.
    """
    import os

    if not dataset:
        return "<dataset>", None
    if (os.sep in dataset or "/" in dataset) and os.path.isdir(dataset):
        # Local dataset path. Derive a repo-id-like label from the last two
        # path components so cache / logging keep working.
        parts = dataset.rstrip(os.sep).split(os.sep)
        repo_id = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        return repo_id, dataset
    return dataset, None


def _lerobot_policy_type(target_arch: str) -> str:
    """Map ``--target_arch`` to the LeRobot ``--policy.type`` string.

    Parameters
    ----------
    target_arch:
        One of ``smolvla``, ``act``, ``diffusion``.

    Returns
    -------
    str
        The policy type string accepted by the ``lerobot-train`` CLI.
    """
    mapping = {
        "smolvla": "smolvla",
        "act": "act",
        "diffusion": "diffusion",
    }
    if target_arch not in mapping:
        raise ValueError(
            f"policy_lerobot.run() called with unsupported arch {target_arch!r}. "
            f"Expected one of: {list(mapping)}"
        )
    return mapping[target_arch]


def run(args: argparse.Namespace) -> int:
    """Dispatch a LeRobot policy training run.

    Parameters
    ----------
    args:
        Parsed CLI namespace from ``lerobot_isaac_adapters.train``.
        Expected attributes:
          - ``target_arch``  (str) — one of smolvla/act/diffusion
          - ``dataset``      (str | None)
          - ``config``       (str | None)
          - ``output_dir``   (str)
          - ``steps``        (int)
          - ``batch_size``   (int)
          - ``lr``           (float)
          - ``seed``         (int)
          - ``dry_run``      (bool)
          - ``remainder``    (list[str]) — extra args forwarded to lerobot-train
          - ``video_backend`` (str | None, optional) — overrides default ``pyav``
            video backend.  Default avoids torchcodec ↔ system libavutil
            version mismatches that break LeRobotDataset video loading.

    Returns
    -------
    int
        0 on success, 127 if lerobot-train not found, or the subprocess exit code.
    """
    policy_type = _lerobot_policy_type(args.target_arch)

    # video_backend default: pyav (avoid torchcodec → libavutil version mismatch
    # that breaks LeRobotDataset video loading on systems with newer ffmpeg).
    # Caller can override via --remainder ['--dataset.video_backend=torchcodec'].
    video_backend = getattr(args, "video_backend", None) or "pyav"

    # CLI shape targets lerobot >= 0.5 (was --training.batch_size / --training.num_steps
    # / --training.lr in older releases — those flags were removed). For local datasets
    # the caller can also pass `--dataset.root=<path>` via remainder args; we infer it
    # automatically when `args.dataset` looks like an on-disk path.
    dataset_repo_id, dataset_root = _split_dataset_arg(args.dataset)

    cmd = [
        "lerobot-train",
        f"--policy.type={policy_type}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.video_backend={video_backend}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--optimizer.lr={args.lr}",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir}",
        # Local-only by default. lerobot 0.5+ enforces a `policy.repo_id` when
        # `policy.push_to_hub` is true (the default), even if the user has no
        # HF account configured. Override via remainder to publish to the hub.
        "--policy.push_to_hub=false",
    ]
    if dataset_root:
        cmd.append(f"--dataset.root={dataset_root}")
    if args.config:
        cmd.insert(1, f"--config_path={args.config}")

    # Passthrough extra args (strip leading '--' separator if present)
    if getattr(args, "remainder", None):
        extra = [a for a in args.remainder if a != "--"]
        cmd.extend(extra)

    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    return stream_training_subprocess(
        cmd,
        metric_re=_PC_SUCCESS_RE,
        metric_name="pc_success",
        label="policy_lerobot",
        install_hint="Install LeRobot: pip install lerobot",
    )
