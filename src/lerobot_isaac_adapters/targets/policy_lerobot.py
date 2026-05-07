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
import subprocess
import sys
from typing import Optional

_PC_SUCCESS_RE = re.compile(r"eval/pc_success[=:\s]+([0-9.eE+\-]+)")


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

    Returns
    -------
    int
        0 on success, 127 if lerobot-train not found, or the subprocess exit code.
    """
    policy_type = _lerobot_policy_type(args.target_arch)

    cmd = [
        "lerobot-train",
        f"--policy.type={policy_type}",
        f"--dataset.repo_id={args.dataset or '<dataset>'}",
        f"--training.batch_size={args.batch_size}",
        f"--training.num_steps={args.steps}",
        f"--training.lr={args.lr}",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir}",
    ]
    if args.config:
        cmd.insert(1, f"--config={args.config}")

    # Passthrough extra args (strip leading '--' separator if present)
    if getattr(args, "remainder", None):
        extra = [a for a in args.remainder if a != "--"]
        cmd.extend(extra)

    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print(
            "[policy_lerobot] ERROR: 'lerobot-train' not found in PATH. "
            "Install LeRobot: pip install lerobot",
            file=sys.stderr,
        )
        return 127

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        m = _PC_SUCCESS_RE.search(line)
        if m:
            from lerobot_isaac_adapters.metric_extractor import emit
            emit("pc_success", float(m.group(1)))

    proc.wait()
    if proc.returncode != 0:
        print(
            f"\033[31m[policy_lerobot] Training failed (exit={proc.returncode}) "
            f"— see stdout above\033[0m",
            file=sys.stderr,
        )
    return proc.returncode
