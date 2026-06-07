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


def _successful_episode_indices(dataset_root: str) -> list[int] | None:
    """Read the recorder's per-episode success sidecar and return success indices.

    The dual-write recorder (``robot-data-recorder``) drops reward/done from the
    parquet features (lerobot 0.5.1 + numpy>=2 crash on shape-(1,) scalar
    features) and instead writes ``meta/episode_labels.json`` next to the parquet
    data. This reader is intentionally inline JSON — the adapter must NOT import
    ``robot_data_recorder`` (it is a standalone package, not a meta dependency).

    Returns
    -------
    list[int] | None
        Sorted ``episode_index`` values whose episode is a success, or ``None``
        when no sidecar is present (i.e. the dataset was never success-labelled).
    """
    import json
    import os

    sidecar = os.path.join(dataset_root, "meta", "episode_labels.json")
    if not os.path.isfile(sidecar):
        return None
    with open(sidecar) as fh:
        payload = json.load(fh)
    keep: list[int] = []
    for ep in payload.get("episodes", []):
        idx = ep.get("episode_index")
        if idx is None:
            continue
        if bool(ep.get("success", False)) or float(ep.get("terminal_reward", 0.0)) > 0.0:
            keep.append(int(idx))
    return sorted(keep)


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
    #
    # Multi-dataset (B.2): train.py normalises --dataset/--datasets into
    # args.dataset_list. With >1 entry we forward a comma-joined repo_id to
    # lerobot's MultiLeRobotDataset path. (A single local --dataset.root is the
    # only per-dataset root the lerobot CLI accepts; multiple *local* roots must
    # be merged first — flagged below in the dry-run summary.)
    dataset_list = getattr(args, "dataset_list", None) or (
        [args.dataset] if args.dataset else []
    )
    specs = [_split_dataset_arg(d) for d in dataset_list] or [
        _split_dataset_arg(args.dataset)
    ]
    multi = len(specs) > 1
    dataset_repo_id = ",".join(s[0] for s in specs)
    roots = [s[1] for s in specs if s[1]]
    # Single combined root only when exactly one local root is present.
    dataset_root = roots[0] if len(roots) == 1 else None
    multi_local_roots = len(roots) > 1

    # Decide whether to route through the in-process wrapper (cached or LoRA).
    # The wrapper monkey-patches make_dataset (cache) and/or make_policy (LoRA)
    # before dispatching to the same lerobot main.
    use_lora = getattr(args, "use_lora", False)
    needs_wrapper = bool(use_lora or getattr(args, "cache_frames", False))

    if needs_wrapper:
        import sys

        cmd = [
            sys.executable,
            "-m",
            "lerobot_isaac_adapters.cli_train_cached",
        ]
    else:
        cmd = ["lerobot-train"]

    cmd += [
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

    # --successes_only: drop failure demonstrations from BC training by passing
    # the successful episode indices to lerobot-train's `--dataset.episodes`.
    # The success labels come from the recorder's parquet sidecar; filtering is
    # only possible for a single local dataset root (the sidecar lives on disk).
    if getattr(args, "successes_only", False):
        if dataset_root and not multi_local_roots:
            keep = _successful_episode_indices(dataset_root)
            if keep is None:
                print(
                    "[policy_lerobot] --successes_only: no meta/episode_labels.json "
                    f"in {dataset_root}; training on ALL episodes."
                )
            elif len(keep) == 0:
                print(
                    "[policy_lerobot] --successes_only: 0 successful episodes in the "
                    "sidecar; training on ALL episodes (investigate the recording)."
                )
            else:
                cmd.append(f"--dataset.episodes=[{','.join(str(i) for i in keep)}]")
                print(
                    f"[policy_lerobot] --successes_only: training on {len(keep)} "
                    "successful episode(s); failures dropped."
                )
        else:
            print(
                "[policy_lerobot] --successes_only ignored: requires a single local "
                "dataset root (success labels are stored on disk next to the parquet)."
            )

    if args.config:
        cmd.insert(1, f"--config_path={args.config}")

    # Passthrough extra args (strip leading '--' separator if present)
    if getattr(args, "remainder", None):
        extra = [a for a in args.remainder if a != "--"]
        cmd.extend(extra)

    if args.dry_run:
        if multi:
            print(f"[policy_lerobot] multi-dataset ({len(specs)}):")
            for repo_id, root in specs:
                print(f"  - {repo_id}  root={root or '(hf)'}")
            if multi_local_roots:
                print(
                    "[policy_lerobot] WARNING: multiple local roots — lerobot-train "
                    "accepts a single --dataset.root. Merge first via "
                    "`python -m lerobot_isaac_synthetic.merge` before a real run."
                )
        print(shlex.join(cmd))
        if use_lora:
            print(
                f"[policy_lerobot] LoRA enabled: r={args.lora_rank} "
                f"alpha={args.lora_alpha} dropout={args.lora_dropout} "
                f"target_modules={args.lora_target_modules}"
            )
        return 0

    # Multiple *local* dataset roots cannot be expressed on the lerobot CLI
    # (single --dataset.root only). Refuse a real run and point at merge.
    if multi_local_roots:
        import sys

        print(
            "[policy_lerobot] ERROR: multiple local dataset roots are not "
            "supported by lerobot-train. Merge them first:\n"
            "  python -m lerobot_isaac_synthetic.merge --real <a> --sim <b> "
            "--out <merged>\n"
            "then train on the merged dataset with --dataset <merged>.",
            file=sys.stderr,
        )
        return 2

    # Forward cache-knob to the wrapper subprocess via env (cli_train_cached
    # reads LEROBOT_ISAAC_CACHE_RAM_GB at make_dataset patch time).
    if getattr(args, "cache_frames", False):
        import os

        os.environ["LEROBOT_ISAAC_CACHE_RAM_GB"] = str(
            float(getattr(args, "cache_ram_gb", 8.0))
        )

    # Forward LoRA knobs to the wrapper subprocess via env vars.
    # cli_train_cached reads these at policy-construction time (same pattern
    # as LEROBOT_ISAAC_CACHE_RAM_GB).
    if use_lora:
        import os

        os.environ["LEROBOT_ISAAC_LORA_RANK"] = str(args.lora_rank)
        os.environ["LEROBOT_ISAAC_LORA_ALPHA"] = str(args.lora_alpha)
        os.environ["LEROBOT_ISAAC_LORA_DROPOUT"] = str(args.lora_dropout)
        os.environ["LEROBOT_ISAAC_LORA_TARGET_MODULES"] = args.lora_target_modules
        os.environ["LEROBOT_ISAAC_USE_LORA"] = "1"

    return stream_training_subprocess(
        cmd,
        metric_re=_PC_SUCCESS_RE,
        metric_name="pc_success",
        label="policy_lerobot",
        install_hint="Install LeRobot: pip install lerobot",
    )
