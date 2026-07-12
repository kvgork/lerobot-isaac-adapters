"""
policy_lerobot
==============

Training dispatch for LeRobot policy architectures:
  - smolvla
  - act
  - diffusion
  - vla_jepa    (lerobot >=0.6.0 world-model policy)
  - fastwam     (lerobot >=0.6.0 world-model policy)
  - lingbot_va  (lerobot >=0.6.0 world-model policy)

The world-model policies use the SAME ``lerobot-train`` CLI as the plain
policies (only ``--policy.type`` differs) and report ``eval/pc_success`` the
same way, so no separate dispatch is needed.

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
        One of ``smolvla``, ``act``, ``diffusion`` (plain policies) or
        ``vla_jepa``, ``fastwam``, ``lingbot_va`` (lerobot >=0.6.0 world-model
        policies). The mapping is 1:1 with LeRobot's ``--policy.type`` names.

    Returns
    -------
    str
        The policy type string accepted by the ``lerobot-train`` CLI.
    """
    mapping = {
        "smolvla": "smolvla",
        "act": "act",
        "diffusion": "diffusion",
        # lerobot >=0.6.0 world-model policies (identity map to --policy.type).
        "vla_jepa": "vla_jepa",
        "fastwam": "fastwam",
        "lingbot_va": "lingbot_va",
    }
    if target_arch not in mapping:
        raise ValueError(
            f"policy_lerobot.run() called with unsupported arch {target_arch!r}. "
            f"Expected one of: {list(mapping)}"
        )
    return mapping[target_arch]


# --- Pretrained world-model fine-tune recipe (vla_jepa on a 10 GB RTX 3080) ----
#
# GPU-verified 2026-07-11: a naked `--policy.path=lerobot/VLA-JEPA-Pretrain` fails
# through THREE blockers before it trains on SO-101 data. This module auto-applies
# the recipe so callers no longer hand-assemble 3 flags + a patched config:
#   1. state-dict size mismatch (pretrain 7-action/8-state vs SO-101 6/12)
#      -> --policy.reinit_modules (re-init the action/state heads).
#   2. camera KeyError (pretrain declares 2 cams; model hard-requires each) ->
#      a config.json patched down to the dataset's camera set (rename_map cannot
#      change the COUNT).
#   3. CUDA OOM at batch 4 AND 2 (full fp32 2B fine-tune) ->
#      --policy.freeze_qwen=true (freeze the VLM backbone; ~9 GB at batch 2).
# Opt out entirely with LEROBOT_ISAAC_WM_AUTORECIPE=0, or override any single flag
# by passing it explicitly. See docs/runbook/03-train-policy.md.

# Action/state heads whose dimensions differ from the pretrain corpus and so must
# be re-initialised when fine-tuning onto SO-101 (6-action/12-state).
_VLA_JEPA_REINIT_MODULES = (
    "model.action_model.action_encoder",
    "model.action_model.action_decoder",
    "model.action_model.state_encoder",
)


def _dataset_camera_keys(dataset_root: str | None) -> list[str]:
    """Return the sorted ``observation.images.*`` feature keys of a local dataset.

    Reads ``meta/info.json``. Returns ``[]`` when the root / info.json is missing
    or unreadable — callers treat empty as "unknown → do not patch".
    """
    import json
    import os

    if not dataset_root:
        return []
    info = os.path.join(dataset_root, "meta", "info.json")
    if not os.path.isfile(info):
        return []
    try:
        with open(info) as fh:
            feats = json.load(fh).get("features", {})
    except (OSError, ValueError):
        return []
    return sorted(k for k in feats if k.startswith("observation.images."))


def patch_wm_policy_config(config: dict, target_camera_keys: list[str]) -> dict:
    """Return a copy of a WM-policy ``config.json`` patched to a target camera set.

    Keeps ``len(target_camera_keys)`` image features (renamed to the target keys,
    using the first pretrain image feature as the shape template) and drops the
    surplus. Only DOWN-adapts (target fewer than pretrain); returns an unchanged
    copy when counts already match, the target is empty, or the target would ADD
    cameras (no weights to invent). Pure — no I/O.
    """
    import copy

    patched = copy.deepcopy(config)
    feats = patched.get("input_features", {})
    image_keys = [k for k in feats if k.startswith("observation.images.")]
    if (
        not image_keys
        or not target_camera_keys
        or len(image_keys) <= len(target_camera_keys)
    ):
        return patched
    template = feats[image_keys[0]]
    for k in image_keys:
        del feats[k]
    for tgt in target_camera_keys:
        feats[tgt] = copy.deepcopy(template)
    return patched


def _wm_config_image_keys(ckpt_dir: str) -> list[str] | None:
    """Return the ``observation.images.*`` input-feature keys of a local checkpoint.

    ``None`` when ``config.json`` is absent or unreadable (unknown → do not patch).
    """
    import json
    import os

    cfg = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg) as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return None
    return [
        k
        for k in config.get("input_features", {})
        if k.startswith("observation.images.")
    ]


def _materialize_patched_wm_checkpoint(
    src_dir: str, target_camera_keys: list[str], out_dir: str
) -> str:
    """Write a camera-count-patched copy of a local WM-policy checkpoint.

    Patches ``config.json`` via :func:`patch_wm_policy_config`, writes it to
    ``out_dir``, and symlinks every other checkpoint file (safetensors, etc.).
    Returns ``out_dir``.
    """
    import json
    import os

    with open(os.path.join(src_dir, "config.json")) as fh:
        config = json.load(fh)
    patched = patch_wm_policy_config(config, target_camera_keys)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(patched, fh, indent=2)
    for name in os.listdir(src_dir):
        if name == "config.json":
            continue
        dst = os.path.join(out_dir, name)
        if os.path.lexists(dst):
            continue
        os.symlink(os.path.abspath(os.path.join(src_dir, name)), dst)
    return out_dir


def _augment_pretrained_wm(
    target_arch: str,
    remainder: list[str],
    dataset_root: str | None,
    output_dir: str,
    *,
    dry_run: bool,
) -> list[str]:
    """Auto-apply the pretrained-vla_jepa fine-tune recipe to the remainder args.

    Only acts for ``vla_jepa`` fine-tuned from a pretrained ``--policy.path``.
    Appends ``--policy.freeze_qwen=true`` and ``--policy.reinit_modules`` unless
    the caller set them, and — when ``--policy.path`` is a LOCAL checkpoint whose
    config declares more cameras than the dataset — materialises a camera-count-
    patched copy and rewrites ``--policy.path`` to it. Returns a new list; the
    input is not mutated. No-op unless the recipe applies.
    """
    import os

    if os.environ.get("LEROBOT_ISAAC_WM_AUTORECIPE", "1") == "0":
        return list(remainder)
    # freeze_qwen / reinit_modules are vla_jepa policy attributes; injecting them
    # for fastwam/lingbot_va would be a draccus error, so gate strictly.
    if target_arch != "vla_jepa":
        return list(remainder)
    remainder = list(remainder)
    idx_by_key = {a.split("=", 1)[0]: i for i, a in enumerate(remainder)}
    if "--policy.path" not in idx_by_key:
        return remainder  # training from scratch: recipe N/A

    # (2) camera-count adaptation — local checkpoints only.
    path_idx = idx_by_key["--policy.path"]
    entry = remainder[path_idx]
    path_val = entry.split("=", 1)[1] if "=" in entry else ""
    if path_val and os.path.isdir(path_val):
        cams = _dataset_camera_keys(dataset_root)
        src_img_keys = _wm_config_image_keys(path_val)
        if cams and src_img_keys is not None and len(src_img_keys) > len(cams):
            out_dir = os.path.join(output_dir or ".", "_wm_policy_patched")
            if not dry_run:
                _materialize_patched_wm_checkpoint(path_val, cams, out_dir)
            remainder[path_idx] = f"--policy.path={out_dir}"
            print(
                f"[policy_lerobot] vla_jepa: camera count {len(src_img_keys)}->"
                f"{len(cams)} {cams}; "
                f"{'patched' if not dry_run else 'would patch'} checkpoint at {out_dir}"
            )
    elif path_val and not os.path.isdir(path_val) and not dry_run:
        print(
            f"[policy_lerobot] vla_jepa: --policy.path={path_val} is not a local "
            "dir; skipping camera-count adaptation (materialise it locally first "
            "if the pretrain camera count differs from the dataset)."
        )

    # (1) freeze_qwen + reinit_modules — append only if the caller did not set them.
    keys = {a.split("=", 1)[0] for a in remainder}
    if "--policy.freeze_qwen" not in keys:
        remainder.append("--policy.freeze_qwen=true")
        print(
            "[policy_lerobot] vla_jepa: auto-added --policy.freeze_qwen=true "
            "(RTX-3080 fit lever; override with --policy.freeze_qwen=false)."
        )
    if "--policy.reinit_modules" not in keys:
        mods = ",".join(f'"{m}"' for m in _VLA_JEPA_REINIT_MODULES)
        remainder.append(f"--policy.reinit_modules=[{mods}]")
        print(
            "[policy_lerobot] vla_jepa: auto-added --policy.reinit_modules "
            "(re-init action/state heads for a different action/state dim)."
        )
    return remainder


def run(args: argparse.Namespace) -> int:
    """Dispatch a LeRobot policy training run.

    Parameters
    ----------
    args:
        Parsed CLI namespace from ``lerobot_isaac_adapters.train``.
        Expected attributes:
          - ``target_arch``  (str) — one of smolvla/act/diffusion or the
            lerobot >=0.6.0 world-model policies vla_jepa/fastwam/lingbot_va
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

    # Omit --policy.type when the caller loads a pretrained checkpoint via
    # `--policy.path=` (lerobot infers the policy type from the checkpoint;
    # passing both --policy.path and --policy.type is a draccus conflict). This
    # is the recommended entry for the world-model policies, e.g.
    # `-- --policy.path=lerobot/VLA-JEPA-Pretrain`. The smolvla resume flag
    # `--policy.pretrained_path=` coexists with --policy.type and is left alone.
    _remainder = getattr(args, "remainder", None) or []
    _has_policy_path = any(a.split("=", 1)[0] == "--policy.path" for a in _remainder)
    if not _has_policy_path:
        cmd.append(f"--policy.type={policy_type}")

    cmd += [
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
        # Append (do NOT insert at a fixed index): when needs_wrapper is True the
        # cmd prefix is [python, "-m", module], so cmd.insert(1, ...) would wedge
        # the flag between the interpreter and "-m" and abort before the module
        # loads. lerobot/draccus parse --key=value order-independently, so
        # appending among the other lerobot flags is safe for both cmd shapes.
        cmd.append(f"--config_path={args.config}")

    # Passthrough extra args (strip leading '--' separator if present), then
    # auto-apply the pretrained world-model fine-tune recipe (vla_jepa on a
    # 10 GB GPU): freeze_qwen + reinit_modules + camera-count config patch.
    # See _augment_pretrained_wm + docs/runbook/03-train-policy.md.
    extra = [a for a in (getattr(args, "remainder", None) or []) if a != "--"]
    extra = _augment_pretrained_wm(
        args.target_arch, extra, dataset_root, args.output_dir, dry_run=args.dry_run
    )
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
