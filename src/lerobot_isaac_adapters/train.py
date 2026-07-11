"""
train.py
========

Single entrypoint for all lerobot-isaac training runs.

Usage
-----
::

    lerobot-isaac-train --target_arch smolvla --dataset /data/my_dataset ...
    python -m lerobot_isaac_adapters.train --target_arch dreamerv3 ...

The ``--target_arch`` argument determines which backend is invoked:

- ``smolvla``        -> ``targets.policy_lerobot.run()``
- ``act``            -> ``targets.policy_lerobot.run()``
- ``diffusion``      -> ``targets.policy_lerobot.run()``
- ``vla_jepa``       -> ``targets.policy_lerobot.run()``  (lerobot >=0.6.0 WM policy)
- ``fastwam``        -> ``targets.policy_lerobot.run()``  (lerobot >=0.6.0 WM policy)
- ``lingbot_va``     -> ``targets.policy_lerobot.run()``  (lerobot >=0.6.0 WM policy)
- ``dreamerv3``      -> ``targets.wm_dreamerv3.run()``
- ``le_world_model`` -> ``targets.wm_leworldmodel.run()``

The ``vla_jepa`` / ``fastwam`` / ``lingbot_va`` archs are the world-model
*policies* introduced in lerobot 0.6.0. They are ordinary LeRobot policies
(they emit ``pc_success``) that use a world model as a training-time auxiliary,
so they dispatch through the same ``lerobot-train`` subprocess as the plain
policies — NOT through the predictive world-model backends (dreamerv3 /
le_world_model), which are a different concept.

All backends accept the same ``argparse.Namespace`` argument and emit metrics
to stdout via ``metric_extractor.emit()``.

Extra arguments (after ``--``) are passed through to the backend unchanged via
``args.remainder``.
"""

from __future__ import annotations

import argparse
import sys

_POLICY_ARCHS = ("smolvla", "act", "diffusion")
# lerobot >=0.6.0 world-model policies. These are POLICIES (they emit
# pc_success) that use a world model as a *training-time* auxiliary; they
# dispatch through the same `lerobot-train` subprocess as the plain policies
# above (targets.policy_lerobot), NOT through the predictive world-model
# backends below. vla_jepa (~2B, WM dropped at inference, ships pretrained
# ckpts) is the only one that fits an RTX 3080 10GB; fastwam (~5B) and
# lingbot_va (~5B + ~20GB frozen components) are registered for larger
# hardware — see docs/runbook/03-train-policy.md and the RTX-3080 pitfalls.
_WM_POLICY_ARCHS = ("vla_jepa", "fastwam", "lingbot_va")
# Predictive world-model backends (separate dispatch + their own metrics).
_WM_ARCHS = ("dreamerv3", "le_world_model")
_ALL_ARCHS = _POLICY_ARCHS + _WM_POLICY_ARCHS + _WM_ARCHS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-isaac-train",
        description=(
            "Unified training entrypoint for LeRobot + Isaac Lab.\n"
            "Dispatches to policy (smolvla/act/diffusion), lerobot 0.6.0 "
            "world-model policy (vla_jepa/fastwam/lingbot_va), or predictive "
            "world-model (dreamerv3/le_world_model) backends based on --target_arch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Any extra arguments after '--' are forwarded to the backend.\n\n"
            "Metric output format (one line per eval, parsed by autoresearch):\n"
            "  pc_success=0.73\n"
            "  recon_loss=0.0317\n"
            "  pred_loss=0.0214\n"
        ),
    )

    parser.add_argument(
        "--target_arch",
        required=True,
        choices=list(_ALL_ARCHS),
        metavar="ARCH",
        help=(
            "Training backend to use. "
            f"Policy archs: {', '.join(_POLICY_ARCHS)}. "
            f"World-model policy archs (lerobot >=0.6.0): "
            f"{', '.join(_WM_POLICY_ARCHS)}. "
            f"Predictive world-model archs: {', '.join(_WM_ARCHS)}."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PATH_OR_REPO_ID",
        help=(
            "Path to a local LeRobotDataset directory OR a HuggingFace repo id "
            "(e.g. 'lerobot/pusht'). Required by all backends."
        ),
    )
    parser.add_argument(
        "--datasets",
        default=None,
        metavar="PATH1,PATH2,...",
        action="append",
        help=(
            "Train on MULTIPLE datasets. Comma-separated and/or repeatable "
            "(e.g. --datasets a,b  or  --datasets a --datasets b). Each entry "
            "is a local LeRobotDataset dir or an HF repo id. Takes precedence "
            "over --dataset. Policy archs forward the combined set to "
            "lerobot-train's MultiLeRobotDataset path (HF repo ids) or merge "
            "local dirs first; world-model archs require a single --dataset."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML config file for the selected backend. "
            "If omitted, defaults from lerobot-isaac-configs are used."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/run",
        metavar="PATH",
        help="Directory where checkpoints and logs are written. Default: %(default)s.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000,
        metavar="N",
        help="Total training steps (or world-model iterations). Default: %(default)s.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="Training batch size. Default: %(default)s.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="F",
        help="Learning rate. Default: %(default)s.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for reproducibility. Default: %(default)s.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Print resolved command and exit without dispatching to target. "
            "Useful for verifying arguments before a long training run."
        ),
    )
    parser.add_argument(
        "--cache_frames",
        action="store_true",
        help=(
            "Pre-decode every dataset row into RAM at train start, then serve "
            "all subsequent steps from memory. Removes the PNG-decode "
            "bottleneck on small datasets (~3-4x steps/s gain on PNG-heavy "
            "LeRobotDataset). See plans/2026-05-15-dataloader-gpu-decode-plan.md "
            "(approach A). Currently wired for policy archs only (smolvla, "
            "act, diffusion). Ignored by world-model backends."
        ),
    )
    parser.add_argument(
        "--cache_ram_gb",
        type=float,
        default=8.0,
        metavar="GB",
        help=(
            "Hard RAM ceiling for --cache_frames. The wrapper raises "
            "MemoryError mid-warmup if exceeded. Default: %(default)s GB."
        ),
    )
    parser.add_argument(
        "--successes_only",
        action="store_true",
        help=(
            "Train only on successful demonstrations. Reads the recorder's "
            "per-episode success sidecar (meta/episode_labels.json) from the "
            "local dataset root and forwards the successful episode indices to "
            "lerobot-train via --dataset.episodes. Policy archs only; requires a "
            "single local --dataset. No-op (with a warning) if the dataset is "
            "unlabelled, has no successes, or is an HF repo / multi-local set."
        ),
    )
    # --- LoRA / PEFT flags (Phase 1.4) ----------------------------------
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help=(
            "Wrap the policy with PEFT LoRA adapters at policy-construction "
            "time. Currently supported for --target_arch smolvla only. "
            "Other archs ignore this flag with a warning."
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        metavar="R",
        help="LoRA rank r. Common range: 4-32. Default: %(default)s.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        metavar="A",
        help=(
            "LoRA scaling factor alpha. Effective scale = alpha/r. "
            "Default: %(default)s (= 2*default_rank)."
        ),
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        metavar="F",
        help="Dropout on the LoRA path. Default: %(default)s.",
    )
    parser.add_argument(
        "--lora_target_modules",
        default="attn_qv",
        metavar="SPEC",
        help=(
            "LoRA target modules. Either a preset "
            "(attn_qv | attn_qkvo | expert_only) or a comma-separated list "
            "of layer-name suffixes (e.g. 'q_proj,v_proj'). "
            "Default: %(default)s."
        ),
    )
    # --- sheeprl exp selection (Plan2Explore / p2e_dv3) ------------------
    parser.add_argument(
        "--exp",
        default=None,
        metavar="EXP_NAME",
        help=(
            "sheeprl experiment config name passed as 'exp=<name>' to the "
            "hydra composition (dreamerv3 backend only). Defaults to "
            "'dreamer_v3'. Use 'p2e_dv3_exploration' for reward-free "
            "Plan2Explore intrinsic-reward pre-training, or "
            "'p2e_dv3_finetuning' to resume with extrinsic rewards. "
            "The env var LEROBOT_ISAAC_EXP is also consulted when this flag "
            "is not set. Ignored by policy backends."
        ),
    )
    parser.add_argument(
        "--exploration_ckpt",
        default=None,
        metavar="PATH",
        help=(
            "Absolute path to a p2e_dv3_exploration checkpoint directory. "
            "Required (or set via LEROBOT_ISAAC_EXPLORATION_CKPT env var) "
            "when --exp p2e_dv3_finetuning is used and "
            "'checkpoint.exploration_ckpt_path=' is not already present in "
            "the remainder. Forwarded to sheeprl as "
            "'checkpoint.exploration_ckpt_path=<path>'."
        ),
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        metavar="PATH",
        help=(
            "Resume a sheeprl world-model run from a checkpoint (dreamerv3 "
            "backend only). Forwarded to sheeprl as "
            "'checkpoint.resume_from=<path>' for ANY exp (native sheeprl "
            "resume). The env var LEROBOT_ISAAC_RESUME_FROM is also consulted "
            "when this flag is not set. Distinct from --exploration_ckpt, which "
            "is the Plan2Explore finetuning-only exploration weight path. "
            "Ignored by policy backends."
        ),
    )
    # --- world-model bridge conversion overrides (dreamerv3 backend) ------
    parser.add_argument(
        "--camera_key",
        default=None,
        metavar="KEY",
        help=(
            "Dataset image-observation key to convert (dreamerv3 backend "
            "only), e.g. 'observation.images.overhead'. Forwarded to the "
            "world-model bridge as image_keys=[<key>]. If omitted, the bridge "
            "auto-detects the image key(s). Ignored by policy backends."
        ),
    )
    parser.add_argument(
        "--state_keys",
        default=None,
        metavar="KEY1,KEY2,...",
        help=(
            "Comma-separated dataset state-observation keys to convert "
            "(dreamerv3 backend only), e.g. 'observation.state'. Forwarded to "
            "the world-model bridge as state_keys=[...]. If omitted, the bridge "
            "auto-detects the state key(s). Ignored by policy backends."
        ),
    )
    parser.add_argument(
        "--image_size",
        default=None,
        metavar="N | H,W",
        help=(
            "Override the world-model bridge image size (dreamerv3 backend "
            "only). Either a single int N giving an (N, N) square, or 'H,W'. "
            "If omitted, defaults to 64,64. Ignored by policy backends."
        ),
    )
    # Capture any extra args after '--' to forward to the backend
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help=(
            "Extra arguments forwarded verbatim to the backend "
            "(e.g. -- --policy.n_action_steps=100)."
        ),
    )
    return parser


def _resolve_dataset_list(args: argparse.Namespace) -> list[str]:
    """Flatten ``--datasets`` (repeatable + comma-separated) into one list.

    ``--datasets`` takes precedence over ``--dataset``. Returns ``[]`` when
    neither is given. Each ``--datasets`` occurrence may itself be a
    comma-separated string, so ``--datasets a,b --datasets c`` -> [a, b, c].
    """
    raw = getattr(args, "datasets", None)
    if raw:
        out: list[str] = []
        for chunk in raw:
            out.extend(d.strip() for d in str(chunk).split(",") if d.strip())
        return out
    if getattr(args, "dataset", None):
        return [args.dataset]
    return []


def _dispatch(args: argparse.Namespace) -> int:
    """Route to the correct backend module based on ``args.target_arch``.

    Returns
    -------
    int
        Exit code (0 on success, including dry-run).
    """
    # Normalise multi-dataset input once; backends read args.dataset_list.
    args.dataset_list = _resolve_dataset_list(args)

    # World-model backends accept exactly one dataset.
    if len(args.dataset_list) > 1 and args.target_arch in _WM_ARCHS:
        print(
            f"[lerobot-isaac-train] ERROR: --datasets with "
            f"{len(args.dataset_list)} entries is unsupported for "
            f"target_arch={args.target_arch!r}; world-model backends accept a "
            f"single --dataset. Merge first or pass one dataset.",
            file=sys.stderr,
        )
        return 2
    # smolvla-only guard: warn and clear use_lora for unsupported archs.
    if getattr(args, "use_lora", False) and args.target_arch != "smolvla":
        print(
            f"[lerobot-isaac-train] WARNING: LoRA is only wired for smolvla; "
            f"ignoring --use_lora for target_arch={args.target_arch!r}.",
            file=sys.stderr,
        )
        args.use_lora = False

    if args.dry_run:
        # Global dry_run summary — backends also handle their own dry_run output
        print(
            f"[dry_run] target_arch={args.target_arch} "
            f"dataset={args.dataset} "
            f"datasets={args.dataset_list} "
            f"output_dir={args.output_dir} "
            f"steps={args.steps} "
            f"batch_size={args.batch_size} "
            f"lr={args.lr} "
            f"seed={args.seed} "
            f"cache_frames={args.cache_frames} "
            f"cache_ram_gb={args.cache_ram_gb} "
            f"use_lora={args.use_lora} "
            f"lora_rank={args.lora_rank} "
            f"lora_alpha={args.lora_alpha} "
            f"lora_dropout={args.lora_dropout} "
            f"lora_target_modules={args.lora_target_modules}"
        )

    arch = args.target_arch

    if arch in _POLICY_ARCHS or arch in _WM_POLICY_ARCHS:
        # Plain policies AND lerobot 0.6.0 world-model policies both train via
        # the `lerobot-train` subprocess (policy_lerobot maps target_arch ->
        # --policy.type 1:1) and report pc_success.
        from lerobot_isaac_adapters.targets import policy_lerobot as backend
    elif arch == "dreamerv3":
        from lerobot_isaac_adapters.targets import wm_dreamerv3 as backend  # type: ignore[assignment]
    elif arch == "le_world_model":
        from lerobot_isaac_adapters.targets import wm_leworldmodel as backend  # type: ignore[assignment]
    else:
        # Should never reach here because argparse enforces choices
        raise ValueError(f"Unknown target_arch: {arch!r}")

    rc = backend.run(args)
    if not args.dry_run and rc != 0:
        print(
            f"\033[31mTraining failed (exit={rc}) — see stdout above\033[0m",
            file=sys.stderr,
        )
    return rc


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate training backend.

    This function is the ``console_scripts`` target for the ``lerobot-isaac-train``
    command. It is also callable directly from Python::

        from lerobot_isaac_adapters.train import main
        main(["--target_arch", "smolvla", "--dataset", "/tmp/ds"])
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = _dispatch(args)
    sys.exit(rc)


if __name__ == "__main__":
    main(sys.argv[1:])
