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
- ``dreamerv3``      -> ``targets.wm_dreamerv3.run()``
- ``le_world_model`` -> ``targets.wm_leworldmodel.run()``

All backends accept the same ``argparse.Namespace`` argument and emit metrics
to stdout via ``metric_extractor.emit()``.

Extra arguments (after ``--``) are passed through to the backend unchanged via
``args.remainder``.
"""

from __future__ import annotations

import argparse
import sys

_POLICY_ARCHS = ("smolvla", "act", "diffusion")
_WM_ARCHS = ("dreamerv3", "le_world_model")
_ALL_ARCHS = _POLICY_ARCHS + _WM_ARCHS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-isaac-train",
        description=(
            "Unified training entrypoint for LeRobot + Isaac Lab.\n"
            "Dispatches to policy (smolvla/act/diffusion) or world-model "
            "(dreamerv3/le_world_model) backends based on --target_arch."
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
            f"World-model archs: {', '.join(_WM_ARCHS)}."
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


def _dispatch(args: argparse.Namespace) -> int:
    """Route to the correct backend module based on ``args.target_arch``.

    Returns
    -------
    int
        Exit code (0 on success, including dry-run).
    """
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

    if arch in _POLICY_ARCHS:
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
