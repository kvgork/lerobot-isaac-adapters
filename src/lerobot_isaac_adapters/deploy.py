"""deploy.py — Run a trained policy on a real SO-101 arm.

Loads a checkpoint produced by ``lerobot_isaac_adapters.train`` (or vanilla
``lerobot-train``), connects to the physical SO-101 follower arm via the
``lerobot.robots.so_follower.SO101Follower`` driver, and steps the control
loop at a fixed rate.

Safety design
-------------
The SO101Follower already enforces ``max_relative_target`` per-joint at the
DYNAMIXEL layer — actions that would move any joint more than that delta in
one step are clipped server-side. This module additionally:

* runs in ``--dry-run`` mode by default — connects, reads obs, prints
  predicted actions, but does NOT send commands.
* catches SIGINT / SIGTERM, disables torque, and exits cleanly.
* enforces an outer ``--duration-s`` wall-clock cap.
* warns when the same action repeats N steps (motors stuck or policy NaN).
* if ``--home-on-exit`` is set, sends a zero-position command before
  disconnecting.

Use ``--dry-run`` first to validate the policy + obs preprocessor against
the live observation stream BEFORE letting it move the arm.

CLI
---
::

    lerobot-isaac-deploy \\
        --policy-path outputs/.../checkpoints/last/pretrained_model \\
        --port /dev/ttyACM0 \\
        --rate-hz 30 --duration-s 60 \\
        --dataset-root datasets/kvgork/so101-pickplace1 \\
        --camera d435_rgb=/dev/video0,640,480 \\
        --max-relative-target 5.0 \\
        --home-on-exit
        # add --execute to leave dry-run mode

Console script entry: ``lerobot-isaac-deploy``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lerobot-isaac-deploy",
        description=(
            "Run a trained policy on a real SO-101 arm. Default is DRY-RUN — "
            "policy is queried but no motor commands are sent. Add --execute "
            "to enable real motor writes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--policy-path",
        required=True,
        help=(
            "Path to a `pretrained_model/` directory (containing model.safetensors, "
            "policy_processor.json, etc). lerobot 0.5+ layout."
        ),
    )
    p.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Serial device for the SO-101 follower (U2D2 or built-in adapter).",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "Optional LeRobotDataset root used to infer obs/action feature "
            "shapes when the policy config doesn't carry them. Required when "
            "the checkpoint does not embed `policy_features` (most older lerobot "
            "checkpoints — pass the same dataset used for training)."
        ),
    )
    p.add_argument(
        "--camera",
        action="append",
        default=[],
        help=(
            "Camera spec `name=device,W,H`. Repeatable. Example: "
            "`d435_rgb=/dev/video0,640,480`. If your policy expects images "
            "under `observation.images.<name>`, the name MUST match."
        ),
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=30.0,
        help="Control loop frequency (Hz). SO-101 captures at 30 Hz by default.",
    )
    p.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Hard wall-clock cap. Loop exits when reached.",
    )
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help=(
            "Per-joint max delta per timestep (degrees or normalized units, "
            "matching --use-degrees). Server-side clip; smaller = safer."
        ),
    )
    p.add_argument(
        "--use-degrees",
        action="store_true",
        help=(
            "Bus reads/writes in degrees (matches LeRobotDataset recordings "
            "where observation.state is in degrees)."
        ),
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Enable real motor writes. Default is dry-run (predict-only). "
            "ALWAYS run with --dry-run (default) first to verify the policy "
            "produces sane actions before commanding the real arm."
        ),
    )
    p.add_argument(
        "--home-on-exit",
        action="store_true",
        help=(
            "Ramp arm to zero-position before disconnecting (rate-limited by "
            "--max-step-deg). Disable if homing through zero would collide "
            "with a workspace object."
        ),
    )
    p.add_argument(
        "--max-step-deg",
        type=float,
        default=1.0,
        help=(
            "Maximum per-step joint motion in degrees (arm joints 0-4). "
            "Hard-clamps actor output before send_action. Independent of "
            "server-side max_relative_target. 1.0 = tight, 3.0 = loose."
        ),
    )
    p.add_argument(
        "--require-real-ckpt",
        action="store_true",
        help=(
            "Refuse motor writes against any checkpoint with a "
            "synthetic_marker.json file. Also honored via env var "
            "LI_DEPLOY_REQUIRE_REAL_CKPT=1. Defense-in-depth against "
            "running test fixtures on real hardware."
        ),
    )
    p.add_argument(
        "--repeat-warn-steps",
        type=int,
        default=30,
        help=(
            "Warn if the predicted action stays within ε of the previous "
            "action for this many consecutive steps (≈ 1 s at 30 Hz). "
            "Suggests policy stuck or NaN."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Torch seed for any non-deterministic policy layers.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Log every action."
    )
    return p


# ---------------------------------------------------------------------------
# Policy loader (mirrors scripts/_open_loop_eval.py)
# ---------------------------------------------------------------------------

@dataclass
class _LoadedPolicy:
    policy: Any
    device: str


def _load_policy(policy_path: Path, dataset_root: Path | None, seed: int) -> _LoadedPolicy:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.pretrained_path = Path(policy_path)

    ds_meta = None
    if dataset_root is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        parts = Path(dataset_root).resolve().parts
        repo_id = (
            "/".join(parts[-2:]) if len(parts) >= 2 else Path(dataset_root).name
        )
        ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))

    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.to(device)
    policy.eval()
    return _LoadedPolicy(policy=policy, device=device)


# ---------------------------------------------------------------------------
# Robot setup
# ---------------------------------------------------------------------------

def _parse_camera_specs(specs: list[str]) -> dict[str, Any]:
    """Parse `name=device,W,H` entries into a SO101FollowerConfig.cameras dict."""
    if not specs:
        return {}
    from lerobot.cameras.opencv import OpenCVCameraConfig

    out: dict[str, Any] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--camera spec must be `name=device,W,H`: {spec!r}")
        name, rhs = spec.split("=", 1)
        bits = [b.strip() for b in rhs.split(",")]
        if len(bits) < 3:
            raise ValueError(f"--camera spec needs device,W,H: {spec!r}")
        device = bits[0]
        try:
            w = int(bits[1])
            h = int(bits[2])
        except ValueError as exc:
            raise ValueError(f"camera W,H must be ints: {spec!r}") from exc
        out[name.strip()] = OpenCVCameraConfig(
            index_or_path=device, width=w, height=h, fps=30
        )
    return out


def _build_robot(args: argparse.Namespace) -> Any:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(
        port=args.port,
        max_relative_target=float(args.max_relative_target),
        use_degrees=bool(args.use_degrees),
        cameras=_parse_camera_specs(args.camera),
    )
    return SO101Follower(cfg)


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------

def _obs_to_policy_input(obs: dict, device: str) -> dict:
    """Convert SO101Follower.get_observation() output to a batched policy input.

    SO101Follower returns `{"<motor>.pos": float, "<camera>": ndarray, ...}`.
    LeRobot policies expect `{"observation.state": (1, D), "observation.images.<key>": (1, C, H, W), ...}`.
    """
    import numpy as np
    import torch

    motor_keys = [k for k in obs if k.endswith(".pos")]
    image_keys = [k for k in obs if not k.endswith(".pos")]

    state = np.array([float(obs[k]) for k in sorted(motor_keys)], dtype=np.float32)
    out: dict[str, torch.Tensor] = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
    }
    for k in image_keys:
        arr = np.asarray(obs[k])
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device).float() / 255.0
        out[f"observation.images.{k}"] = tensor
    return out


def _action_to_robot_dict(action: Any, motor_names: list[str]) -> dict[str, float]:
    """Convert policy output tensor (1, A) into the dict robot.send_action wants.

    .. deprecated::
        Prefer ``_compute_safe_targets()`` which applies the [-1, 1] clip,
        joint-limit clamp, and per-step bound via lerobot_isaac_deploy.
        Kept only for dry-run printing where motor writes are inhibited.
    """
    import numpy as np

    arr = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[0] != len(motor_names):
        raise ValueError(
            f"Policy emitted {arr.shape[0]} action dims but robot has "
            f"{len(motor_names)} motors: {motor_names!r}"
        )
    return {f"{m}.pos": float(arr[i]) for i, m in enumerate(motor_names)}


def _read_current_jp(obs: dict, joint_order: list[str]) -> "np.ndarray":  # noqa: F821
    """Extract current joint positions from a SO101Follower observation dict.

    Returns float32 array shape (6,) in canonical SO101_JOINT_NAMES order.
    Raises ValueError on missing keys or non-finite values — caller should
    treat as a hard stop and not send an action this tick.
    """
    import numpy as np

    vals = []
    for name in joint_order:
        key = f"{name}.pos"
        if key not in obs:
            raise ValueError(f"obs missing motor key: {key}")
        v = float(obs[key])
        if not np.isfinite(v):
            raise ValueError(f"non-finite current_jp[{name}]: {v}")
        vals.append(v)
    return np.asarray(vals, dtype=np.float32)


def _compute_safe_targets(
    action: Any,
    current_jp: "np.ndarray",  # noqa: F821
    max_step_deg: float,
) -> dict[str, float]:
    """Safety-clamped action -> send_action dict.

    Applies the lerobot_isaac_deploy.arm_motor_writer two-layer clamp:
        1. action clipped to [-1, 1] (bounds per-step motion regardless of
           upstream actor pathology — raw logits, NaN, unconverged head).
        2. target = current + clipped_action * max_step_deg clamped to the
           hardcoded joint-limit floor (includes elbow_flex >= -10° table
           avoidance).
    """
    import numpy as np

    from lerobot_isaac_deploy.arm_motor_writer import (
        SO101_JOINT_NAMES,
        compute_targets,
    )

    arr = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[0] != 6:
        raise ValueError(
            f"policy emitted {arr.shape[0]} action dims; SO-101 expects 6 "
            f"(joints {SO101_JOINT_NAMES})"
        )
    targets = compute_targets(
        current_jp=current_jp,
        action=arr,
        max_step_deg=max_step_deg,
    )
    return {f"{name}.pos": float(targets[i]) for i, name in enumerate(SO101_JOINT_NAMES)}


def _check_real_ckpt_gate(policy_path: Path, require_real: bool) -> None:
    """Refuse motor writes against synthetic checkpoint fixtures.

    Mirrors the gate in src/lerobot-isaac-deploy/session.py — synthetic
    checkpoints carry a ``synthetic_marker.json`` file at the checkpoint
    root. They exist so the test suite can run without GPU/heavy deps; they
    MUST NOT drive real motors.

    Raises RuntimeError if the gate is active (CLI flag or env var) AND
    the marker is present.
    """
    import os

    active = require_real or os.environ.get("LI_DEPLOY_REQUIRE_REAL_CKPT", "") == "1"
    if not active:
        return
    marker = Path(policy_path).parent / "synthetic_marker.json"
    if not marker.exists():
        marker = Path(policy_path) / "synthetic_marker.json"
    if marker.exists():
        raise RuntimeError(
            "refusing motor writes against synthetic checkpoint at "
            f"{policy_path} (marker: {marker}). Unset --require-real-ckpt / "
            "LI_DEPLOY_REQUIRE_REAL_CKPT=1 to override (dangerous)."
        )


_stop = {"flag": False}


def _install_signal_handlers() -> None:
    def _h(_sig: int, _frame: Any) -> None:
        _stop["flag"] = True

    signal.signal(signal.SIGINT, _h)
    signal.signal(signal.SIGTERM, _h)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    logger.info(
        "deploy: policy=%s port=%s rate=%.1fHz duration=%.1fs dry_run=%s",
        args.policy_path,
        args.port,
        args.rate_hz,
        args.duration_s,
        not args.execute,
    )

    # Safety gate: refuse motor writes against synthetic-checkpoint fixtures
    # when --require-real-ckpt or LI_DEPLOY_REQUIRE_REAL_CKPT=1 is active.
    # Only enforced when --execute is on; dry-run is always allowed.
    if args.execute:
        try:
            _check_real_ckpt_gate(Path(args.policy_path), args.require_real_ckpt)
        except RuntimeError as exc:
            logger.error("synthetic-ckpt gate: %s", exc)
            return 9

    try:
        loaded = _load_policy(Path(args.policy_path), Path(args.dataset_root) if args.dataset_root else None, args.seed)
    except Exception as exc:  # noqa: BLE001
        logger.error("policy load failed: %s", exc)
        return 2

    try:
        robot = _build_robot(args)
    except Exception as exc:  # noqa: BLE001
        logger.error("robot setup failed: %s", exc)
        return 3

    _install_signal_handlers()

    try:
        robot.connect()
    except Exception as exc:  # noqa: BLE001
        logger.error("robot.connect() failed — is the arm plugged in? %s", exc)
        return 4

    motor_names = [m for m in robot.bus.motors]
    # Use the canonical SO101 joint order for safety-clamped targets — the
    # bus.motors iteration order is not guaranteed to match.
    from lerobot_isaac_deploy.arm_motor_writer import SO101_JOINT_NAMES

    dt = 1.0 / args.rate_hz
    deadline = time.monotonic() + args.duration_s

    repeat_count = 0
    last_action_dict: dict[str, float] | None = None
    n_steps = 0
    rc = 0
    try:
        while not _stop["flag"] and time.monotonic() < deadline:
            step_start = time.monotonic()
            try:
                obs = robot.get_observation()
            except Exception as exc:  # noqa: BLE001
                logger.error("get_observation failed: %s", exc)
                rc = 5
                break

            # Validate current joint state BEFORE inference. Bad sensor reads
            # (NaN / missing key / implausible value) -> hard stop, do not
            # propagate to a motor write.
            try:
                current_jp = _read_current_jp(obs, SO101_JOINT_NAMES)
            except ValueError as exc:
                logger.error("current_jp validation failed: %s — skipping step", exc)
                # Skip this tick rather than send an action against bad state.
                slack = dt - (time.monotonic() - step_start)
                if slack > 0:
                    time.sleep(slack)
                continue

            try:
                import torch

                with torch.no_grad():
                    policy_input = _obs_to_policy_input(obs, loaded.device)
                    action = loaded.policy.select_action(policy_input)
                # Safety-clamped target dict: [-1,1] action clip + joint
                # limits intersected with elbow-floor preservation, applied
                # via lerobot_isaac_deploy.arm_motor_writer.compute_targets.
                action_dict = _compute_safe_targets(
                    action, current_jp, max_step_deg=args.max_step_deg
                )
            except ValueError as exc:
                # Non-finite action from compute_targets -> hard stop.
                logger.error("safe-targets rejected action: %s", exc)
                rc = 6
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("policy inference failed: %s", exc)
                rc = 6
                break

            if last_action_dict is not None:
                if all(
                    abs(action_dict[k] - last_action_dict[k]) < 1e-4
                    for k in action_dict
                ):
                    repeat_count += 1
                    if repeat_count == args.repeat_warn_steps:
                        logger.warning(
                            "policy action unchanged for %d consecutive steps — "
                            "policy may be stuck or returning NaN; consider e-stop",
                            repeat_count,
                        )
                else:
                    repeat_count = 0
            last_action_dict = action_dict

            if args.verbose:
                logger.info(
                    "step %d action=%s",
                    n_steps,
                    {k: round(v, 3) for k, v in action_dict.items()},
                )

            if args.execute:
                try:
                    robot.send_action(action_dict)
                except Exception as exc:  # noqa: BLE001
                    logger.error("send_action failed: %s", exc)
                    rc = 7
                    break
            n_steps += 1

            slack = dt - (time.monotonic() - step_start)
            if slack > 0:
                time.sleep(slack)
    finally:
        if args.home_on_exit and args.execute:
            try:
                logger.info("ramped home on exit (max %.2f deg/step)", args.max_step_deg)
                from lerobot_isaac_deploy.arm_motor_writer import ramped_home

                # Read final current_jp; if obs fails or is invalid, fall
                # back to a single-shot zero target (cannot ramp without a
                # known start).
                try:
                    final_obs = robot.get_observation()
                    final_jp = _read_current_jp(final_obs, SO101_JOINT_NAMES)
                    ramped_home(
                        robot,
                        current_jp=final_jp,
                        max_step_deg=args.max_step_deg,
                        rate_hz=args.rate_hz,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "ramped_home read failed (%s) — falling back to "
                        "single-shot zero (server-side max_relative_target "
                        "is the only remaining clamp)",
                        exc,
                    )
                    home = {f"{m}.pos": 0.0 for m in motor_names}
                    robot.send_action(home)
                    time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("home-on-exit failed: %s", exc)
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("robot.disconnect() failed: %s", exc)

    logger.info(
        "deploy: %d steps in %.1fs (rc=%d, dry_run=%s)",
        n_steps,
        args.duration_s,
        rc,
        not args.execute,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
