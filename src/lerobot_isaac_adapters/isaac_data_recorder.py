"""
isaac_data_recorder
===================

Records Isaac Lab rollout episodes to a LeRobotDataset Parquet file.

Both ``isaaclab`` and ``lerobot`` are soft-imported at call time, not at module
level, so this file can be imported in environments where neither is installed.

CLI
---
::

    python -m lerobot_isaac_adapters.isaac_data_recorder \\
        --env_id Isaac-SO101-PickPlace-v0 \\
        --output_dir datasets/dr_episodes \\
        --num_episodes 50 \\
        --seed 0

    # With a policy checkpoint:
    python -m lerobot_isaac_adapters.isaac_data_recorder \\
        --env_id Isaac-SO101-PickPlace-v0 \\
        --output_dir datasets/dr_episodes \\
        --num_episodes 50 \\
        --policy_checkpoint outputs/run/checkpoints/last.pt

Observation schema (LeRobotDataset v3.0)
-----------------------------------------
- ``observation.state``           — shape (12,): joint pos (6) + joint vel (6)
- ``observation.images.wrist``    — shape (H, W, 3) uint8, wrist camera
- ``observation.images.overhead`` — shape (H, W, 3) uint8, overhead camera
- ``action``                      — shape (6,): joint position targets (radians)
- ``episode_index``               — int64
- ``frame_index``                 — int64
- ``timestamp``                   — float32 (seconds since episode start)

Source tags
-----------
Episodes recorded by this module are tagged with ``source="dr"`` (domain
randomization) in the merged dataset meta. This tag is injected by
``lerobot_isaac_synthetic.merge_utilities.merge_datasets()``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Callable


# ---------------------------------------------------------------------------
# Feature schema (LeRobotDataset v3.0)
# ---------------------------------------------------------------------------

_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "joint_pos_0",
            "joint_pos_1",
            "joint_pos_2",
            "joint_pos_3",
            "joint_pos_4",
            "joint_pos_5",
            "joint_vel_0",
            "joint_vel_1",
            "joint_vel_2",
            "joint_vel_3",
            "joint_vel_4",
            "joint_vel_5",
        ],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.overhead": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
        ],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_episodes(
    env_id: str,
    output_dir: str | Path,
    num_episodes: int = 10,
    policy_fn: Callable | None = None,
    policy_checkpoint: str | None = None,
    seed: int = 42,
) -> Path:
    """Roll out episodes in an Isaac Lab env and write to LeRobotDataset Parquet.

    Parameters
    ----------
    env_id:
        Gym environment id registered by ``lerobot_isaac_env``,
        e.g. ``'Isaac-SO101-PickPlace-v0'``.
    output_dir:
        Directory where the LeRobotDataset Parquet files are written.
    num_episodes:
        Number of rollout episodes to record.
    policy_fn:
        Optional callable ``(obs_dict) -> action_array``.
        If ``None`` and ``policy_checkpoint`` is also ``None``, random actions
        are sampled from the environment's action space.
    policy_checkpoint:
        Optional path to a LeRobot policy checkpoint (``*.pt``).
        Ignored if ``policy_fn`` is provided directly.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Path
        Path to the created LeRobotDataset root directory.

    Raises
    ------
    ImportError
        If ``isaaclab`` (gymnasium + Isaac Lab) or ``lerobot`` are not installed.
    """
    # Soft-import Isaac Lab / gymnasium
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "gymnasium is required for Isaac Lab rollouts. "
            "Install Isaac Lab per https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/ "
            "then activate its conda environment."
        ) from exc

    # Soft-import lerobot dataset API
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "lerobot is required to write LeRobotDataset Parquet files. "
            "Install: pip install lerobot"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy from checkpoint if requested
    if policy_fn is None and policy_checkpoint is not None:
        policy_fn = _load_policy_checkpoint(policy_checkpoint)

    # Create env
    env = gym.make(env_id, headless=True, num_envs=1, seed=seed)
    env.reset(seed=seed)

    # Create LeRobotDataset
    repo_id = f"local/{env_id.lower().replace('-', '_')}_dr"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(output_dir),
        fps=30,
        features=_FEATURES,
    )

    import numpy as np

    for ep_idx in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_idx = 0

        while not done:
            if policy_fn is not None:
                action = policy_fn(obs)
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            frame = {
                "observation.state": np.array(obs["joint_pos_vel"], dtype="float32"),
                "observation.images.wrist": np.array(
                    obs["wrist_cam_rgb"], dtype="uint8"
                ),
                "observation.images.overhead": np.array(
                    obs["overhead_cam_rgb"], dtype="uint8"
                ),
                "action": np.array(action, dtype="float32"),
            }
            dataset.add_frame(frame)

            obs = next_obs
            step_idx += 1

        dataset.save_episode()
        print(
            f"[isaac_data_recorder] episode {ep_idx + 1}/{num_episodes} "
            f"({step_idx} steps)"
        )

    env.close()

    # Consolidate — writes meta/info.json, meta/stats.json, meta/episodes.parquet
    dataset.consolidate()
    print(f"[isaac_data_recorder] Dataset written to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_policy_checkpoint(checkpoint_path: str) -> Callable:
    """Load a LeRobot policy checkpoint and return an inference callable.

    The returned callable accepts an obs dict and returns a numpy action array.
    Soft-imports lerobot and torch.
    """
    try:
        import torch
        from lerobot.common.policies.factory import make_policy
    except ImportError as exc:
        raise ImportError(
            "torch and lerobot are required to load a policy checkpoint. "
            "Install: pip install lerobot torch"
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy = make_policy(checkpoint["policy_cfg"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    def _infer(obs_dict):
        import numpy as np

        with torch.no_grad():
            obs_tensors = {
                k: torch.from_numpy(np.array(v)).float().unsqueeze(0)
                for k, v in obs_dict.items()
            }
            action = policy.select_action(obs_tensors)
        return action.squeeze(0).numpy()

    return _infer


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list | None = None) -> None:
    """CLI wrapper for ``record_episodes``.

    Example::

        python -m lerobot_isaac_adapters.isaac_data_recorder \\
            --env_id Isaac-SO101-PickPlace-v0 \\
            --output_dir datasets/dr_episodes \\
            --num_episodes 50 \\
            --seed 0
    """
    parser = argparse.ArgumentParser(
        prog="isaac-data-recorder",
        description=(
            "Record Isaac Lab rollout episodes to a LeRobotDataset Parquet. "
            "Requires isaaclab and lerobot to be installed."
        ),
    )
    parser.add_argument(
        "--env_id",
        required=True,
        help="Isaac Lab gym env id, e.g. 'Isaac-SO101-PickPlace-v0'.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the LeRobotDataset.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to record. Default: %(default)s.",
    )
    parser.add_argument(
        "--policy_checkpoint",
        default=None,
        help="Path to LeRobot policy checkpoint (optional; random actions if omitted).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: %(default)s.",
    )
    args = parser.parse_args(argv)
    record_episodes(
        env_id=args.env_id,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        policy_checkpoint=args.policy_checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
