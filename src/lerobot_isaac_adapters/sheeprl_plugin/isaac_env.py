"""Isaac Lab SO-101 pick-place env wrapped as a sheeprl-compatible gym.Env.

Phase C of `plans/2026-05-23-wm-isaac-env-plan.md`. Replaces the
`HDF5ReplayEnv` in the DreamerV3 training stack:

  * `HDF5ReplayEnv.step()` ignores actions + returns reward=0.0 — the
    actor head learns nothing useful. Result: WM-actor cannot drive a
    real robot.
  * `IsaacSO101Env.step()` runs a real Isaac Lab physics tick + emits a
    shaped pick-place reward, so the DreamerV3 actor receives causal
    feedback + a task signal.

The env reuses `lerobot_isaac_env.so101_articulation` + the existing
pick-place task cfg from `lerobot-isaac-env`. Soft-imports throughout so
the module remains importable in any env (sheeprl-only, dashboard-only)
— Isaac Lab is only hauled in inside `IsaacSO101Env.__init__`.

Status: SKELETON. 5 TODO(C1.<n>) markers track what still needs bodies.
"""
from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# Mandatory warm-up tick count after sim.reset() before camera obs are
# valid. See plans/2026-05-23-sim-deploy-pipeline.md §pitfalls (this is
# inherited from isaac-auto-scene's CLAUDE.md notes).
WARM_UP_FRAMES = 30


class IsaacSO101Env(gym.Env):
    """SO-101 pick-place env wrapped for sheeprl + DreamerV3.

    Observation:
        dict with keys
            "rgb":   uint8 (3, H, W) — wrist or overhead camera.
            "state": float32 (6,)    — joint positions.

    Action: float32 (6,) joint position deltas in [-1, 1].

    Reward (per `IsaacSO101Env._compute_reward`):
        0.1 * exp(-‖gripper - object‖)
      + 0.5  if gripper_closed_around_object
      + 1.0  * object_z / basket_height_target
      + 5.0  if object_in_basket
      - 0.01 if self_collision
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: str = "pickplace",
        num_envs: int = 1,
        image_size: int = 64,
        rate_hz: float = 30.0,
        max_episode_steps: int = 600,
        headless: bool = True,
        device: str = "cuda",
        seed: int | None = None,
        dr_config: str | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.num_envs = num_envs
        self.image_size = image_size
        self.rate_hz = rate_hz
        self.max_episode_steps = max_episode_steps
        self.headless = headless
        self.device = device
        self.dr_config = dr_config

        # Spaces. Construct here so they're available even before the
        # actual Isaac Sim env spins up — sheeprl's space-inspection
        # codepath touches them at make_env time.
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, image_size, image_size),
                    dtype=np.uint8,
                ),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.reward_range = (-np.inf, np.inf)

        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._isaac_env: Any = None  # populated by _boot()
        self._booted = False

    # ------------------------------------------------------------------ #
    # gym.Env API
    # ------------------------------------------------------------------ #

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if not self._booted:
            self._boot()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        # TODO(C1.1): call self._isaac_env.reset() and read the first obs.
        # For now return zero-init buffers so sheeprl's space-inspection
        # path doesn't crash. Replace with real obs once the Isaac Lab
        # backend is wired.
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self._booted:
            raise RuntimeError("call reset() before step()")
        self._t += 1
        # TODO(C1.2): scale + clip action, call self._isaac_env.step(action),
        # read obs/reward/done from the Isaac Lab return.
        obs = self._get_obs()
        reward = self._compute_reward(obs, {})
        terminated = False  # TODO(C1.3): self._success_criterion(obs)
        truncated = self._t >= self.max_episode_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        obs = self._get_obs()
        # Return HWC for sheeprl's RecordVideoV0 wrapper.
        return obs["rgb"].transpose(1, 2, 0)

    def close(self) -> None:
        if self._isaac_env is not None:
            try:
                self._isaac_env.close()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "IsaacSO101Env.close raised; SimulationApp.close "
                    "deadlocks on Isaac Sim 6.0 — caller should os._exit(0)."
                )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _boot(self) -> None:
        """Spin up Isaac Lab. Idempotent — safe to call from reset()."""
        if self._booted:
            return
        # TODO(C1.4): boot Isaac Lab + instantiate the pick-place task env.
        # Reuse the existing scaffold:
        #
        #     from isaaclab.app import AppLauncher
        #     launcher = AppLauncher(headless=self.headless, enable_cameras=True)
        #     # ... import Isaac Lab modules AFTER AppLauncher ...
        #     from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg
        #     from lerobot_isaac_env.tasks.pickplace import PickAndPlaceEnvCfg
        #     cfg = PickAndPlaceEnvCfg()
        #     cfg.scene.num_envs = self.num_envs
        #     # cfg.image_size = self.image_size  # ensure cameras emit 64×64
        #     from isaaclab.envs import ManagerBasedRLEnv  # or similar
        #     self._isaac_env = ManagerBasedRLEnv(cfg=cfg)
        #     for _ in range(WARM_UP_FRAMES):
        #         self._isaac_env.sim.step(render=True)
        #
        raise NotImplementedError(
            "TODO(C1.4): boot Isaac Lab + pick-place env. See "
            "plans/2026-05-23-wm-isaac-env-plan.md §Phase C1."
        )
        self._booted = True

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Read camera + joint state from the live env.

        Skeleton returns zero buffers so sheeprl's `make_env` space
        inspection succeeds before the real backend is wired.
        """
        # TODO(C1.5): read RGB camera + joint positions from
        # self._isaac_env. Until then return zeros so the type contract
        # holds for sheeprl's space-inspection codepath.
        return {
            "rgb": np.zeros(
                (3, self.image_size, self.image_size), dtype=np.uint8
            ),
            "state": np.zeros(6, dtype=np.float32),
        }

    def _compute_reward(
        self, obs: dict[str, np.ndarray], info: dict[str, Any]
    ) -> float:
        """Shaped pick-place reward.

        Phase C3 tunes the weights. Skeleton returns 0 — replace with the
        weighted sum from the plan once obs fields are real.
        """
        # Plan §Reward function:
        #     r = 0
        #     r += 0.1 * exp(-‖gripper - object‖)
        #     r += 0.5 if gripper_closed_around_object
        #     r += 1.0 * object_z / basket_height_target
        #     r += 5.0 if object_in_basket
        #     r -= 0.01 if self_collision
        return 0.0


# --------------------------------------------------------------------------- #
# Hydra factory — sheeprl loads this via `env._target_`
# --------------------------------------------------------------------------- #


def get_isaac_env(
    task: str = "pickplace",
    image_size: int = 64,
    num_envs: int = 1,
    rate_hz: float = 30.0,
    max_episode_steps: int = 600,
    headless: bool = True,
    device: str = "cuda",
    seed: int | None = None,
    dr_config: str | None = None,
) -> IsaacSO101Env:
    """Hydra-friendly factory wrapping :class:`IsaacSO101Env`.

    Drop-in replacement for ``hdf5_env.get_hdf5_env``. Use via:

        env=isaac_so101   # in `configs/env/isaac_so101.yaml`
    """
    return IsaacSO101Env(
        task=task,
        num_envs=num_envs,
        image_size=image_size,
        rate_hz=rate_hz,
        max_episode_steps=max_episode_steps,
        headless=headless,
        device=device,
        seed=seed,
        dr_config=dr_config,
    )
