"""Isaac Lab SO-101 pick-place env wrapped as a sheeprl-compatible gym.Env.

Phase C of `plans/2026-05-23-wm-isaac-env-plan.md`. Replaces the
`HDF5ReplayEnv` in the DreamerV3 training stack:

  * `HDF5ReplayEnv.step()` ignores actions + returns reward=0.0 — the
    actor head learns nothing useful. Result: WM-actor cannot drive a
    real robot.
  * `IsaacSO101Env.step()` runs a real Isaac Lab physics tick + emits
    the shaped pick-place reward from the env's own RewardManager
    (`lerobot_isaac_env.rewards`), so the DreamerV3 actor receives
    causal feedback + a task signal.

Wraps the existing `lerobot_isaac_env.make_env(...)` factory — no need
to re-author the SO-101 scene/articulation/observation/reward managers
(they live in the lerobot-isaac-env sibling). This module is the THIN
gym.Env adapter that:

  1. Boots ManagerBasedRLEnv via `make_env("pick_and_place", num_envs=1)`.
  2. Translates batched (num_envs, ...) tensors → single-env (...,) numpy
     arrays sheeprl expects.
  3. Exposes the canonical sheeprl obs key shape: `{"rgb": (3, H, W),
     "state": (6,)}` by default; expands `state` to (13,) when
     `LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1`.

Soft-imports throughout — module remains importable in any env
(sheeprl-only, dashboard-only). Isaac Lab is only loaded inside
`IsaacSO101Env._boot()`.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# Mandatory warm-up tick count after sim.reset() before camera obs are
# valid. Inherited from isaac-auto-scene's pitfall list.
WARM_UP_FRAMES = 30

# Module-level singleton: the underlying ManagerBasedRLEnv. Isaac Lab's
# SimulationContext is a process-wide singleton, so multiple
# IsaacSO101Env instances (e.g. sheeprl's train + test envs) MUST share
# one backing env. Without this, the second instance's _boot() tries to
# create a second SimulationContext and gets
# `RuntimeError: Simulation context already exists`.
_GLOBAL_BACKING_ISAAC_ENV: Any = None

# Default obs key set the wrapper exposes to sheeprl. The Isaac Lab env's
# `policy` ObservationGroup must include a `joint_pos`-style term (mapped
# to `state`) AND a camera term (mapped to `rgb`). Camera wiring lives in
# lerobot-isaac-env's `wrist_camera_rgb` / `overhead_camera_rgb` — currently
# scaffolded with NotImplementedError; the wrapper detects that and falls
# back to zero RGB until those land. See CLAUDE.md §"Camera observation
# wiring" in the training workspace.
DEFAULT_STATE_KEY = "joint_pos"
DEFAULT_CAMERA_KEY = "wrist_camera_rgb"

# Opt-in object_pose actor obs — diagnostic for the 2026-05-24 sweep where
# Grads/actor → 0 because the actor had no object-location signal.
# When enabled, state_dim expands from 6 to 13 (joint_pos[6] + object_pose[7]).
_INCLUDE_OBJECT_POSE = os.environ.get("LEROBOT_ISAAC_INCLUDE_OBJECT_POSE", "0") not in (
    "0",
    "",
    "false",
    "False",
)
_STATE_DIM_BASE = 6  # joint_pos (6-DOF)
_STATE_DIM_OBJECT_POSE = 7  # pos[3] + quat[4]


class IsaacSO101Env(gym.Env):
    """SO-101 pick-place env wrapped for sheeprl + DreamerV3.

    Observation:
        dict with keys
            "rgb":   uint8 (3, H, W) — wrist camera, falls back to zeros
                                       until lerobot-isaac-env camera term
                                       wiring lands.
            "state": float32 (6,)    — joint positions (default).
                     float32 (13,)   — joint_pos[6] + object_pose[7] when
                                       LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1.

    Action: float32 (6,) — joint position targets in [-1, 1] (env's
            JointPositionActionCfg scales these internally).

    Reward: passthrough from `ManagerBasedRLEnv.step()[1]`, which
            aggregates the terms wired in
            `lerobot_isaac_env.rewards` (`success_reward`,
            `action_l2_penalty`, `joint_vel_penalty`).
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
        state_key: str = DEFAULT_STATE_KEY,
        camera_key: str = DEFAULT_CAMERA_KEY,
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
        self.state_key = state_key
        self.camera_key = camera_key

        # Compute state dimension based on env-var flag.
        state_dim = _STATE_DIM_BASE + (_STATE_DIM_OBJECT_POSE if _INCLUDE_OBJECT_POSE else 0)
        self._state_dim = state_dim

        # Spaces declared up-front so sheeprl's make_env() space-inspection
        # codepath succeeds without booting Isaac Lab.
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, image_size, image_size),
                    dtype=np.uint8,
                ),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.reward_range = (-np.inf, np.inf)

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._isaac_env: Any = None  # populated by _boot()
        self._app: Any = None        # SimulationApp handle
        self._booted = False
        self._has_camera_term = False  # set by _boot() probe

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
        # ManagerBasedRLEnv.reset returns (obs_dict, info_dict). obs_dict
        # is keyed by ObservationGroup name; we use "policy".
        raw_obs, raw_info = self._isaac_env.reset(seed=seed)
        return self._translate_obs(raw_obs), self._scalar_info(raw_info)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self._booted:
            raise RuntimeError("call reset() before step()")
        self._t += 1
        # ManagerBasedRLEnv expects action shape (num_envs, action_dim).
        # We're single-env → add batch dim; cast to torch on device.
        action_t = self._to_torch(action).view(self.num_envs, -1)
        raw_obs, raw_reward, raw_term, raw_trunc, raw_info = self._isaac_env.step(action_t)
        obs = self._translate_obs(raw_obs)
        reward = float(self._scalar(raw_reward))
        terminated = bool(self._scalar(raw_term))
        # Isaac Lab tracks its own truncation; combine with the wrapper's
        # max_episode_steps cap so sheeprl's done-handling is correct.
        truncated = bool(self._scalar(raw_trunc)) or (self._t >= self.max_episode_steps)
        return obs, reward, terminated, truncated, self._scalar_info(raw_info)

    def render(self) -> np.ndarray:
        # Return HWC for sheeprl's RecordVideoV0 wrapper.
        return self._last_rgb_hwc.copy() if hasattr(self, "_last_rgb_hwc") else (
            np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        )

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
    # boot
    # ------------------------------------------------------------------ #

    def _boot(self) -> None:
        """Spin up Isaac Lab + the SO-101 pick-place env. Idempotent.

        AppLauncher MUST run BEFORE any `isaaclab.*` import — the
        managers import `omni.kit.app` at module-load time, which only
        exists once SimulationApp is alive. Failing to do this gives
        `ModuleNotFoundError: omni.kit.app`. Same recipe as Isaac Lab's
        own example scripts.
        """
        if self._booted:
            return

        # 1. Boot SimulationApp via AppLauncher FIRST — unless the caller
        #    already booted it (e.g. scripts/_wm_isaac_entry.py does this
        #    to claim libgobject before sheeprl imports). AppLauncher is
        #    NOT a singleton — calling it twice hangs waiting for kit
        #    extension reload. Probe `omni.kit.app` for an existing app.
        existing_app = None
        try:
            import omni.kit.app as _kit_app  # type: ignore[import]
            existing_app = _kit_app.get_app()
        except Exception:  # noqa: BLE001
            existing_app = None
        if existing_app is not None and getattr(existing_app, "is_running", lambda: False)():
            logger.info("SimulationApp already alive — skipping AppLauncher")
            self._app = existing_app
        else:
            try:
                from isaaclab.app import AppLauncher  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "Isaac Lab (isaaclab.app.AppLauncher) is required. "
                    "Run `pixi install -e sim && pixi run install-isaac-lab` "
                    f"in the training workspace. ({exc})"
                ) from exc
            launcher = AppLauncher(
                headless=self.headless, enable_cameras=True
            )
            self._app = launcher.app
            for _ in range(2):
                self._app.update()

        # 2. NOW it's safe to import lerobot_isaac_env (which transitively
        #    imports isaaclab.envs / managers).
        try:
            from lerobot_isaac_env import make_env  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "lerobot_isaac_env required for IsaacSO101Env. "
                "Install via the training workspace's editable-siblings "
                f"feature (pixi install -e sim). ({exc})"
            ) from exc

        # Translate this wrapper's `task` to lerobot_isaac_env's task name.
        # Sibling accepts: 'pick' | 'pick_and_place' | full gym IDs.
        task_alias = {
            "pickplace": "pick_and_place",
            "pick_and_place": "pick_and_place",
            "pick": "pick",
        }.get(self.task, self.task)

        global _GLOBAL_BACKING_ISAAC_ENV
        if _GLOBAL_BACKING_ISAAC_ENV is None:
            logger.info(
                "booting Isaac Lab env task=%s num_envs=%d headless=%s",
                task_alias, self.num_envs, self.headless,
            )
            _GLOBAL_BACKING_ISAAC_ENV = make_env(
                task=task_alias,
                num_envs=self.num_envs,
                headless=self.headless,
            )
        else:
            logger.info(
                "reusing existing Isaac Lab backing env (task=%s) — "
                "SimulationContext singleton enforced",
                task_alias,
            )
        self._isaac_env = _GLOBAL_BACKING_ISAAC_ENV
        # Eager flag: warm-up below is best-effort; if it throws we must
        # NOT re-enter _boot() and re-create the SimulationContext.
        self._booted = True

        # 30-frame warm-up so camera buffers are populated. Use the env's
        # sim handle; fall back to no-op if not exposed.
        sim = getattr(self._isaac_env, "sim", None)
        if sim is not None:
            for _ in range(WARM_UP_FRAMES):
                try:
                    sim.step(render=True)
                except Exception:  # noqa: BLE001
                    break

        # Probe whether the camera obs term is wired. If lerobot-isaac-env
        # still has NotImplementedError stubs for cameras, we'll find out
        # at the first translate and fall back to zeros without crashing.

    # ------------------------------------------------------------------ #
    # obs / action translation
    # ------------------------------------------------------------------ #

    def _translate_obs(self, raw_obs: Any) -> dict[str, np.ndarray]:
        """Convert ManagerBasedRLEnv obs (dict[group]→dict[term]→tensor)
        into the flat {rgb, state} dict sheeprl expects.

        Defensive: if camera term raises (the lerobot-isaac-env scaffold
        still has NotImplementedError for `wrist_camera_rgb`), return a
        zero RGB. Logs once.

        When LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1, concatenates the
        object_pose term (7 dims) to the 6-dim joint_pos vector, yielding
        a 13-dim state vector that gives the actor direct access to object
        location — the key diagnostic for the 2026-05-24 sweep collapse.
        """
        # raw_obs shapes seen in the wild:
        #   * dict[group(str)] -> dict[term(str)] -> Tensor    (older API)
        #   * dict[group(str)] -> Tensor (concat of all terms) (newer API,
        #     ObservationGroup with concatenate_terms=True default)
        if isinstance(raw_obs, dict):
            group = raw_obs.get("policy", raw_obs)
        else:
            group = raw_obs

        # ---- state (joint positions + optional object_pose) ----
        if isinstance(group, dict):
            # Term-wise obs (older API): concat joint_pos + object_pose if enabled.
            jp = group.get(self.state_key)
            parts = [
                self._tensor_to_np(
                    jp, default_shape=(_STATE_DIM_BASE,), default_dtype=np.float32
                ).reshape(-1)[:_STATE_DIM_BASE]
            ]
            if _INCLUDE_OBJECT_POSE:
                op = group.get("object_pose")
                op_np = self._tensor_to_np(
                    op,
                    default_shape=(_STATE_DIM_OBJECT_POSE,),
                    default_dtype=np.float32,
                ).reshape(-1)[:_STATE_DIM_OBJECT_POSE]
                parts.append(op_np)
            state_np = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        elif hasattr(group, "shape"):
            # Flat concat tensor (newer API): field order from PolicyObsGroupCfg:
            #   joint_pos[0:6] + joint_vel[6:12] + last_action[12:18]
            #   + object_pose[18:25] (when INCLUDE_OBJECT_POSE=1).
            flat = self._tensor_to_np(
                group, default_shape=(_STATE_DIM_BASE,), default_dtype=np.float32
            ).reshape(-1)
            parts = [flat[:_STATE_DIM_BASE]]
            if _INCLUDE_OBJECT_POSE:
                # object_pose lives at dims 18..25 per PolicyObsGroupCfg field order.
                if flat.size >= 25:
                    parts.append(flat[18:25])
                else:
                    parts.append(np.zeros(_STATE_DIM_OBJECT_POSE, dtype=np.float32))
            state_np = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        else:
            state_np = np.zeros(self._state_dim, dtype=np.float32)

        # Post-process: squeeze batch dim when single-env; coerce to declared dim.
        if state_np.ndim == 2 and state_np.shape[0] == self.num_envs:
            state_np = state_np[0]
        if state_np.size >= self._state_dim:
            state_np = state_np.reshape(-1)[: self._state_dim]
        else:
            state_np = np.zeros(self._state_dim, dtype=np.float32)

        # ---- rgb (camera) ----
        # Concat-tensor group has no camera key extraction path → falls
        # back to zero RGB until cameras are wired in lerobot-isaac-env.
        rgb_val = group.get(self.camera_key) if isinstance(group, dict) else None
        try:
            rgb_np = self._tensor_to_np(
                rgb_val,
                default_shape=(self.image_size, self.image_size, 3),
                default_dtype=np.uint8,
            )
        except NotImplementedError:
            # lerobot-isaac-env camera term is a stub; fall back to zeros
            # and remember so we don't retry every step.
            rgb_np = np.zeros(
                (self.image_size, self.image_size, 3), dtype=np.uint8
            )
            self._has_camera_term = False
        else:
            self._has_camera_term = rgb_val is not None

        # Normalise shape — Isaac Lab cameras emit (num_envs, H, W, 3) uint8.
        if rgb_np.ndim == 4 and rgb_np.shape[0] == self.num_envs:
            rgb_np = rgb_np[0]
        if rgb_np.ndim == 3 and rgb_np.shape[-1] == 3:
            self._last_rgb_hwc = rgb_np  # for render()
            rgb_np = rgb_np.transpose(2, 0, 1)  # → (3, H, W)
        elif rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
            self._last_rgb_hwc = rgb_np.transpose(1, 2, 0)
        # If shape is still off, coerce to the declared obs space.
        if rgb_np.shape != (3, self.image_size, self.image_size):
            rgb_np = np.zeros((3, self.image_size, self.image_size), dtype=np.uint8)
            self._last_rgb_hwc = np.zeros(
                (self.image_size, self.image_size, 3), dtype=np.uint8
            )

        return {
            "rgb": rgb_np.astype(np.uint8, copy=False),
            "state": state_np.astype(np.float32, copy=False),
        }

    def _scalar_info(self, raw_info: Any) -> dict[str, Any]:
        """Flatten Isaac Lab's batched info dict to a single-env dict."""
        if not isinstance(raw_info, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in raw_info.items():
            if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1:
                try:
                    out[k] = float(self._scalar(v))
                except Exception:  # noqa: BLE001
                    out[k] = v
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------ #
    # tiny helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tensor_to_np(
        val: Any,
        *,
        default_shape: tuple[int, ...],
        default_dtype: type,
    ) -> np.ndarray:
        if val is None:
            return np.zeros(default_shape, dtype=default_dtype)
        if hasattr(val, "detach"):
            return val.detach().cpu().numpy()
        if hasattr(val, "cpu"):
            return val.cpu().numpy()
        return np.asarray(val)

    def _to_torch(self, arr: np.ndarray) -> Any:
        """Bring an action array onto the env's torch device."""
        import torch  # local import — keep module light

        if hasattr(arr, "to"):
            return arr.to(self.device)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    @staticmethod
    def _scalar(t: Any) -> Any:
        """Squeeze a (1,)-shape tensor or array to a python scalar."""
        if t is None:
            return 0.0
        if hasattr(t, "detach"):
            return t.detach().cpu().reshape(-1)[0].item()
        if hasattr(t, "item"):
            try:
                return t.item()
            except Exception:  # noqa: BLE001
                pass
        arr = np.asarray(t).reshape(-1)
        return arr[0] if arr.size else 0.0


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

    Drop-in replacement for ``hdf5_env.get_hdf5_env``. Activate via
    ``env=isaac_so101`` (resolved against
    ``configs/env/isaac_so101.yaml``).
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
