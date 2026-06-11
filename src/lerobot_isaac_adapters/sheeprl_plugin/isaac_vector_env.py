"""IsaacSO101VectorEnv — true num_envs>1 vectorization for online Isaac DreamerV3.

Fix 2 (plans/2026-06-08-fix2-isaac-vectorization-plan.md). The single-env
``IsaacSO101Env`` boots ONE Isaac ``ManagerBasedRLEnv`` with num_envs=N but
collapses every output to env-0, so sheeprl (told num_envs) gets 1-env batches →
crashes at num_envs>1. This class boots the SAME backing env (singleton) and
returns FULL batched (N, …) results, exposing the gymnasium VectorEnv API.

Activated only at num_envs>1 via a monkeypatch in scripts/_wm_isaac_entry.py;
num_envs=1 keeps using the untouched IsaacSO101Env (zero regression).
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import (
    DEFAULT_CAMERA_KEY,
    DEFAULT_STATE_KEY,
    IsaacSO101Env,
    _STATE_DIM_BASE,
    _STATE_DIM_OBJECT_POSE,
    _INCLUDE_OBJECT_POSE,
)

logger = logging.getLogger(__name__)


class IsaacSO101VectorEnv(gym.vector.VectorEnv):
    """Batched SO-101 pick-place env over ONE Isaac backing env (num_envs=N)."""

    metadata = {"render_modes": ["rgb_array"], "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}

    def __init__(self, existing_env: IsaacSO101Env, num_envs: int | None = None) -> None:
        """Wrap an ALREADY-CONSTRUCTED single-env IsaacSO101Env (which boots ONE
        Isaac ManagerBasedRLEnv with num_envs=N internally) and expose its N
        sub-envs as a batched VectorEnv. The patch builds the single wrapper from
        env_fns[0]() (boots the singleton) and hands it here — so envs 1..N-1 are
        never separately constructed (avoids the singleton crash).
        """
        w = existing_env
        self._w = w
        self.num_envs = int(num_envs if num_envs is not None else w.num_envs)
        self.task = w.task
        self.image_size = w.image_size
        self.max_episode_steps = w.max_episode_steps
        self.state_key = w.state_key
        self.camera_key = w.camera_key
        self.enable_cameras = w.enable_cameras
        self.device = w.device
        self._state_dim = _STATE_DIM_BASE + (_STATE_DIM_OBJECT_POSE if _INCLUDE_OBJECT_POSE else 0)
        # Episode-return tracking → emit infos["episode"]/["_episode"] like
        # gymnasium's RecordEpisodeStatistics, so sheeprl's rew_avg logger works
        # (paired with _patch_gym_vector_final_info in _wm_isaac_entry.py).
        self._ep_r = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_l = np.zeros(self.num_envs, dtype=np.int64)

        self.single_observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(low=0, high=255, shape=(3, self.image_size, self.image_size), dtype=np.uint8),
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32),
        })
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        # PER-ENV step counter — a single scalar would desync once envs terminate
        # early (SAME_STEP autoreset), mislabelling truncated + episode length.
        self._t = np.zeros(self.num_envs, dtype=np.int64)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if not self._w._booted:
            self._w._boot()
        self._t[:] = 0
        raw_obs, raw_info = self._w._isaac_env.reset(seed=seed)
        return self._batched_obs(raw_obs), self._batched_info(raw_info)

    def step(self, actions: np.ndarray):
        self._t += 1  # elementwise (per-env)
        import torch

        act = actions if hasattr(actions, "to") else torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        act = act.to(self.device).view(self.num_envs, -1)
        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = self._w._isaac_env.step(act)
        obs = self._batched_obs(raw_obs)
        reward = self._to_np(raw_rew).reshape(-1)[: self.num_envs].astype(np.float32)
        terminated = self._to_np(raw_term).reshape(-1)[: self.num_envs].astype(bool)
        truncated = self._to_np(raw_trunc).reshape(-1)[: self.num_envs].astype(bool)
        # per-env time-limit truncation
        truncated = truncated | (self._t >= self.max_episode_steps)
        info = self._batched_info(raw_info)

        # Episode-return tracking → emit infos["episode"]/["_episode"] on done.
        self._ep_r += reward
        self._ep_l += 1
        done = terminated | truncated
        if done.any():
            info = dict(info)
            info["episode"] = {"r": self._ep_r.copy(), "l": self._ep_l.copy().astype(np.float64)}
            info["_episode"] = done.copy()
            # also emit final_info (sheeprl 0.5.8 reads per-episode stats from it)
            final_info: list = [None] * self.num_envs
            for i in range(self.num_envs):
                if bool(done[i]):
                    final_info[i] = {"episode": {"r": np.array([float(self._ep_r[i])]),
                                                 "l": np.array([float(self._ep_l[i])])}}
            info["final_info"] = final_info
            # zero per-env accumulators + step counter for the envs that finished
            self._ep_r[done] = 0.0
            self._ep_l[done] = 0
            self._t[done] = 0
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_np(v: Any) -> np.ndarray:
        if v is None:
            return np.zeros(1, dtype=np.float32)
        if hasattr(v, "detach"):
            return v.detach().cpu().numpy()
        if hasattr(v, "cpu"):
            return v.cpu().numpy()
        return np.asarray(v)

    def _batched_obs(self, raw_obs: Any) -> dict[str, np.ndarray]:
        group = raw_obs.get("policy", raw_obs) if isinstance(raw_obs, dict) else raw_obs

        # ---- state (N, state_dim) ----
        if isinstance(group, dict):
            jp = self._to_np(group.get(self.state_key)).reshape(self.num_envs, -1)[:, :_STATE_DIM_BASE]
            parts = [jp]
            if _INCLUDE_OBJECT_POSE:
                op = self._to_np(group.get("object_pose")).reshape(self.num_envs, -1)[:, :_STATE_DIM_OBJECT_POSE]
                parts.append(op)
            state = np.concatenate(parts, axis=1)
        else:
            flat = self._to_np(group).reshape(self.num_envs, -1)
            parts = [flat[:, :_STATE_DIM_BASE]]
            if _INCLUDE_OBJECT_POSE:
                parts.append(flat[:, 18:25] if flat.shape[1] >= 25 else np.zeros((self.num_envs, _STATE_DIM_OBJECT_POSE), np.float32))
            state = np.concatenate(parts, axis=1)
        if state.shape[1] < self._state_dim:
            state = np.pad(state, ((0, 0), (0, self._state_dim - state.shape[1])))
        state = state[:, : self._state_dim].astype(np.float32, copy=False)

        # ---- rgb (N, 3, image_size, image_size) ----
        rgb_val = group.get(self.camera_key) if isinstance(group, dict) else None
        rgb = np.zeros((self.num_envs, 3, self.image_size, self.image_size), dtype=np.uint8)
        if rgb_val is not None:
            arr = self._to_np(rgb_val)  # (N, H, W, 3) or (N, 3, H, W)
            if arr.ndim == 4:
                if arr.shape[-1] == 3:
                    arr = arr.transpose(0, 3, 1, 2)  # NHWC → NCHW
                for i in range(min(self.num_envs, arr.shape[0])):
                    chw = arr[i]
                    if chw.shape[1:] != (self.image_size, self.image_size):
                        chw = IsaacSO101Env._resize_chw(chw, self.image_size)
                    rgb[i] = chw.astype(np.uint8, copy=False)
                self._last_rgb_hwc = rgb[0].transpose(1, 2, 0)
        return {"rgb": rgb, "state": state}

    def _batched_info(self, raw_info: Any) -> dict[str, Any]:
        return raw_info if isinstance(raw_info, dict) else {}

    def render(self) -> np.ndarray:
        return getattr(self, "_last_rgb_hwc", np.zeros((self.image_size, self.image_size, 3), np.uint8)).copy()

    def close(self, **kwargs) -> None:
        # NO-OP on the shared backing singleton (same discipline as IsaacSO101Env).
        logger.info("IsaacSO101VectorEnv.close(): no-op — backing env kept alive.")
