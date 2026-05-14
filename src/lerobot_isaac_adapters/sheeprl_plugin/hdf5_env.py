"""HDF5-backed gymnasium env for sheeprl world-model training.

Reads frames produced by `lerobot_world_model_bridge` (shape:
``(B, T, H, W, C)`` uint8 plus ``(B, T, A)`` float32 actions) and replays
them as a gym episode. Each ``reset()`` picks a new window; each ``step()``
advances along the time axis. Reward is zero (no environment dynamics —
the agent learns a self-supervised world model from the observation stream).

This is intentionally minimal — it gives sheeprl's `dreamer_v3` a
`Dict({"rgb": Box, "state": Box})` observation stream that matches its
encoder.cnn_keys / encoder.mlp_keys defaults, so no algo config changes
are required beyond the env config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np


def _load_h5(hdf5_path: Path):
    """Load (B, T, H, W, C) frames + (B, T, A) actions from bridge HDF5."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        if "windows" in f:
            grp = f["windows"]
            frames = grp["frames"][...]
            actions = grp["actions"][...]
        elif "episodes" in f:
            # Stitch first 16 timesteps per episode.
            frames_list, actions_list = [], []
            for ep_name in list(f["episodes"].keys()):
                ep = f["episodes"][ep_name]
                T = min(ep["frames"].shape[0], 16)
                frames_list.append(ep["frames"][:T])
                actions_list.append(ep["actions"][:T])
            T_min = min(x.shape[0] for x in frames_list)
            frames = np.stack([x[:T_min] for x in frames_list], axis=0)
            actions = np.stack([x[:T_min] for x in actions_list], axis=0)
        else:
            raise RuntimeError(
                f"HDF5 has neither `windows` nor `episodes`: {hdf5_path}"
            )
    return frames, actions


class HDF5ReplayEnv(gym.Env):
    """Replay-style env that streams pre-recorded LeRobot frames as observations.

    Parameters
    ----------
    dataset_path:
        Path to HDF5 produced by `lerobot_world_model_bridge`.
    seed:
        Random seed for episode selection.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, dataset_path: str, seed: int | None = None):
        super().__init__()
        self._frames, self._actions = _load_h5(Path(dataset_path))
        # Shape sanity: bridge writes (B, T, H, W, C) uint8.
        assert self._frames.ndim == 5, (
            f"Expected (B,T,H,W,C) frames, got {self._frames.shape}"
        )
        B, T, H, W, C = self._frames.shape
        self._n_windows = B
        self._window_len = T
        self._h, self._w, self._c = H, W, C
        self._action_dim = int(self._actions.shape[-1])

        # Dreamer's default cnn_keys expects "rgb" with shape (C, H, W).
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    0, 255, shape=(C, H, W), dtype=np.uint8
                ),
                "state": gym.spaces.Box(
                    -1e9, 1e9, shape=(self._action_dim,), dtype=np.float32
                ),
            }
        )
        # Continuous action space matching the recorded action dim. Bounded so
        # dreamer's actor head produces in-range outputs.
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self._action_dim,), dtype=np.float32
        )
        self.reward_range = (-np.inf, np.inf)

        self._rng = np.random.default_rng(seed)
        self._window_idx = 0
        self._t = 0

    # ------------------------------------------------------------------
    # gym.Env API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._window_idx = int(self._rng.integers(0, self._n_windows))
        self._t = 0
        return self._get_obs(), {}

    def step(self, action):
        # Action is ignored — replay env. Advance along the recorded window.
        self._t += 1
        done = self._t >= self._window_len - 1
        return self._get_obs(), 0.0, bool(done), False, {}

    def render(self):
        # Return current rgb frame for sheeprl's RecordVideoV0 wrapper.
        return self._frames[self._window_idx, self._t]

    def close(self):
        pass

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _get_obs(self):
        rgb_hwc = self._frames[self._window_idx, self._t]
        rgb_chw = np.transpose(rgb_hwc, (2, 0, 1))
        # State channel exposes the recorded action at the current step so
        # mlp-keys consumers can condition on it if they want.
        state = self._actions[self._window_idx, self._t].astype(np.float32)
        return {"rgb": rgb_chw, "state": state}


def get_hdf5_env(
    dataset_path: str, window_size: int = 16, seed: int | None = None
):
    """Hydra-friendly factory wrapping ``HDF5ReplayEnv``.

    ``window_size`` is accepted for forward compatibility but currently
    inferred from the HDF5 itself.
    """
    del window_size  # currently inferred from HDF5
    return HDF5ReplayEnv(dataset_path=dataset_path, seed=seed)
