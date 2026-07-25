"""Demonstration buffer + BC loss for DreamerV3 warm-start (DreamerFD recipe).

Carry→place won't fall to reward shaping (plateau ~−8.5 across all variants); the
fix is demonstration bootstrapping (research: project-context/research/
dreamerv3-demo-bootstrap-curriculum.md, DreamerFD arXiv:2303.03675). This module
is the schema-agnostic, unit-testable FOUNDATION:

  * load_sim_demos()  — LeRobotDataset (SIM schema: d435_rgb 64², state 18, action
                        6) → sheeprl step-data arrays. NOTE: the real
                        `so101-pickplace-new` has a different schema (overhead cam,
                        12-dim state) → must use SIM demos (scripted/teleop-in-sim).
  * DemoBuffer        — stores episodes, samples (batch, seq_len) sequences.
  * behavior_cloning_loss() — BC term with the DreamerFD decay + "virtual clutch".

The sheeprl train-loop integration (50/50 demo+online sampling, adding the BC
gradient) is a separate monkeypatch in _wm_isaac_entry.py — GPU-gated, needs real
sim demos to verify. This module has NO sheeprl/torch-heavy deps at import so it
unit-tests without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# --------------------------------------------------------------------------- #
# Loader: LeRobotDataset (sim schema) → per-episode sheeprl step-data arrays
# --------------------------------------------------------------------------- #
def load_sim_demos(
    dataset_root: str,
    image_size: int = 64,
    camera_key: str = "observation.images.d435_rgb",
    state_key: str = "observation.state",
    action_key: str = "action",
    max_episodes: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Load SIM demo episodes into sheeprl step-data arrays.

    Returns a list of per-episode dicts with keys matching sheeprl's replay-buffer
    schema: ``rgb`` (T,3,image_size,image_size) uint8, ``state`` (T,state_dim) float32,
    ``actions`` (T,action_dim) float32, ``rewards`` (T,1) float32, ``terminated``
    (T,1) bool, ``truncated`` (T,1) bool, ``is_first`` (T,1) bool. Rewards are 0
    (the world model learns dynamics; the BC loss handles policy imitation —
    DreamerFD). Heavy deps (lerobot/torch) imported lazily so this module imports
    GPU-free.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from pathlib import Path

    parts = Path(dataset_root).resolve().parts
    repo_id = "/".join(parts[-2:]) if len(parts) >= 2 else Path(dataset_root).name
    ds = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))

    # Optional per-episode env-reward sidecar (lerobot 0.5.1 can't store a (1,) reward
    # feature). When present, real rewards replace the reward-0 default — seeding demos
    # with reward 0 poisons the DreamerV3 reward model (warmstart-v1 plateaued ~-30).
    rew_dir = Path(dataset_root) / "meta" / "demo_rewards"

    episodes: list[dict[str, np.ndarray]] = []
    n_eps = ds.meta.total_episodes if max_episodes is None else min(max_episodes, ds.meta.total_episodes)
    for ep_idx in range(n_eps):
        frames = _episode_frames(ds, ep_idx)
        if not frames:
            continue
        sd = _frames_to_stepdata(frames, image_size, camera_key, state_key, action_key)
        rew_file = rew_dir / f"ep_{ep_idx:04d}.npy"
        if rew_file.exists():
            rew = np.load(rew_file).astype(np.float32).reshape(-1)
            if rew.shape[0] == sd["rewards"].shape[0]:
                sd["rewards"] = rew.reshape(-1, 1)
        episodes.append(sd)
    return episodes


def _episode_frames(ds: Any, ep_idx: int) -> list[dict[str, Any]]:
    """Collect the rows of one episode from a LeRobotDataset.

    lerobot 0.5.1 dropped ``episode_data_index``; frames are grouped by the
    ``episode_index`` column of the underlying hf_dataset.
    """
    if getattr(ds, "episode_data_index", None) is not None:
        from_idx = ds.episode_data_index["from"][ep_idx].item()
        to_idx = ds.episode_data_index["to"][ep_idx].item()
        return [ds[i] for i in range(from_idx, to_idx)]
    ep_col = ds.hf_dataset["episode_index"]  # column -> list of ints
    rng = [i for i, e in enumerate(ep_col) if int(e) == ep_idx]
    return [ds[i] for i in rng]


def _frames_to_stepdata(frames, image_size, camera_key, state_key, action_key) -> dict[str, np.ndarray]:
    import torch

    T = len(frames)
    rgb = np.zeros((T, 3, image_size, image_size), dtype=np.uint8)
    state0 = _to_np(frames[0].get(state_key)).reshape(-1)
    state = np.zeros((T, state0.shape[0]), dtype=np.float32)
    act0 = _to_np(frames[0].get(action_key)).reshape(-1)
    actions = np.zeros((T, act0.shape[0]), dtype=np.float32)
    for t, f in enumerate(frames):
        img = _to_np(f.get(camera_key))  # (3,H,W) float[0,1] or uint8, or (H,W,3)
        rgb[t] = _to_chw_uint8(img, image_size)
        state[t] = _to_np(f.get(state_key)).reshape(-1)
        actions[t] = _to_np(f.get(action_key)).reshape(-1)
    z = np.zeros((T, 1), dtype=np.float32)
    is_first = np.zeros((T, 1), dtype=bool)
    is_first[0] = True
    terminated = np.zeros((T, 1), dtype=bool)
    truncated = np.zeros((T, 1), dtype=bool)
    # Success demos: the final step is a TRUE terminal (object placed), not a time
    # truncation → terminated[-1]=True so DreamerV3 doesn't bootstrap past it.
    terminated[-1] = True
    return {"rgb": rgb, "state": state, "actions": actions, "rewards": z,
            "terminated": terminated, "truncated": truncated, "is_first": is_first}


def _to_np(v: Any) -> np.ndarray:
    if v is None:
        return np.zeros(1, dtype=np.float32)
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _to_chw_uint8(img: np.ndarray, size: int) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[-1] == 3:  # HWC → CHW
        a = a.transpose(2, 0, 1)
    if a.dtype != np.uint8:  # float [0,1] → uint8
        a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
    if a.shape[1:] != (size, size):
        a = _resize_chw_bilinear(a, size)
    return a.astype(np.uint8, copy=False)


def _resize_chw_bilinear(chw_np: np.ndarray, size: int) -> np.ndarray:
    """Resize (3,H,W) uint8 → (3,size,size) uint8 with bilinear interpolation —
    SAME method as IsaacSO101Env._resize_chw (the online path), so demo frames are
    pixel-consistent with what the env produces (not aspect-distorted nearest-
    neighbour). torch fallback to stride-subsample if torch is unavailable.
    """
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(np.ascontiguousarray(chw_np)).unsqueeze(0).float()
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        return t.squeeze(0).clamp_(0, 255).to(torch.uint8).numpy()
    except Exception:  # noqa: BLE001
        ys = np.linspace(0, chw_np.shape[1] - 1, size).astype(np.int64)
        xs = np.linspace(0, chw_np.shape[2] - 1, size).astype(np.int64)
        return chw_np[:, ys][:, :, xs].astype(np.uint8, copy=False)


# --------------------------------------------------------------------------- #
# DemoBuffer — stores episodes, samples (batch, seq_len) sequences
# --------------------------------------------------------------------------- #
@dataclass
class DemoBuffer:
    """Fixed demo buffer; samples contiguous (batch, seq_len) sequences within
    episodes (no episode-boundary crossing — matches sheeprl's sequence sampling)."""

    episodes: list[dict[str, np.ndarray]] = field(default_factory=list)
    _rng: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.default_rng(0)
        # only keep episodes long enough to sample 1 step
        self.episodes = [e for e in self.episodes if len(e["actions"]) >= 1]

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def n_transitions(self) -> int:
        return int(sum(len(e["actions"]) for e in self.episodes))

    def sample(self, batch_size: int, seq_len: int) -> dict[str, np.ndarray]:
        """Return a dict of (seq_len, batch_size, ...) arrays (sheeprl layout)."""
        eligible = [i for i, e in enumerate(self.episodes) if len(e["actions"]) >= seq_len]
        if not eligible:
            raise ValueError(f"no demo episode >= seq_len={seq_len} (max ep len "
                             f"{max((len(e['actions']) for e in self.episodes), default=0)})")
        keys = ("rgb", "state", "actions", "rewards", "terminated", "truncated", "is_first")
        out: dict[str, list] = {k: [] for k in keys}
        for _ in range(batch_size):
            ei = eligible[self._rng.integers(len(eligible))]
            ep = self.episodes[ei]
            T = len(ep["actions"])
            s = int(self._rng.integers(0, T - seq_len + 1))
            for k in keys:
                out[k].append(ep[k][s:s + seq_len])
        # stack → (batch, seq, ...) then transpose to (seq, batch, ...)
        return {k: np.stack(out[k], axis=0).swapaxes(0, 1) for k in keys}


# --------------------------------------------------------------------------- #
# Behavior-cloning loss (DreamerFD: weight decays 1→0; "virtual clutch")
# --------------------------------------------------------------------------- #
def bc_weight(step: int, start: float = 1.0, decay_steps: int = 50_000) -> float:
    """Linear BC-weight schedule start → 0.0 over decay_steps (DreamerFD)."""
    if decay_steps <= 0:
        return 0.0  # no schedule / guard div-by-zero
    if step >= decay_steps:
        return 0.0
    return float(start * (1.0 - step / decay_steps))


def behavior_cloning_loss(
    actor_logprob: Callable[[Any, Any], Any],
    latents: Any,
    demo_actions: Any,
    step: int,
    kl: float | None = None,
    kl_clutch: float = 3.0,
    decay_steps: int = 50_000,
) -> Any:
    """BC loss = -w(step) * E[ log π(a_demo | latent) ], with a "virtual clutch"
    that zeroes the BC gradient when the world-model latent KL is large (early
    training instability) — DreamerFD. ``actor_logprob(latents, demo_actions)``
    returns per-sample log-probs (torch tensor). Returns a torch scalar.
    """
    import torch

    w = bc_weight(step, decay_steps=decay_steps)
    if kl is not None and kl > kl_clutch:
        w = 0.0
    if w == 0.0:
        # True fast-path: skip the actor forward pass entirely. The caller does
        # not backward through a zero BC term, so device is irrelevant.
        return torch.zeros((), dtype=torch.float32)
    logp = actor_logprob(latents, demo_actions)
    return -w * logp.mean()
