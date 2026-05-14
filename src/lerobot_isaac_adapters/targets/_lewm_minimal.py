"""
_lewm_minimal
=============

Minimal next-embedding predictor "LeWM-style" trainer.

The HF `lerobot 0.5.x` package does not ship `lerobot.scripts.train_world_model`,
so this module provides an in-process replacement: a small CNN encoder paired
with a linear-in-embedding-space dynamics head trained to predict the
embedding of the next frame from the current frame + action.

It consumes the same HDF5 produced by `lerobot_world_model_bridge` with
`image_size=(96, 96)` and `window_size=16`. Each evaluation step prints

    pred_loss=<float>

on stdout so that the metric-extractor / autoresearch executor regex picks
it up unchanged.

Design constraints
------------------
- No external deep-learning deps beyond torch (already a lerobot dep).
- Runs on RTX 3080 10 GB at batch_size <= 8 with mixed-precision off.
- Pure self-supervised — no reward, no policy, no rollout.
- Soft-import of torch / h5py so the module remains importable without GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator


def _load_h5_windows(hdf5_path: Path, n_windows_cap: int | None = None):
    """Load (windows, T, H, W, C) frames + (windows, T, A) actions from HDF5.

    The bridge writes a top-level ``windows`` group with ``frames`` / ``actions``
    datasets when ``window_size`` is set. Falls back to per-episode arrays if
    only ``episodes`` is present.
    """
    import h5py
    import numpy as np

    with h5py.File(hdf5_path, "r") as f:
        if "windows" in f:
            grp = f["windows"]
            frames = grp["frames"][...]
            actions = grp["actions"][...]
        elif "episodes" in f:
            # Stitch per-episode arrays into a synthetic 1-window-per-ep stack.
            frames_list, actions_list = [], []
            for ep_name in list(f["episodes"].keys()):
                ep = f["episodes"][ep_name]
                frames_list.append(ep["frames"][...])
                actions_list.append(ep["actions"][...])
            T = min(min(x.shape[0] for x in frames_list), 16)
            frames = np.stack([x[:T] for x in frames_list], axis=0)
            actions = np.stack([x[:T] for x in actions_list], axis=0)
        else:
            raise RuntimeError(
                f"HDF5 has neither `windows` nor `episodes` group: {hdf5_path}"
            )

        if n_windows_cap is not None and frames.shape[0] > n_windows_cap:
            frames = frames[:n_windows_cap]
            actions = actions[:n_windows_cap]

        return frames, actions


def _iter_batches(
    frames, actions, batch_size: int, shuffle: bool = True, seed: int = 42
) -> Iterator[tuple]:
    """Infinite iterator over (frames, actions) mini-batches.

    Shapes: frames (B, T, H, W, C) uint8, actions (B, T, A) float32.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = frames.shape[0]
    while True:
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield frames[idx], actions[idx]


def _build_model(img_hw: int, embed_dim: int, action_dim: int):
    """Tiny CNN encoder + linear forward dynamics head."""
    import torch.nn as nn

    encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # img_hw -> /2
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, embed_dim),
    )

    dynamics = nn.Sequential(
        nn.Linear(embed_dim + action_dim, embed_dim * 2),
        nn.ReLU(inplace=True),
        nn.Linear(embed_dim * 2, embed_dim),
    )
    return encoder, dynamics


def train(args: argparse.Namespace) -> int:
    """Train the minimal LeWM-style predictor.

    Returns
    -------
    int
        0 on success, non-zero on failure.

    Stdout contract
    ---------------
    Every ``log_every`` updates emit one line::

        pred_loss=<float>
    """
    try:
        import torch
    except ImportError:
        print(
            "[lewm_minimal] torch not installed; cannot train. "
            "Run `bash scripts/install_train_deps.sh --lewm` first.",
            file=sys.stderr,
        )
        return 127

    hdf5_path = Path(args.dataset)
    if not hdf5_path.exists():
        print(
            f"[lewm_minimal] dataset not found: {hdf5_path}", file=sys.stderr
        )
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_every = int(getattr(args, "log_every", 50) or 50)
    batch_size = max(1, int(args.batch_size))
    lr = float(args.lr)
    total_steps = int(args.steps)
    seed = int(args.seed)

    torch.manual_seed(seed)

    print(f"[lewm_minimal] Loading windows from {hdf5_path}...")
    frames, actions = _load_h5_windows(hdf5_path)
    print(
        f"[lewm_minimal] frames shape={frames.shape} actions shape={actions.shape}"
    )

    assert frames.ndim == 5, f"expected (B,T,H,W,C), got {frames.shape}"
    img_h, img_w = frames.shape[2], frames.shape[3]
    action_dim = actions.shape[-1]
    embed_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, dynamics = _build_model(img_h, embed_dim, action_dim)
    encoder = encoder.to(device)
    dynamics = dynamics.to(device)
    print(
        f"[lewm_minimal] device={device} embed_dim={embed_dim} "
        f"action_dim={action_dim} img={img_h}x{img_w}"
    )

    params = list(encoder.parameters()) + list(dynamics.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    n_params = sum(p.numel() for p in params)
    print(f"[lewm_minimal] n_params={n_params}")

    loss_fn = torch.nn.MSELoss()
    start_time = time.monotonic()
    batches = _iter_batches(frames, actions, batch_size, seed=seed)

    for step, (batch_frames, batch_actions) in enumerate(batches, start=1):
        if step > total_steps:
            break

        # (B, T, H, W, C) uint8 -> (B*T, C, H, W) float32 in [0,1]
        bf = (
            torch.from_numpy(batch_frames)
            .to(device, non_blocking=True)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .float()
            / 255.0
        )
        ba = torch.from_numpy(batch_actions).to(device, non_blocking=True).float()
        B, T, C, H, W = bf.shape

        # Encode all frames in the window.
        z = encoder(bf.view(B * T, C, H, W)).view(B, T, embed_dim)

        # Predict z[t+1] from (z[t], a[t]).
        z_in = z[:, :-1]
        a_in = ba[:, :-1]
        z_tgt = z[:, 1:].detach()
        z_pred = dynamics(torch.cat([z_in, a_in], dim=-1))
        loss = loss_fn(z_pred, z_tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % log_every == 0 or step == total_steps:
            sec = time.monotonic() - start_time
            sps = step / max(sec, 1e-6)
            # Canonical autoresearch metric line — must be one of the last
            # lines emitted, hence the explicit print of just the metric.
            print(
                f"[lewm_minimal] step={step} elapsed={sec:.1f}s sps={sps:.2f}"
            )
            print(f"pred_loss={loss.item():.6f}", flush=True)

    # Persist checkpoint.
    ckpt_path = out_dir / "lewm_minimal_last.pt"
    torch.save(
        {
            "step": min(step, total_steps),
            "encoder": encoder.state_dict(),
            "dynamics": dynamics.state_dict(),
            "config": {
                "embed_dim": embed_dim,
                "action_dim": action_dim,
                "img_h": img_h,
                "img_w": img_w,
            },
        },
        ckpt_path,
    )
    print(f"[lewm_minimal] Checkpoint saved: {ckpt_path}")
    # Emit one final canonical metric line for autoresearch even on clean exit.
    print(f"pred_loss={loss.item():.6f}", flush=True)
    return 0
