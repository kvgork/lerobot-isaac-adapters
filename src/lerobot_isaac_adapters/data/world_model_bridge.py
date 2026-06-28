"""World-model dataset bridge — LeRobotDataset (Parquet+MP4) -> HDF5/npz/WebDataset.

Moved here from the ``lerobot_world_model_bridge`` Claude-Code skill (2026-06-28): core
data-pipeline code belongs in a versioned, testable package module, not a skill (the skill
had a source[claude_code]/installed[~/.claude] split that caused sync friction). The skill
``operations.py`` is now a thin shim re-exporting from this module; the
``lerobot-worldmodel-bridge`` agent still works via that shim. Public API unchanged:
``lerobot_to_worldmodel``, ``inspect_dataset``, ``validate_output``, ``OperationResult``.
"""
from __future__ import annotations

import io
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class OperationResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE_SIZE: Tuple[int, int] = (64, 64)
_NON_FEATURE_COLUMNS = {
    "timestamp", "frame_index", "episode_index", "index",
    "task_index", "next.done", "next.reward",
    # the recorder / DR writers use bare `reward` / `done` (not the gym `next.*`
    # names); exclude them from action/state autodetect, they are carried explicitly.
    "reward", "done",
}


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspect_dataset(dataset_path: str) -> OperationResult:
    """
    Report LeRobotDataset schema: parquet columns, image keys, episode count.

    Returns enough info for the user to choose action/state/image keys
    explicitly if auto-detection picks wrong columns.
    """
    try:
        import pandas as pd
    except ImportError:
        return OperationResult(
            success=False,
            error="pandas required. Run: pip install pandas pyarrow",
        )

    root = Path(dataset_path)
    if not root.exists():
        return OperationResult(success=False, error=f"Dataset not found: {dataset_path}")

    parquet_files = sorted(root.glob("data/**/*.parquet"))
    if not parquet_files:
        return OperationResult(
            success=False,
            error=f"No parquet files under {dataset_path}/data/",
            suggestions=[
                "Confirm the path points to a LeRobotDataset root (must contain data/ and meta/)",
                "Check the dataset version — v2.0 layout differs",
            ],
        )

    sample_df = pd.read_parquet(parquet_files[0])
    columns = list(sample_df.columns)
    action_keys, state_keys, _ = _autodetect_columns(columns)

    image_dirs = sorted(p.name for p in (root / "videos").glob("*") if p.is_dir())
    n_episodes = (
        int(sample_df["episode_index"].nunique())
        if "episode_index" in sample_df.columns else None
    )

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    return OperationResult(
        success=True,
        data={
            "dataset_path": str(root),
            "parquet_files": len(parquet_files),
            "columns": columns,
            "detected_action_keys": action_keys,
            "detected_state_keys": state_keys,
            "image_keys": image_dirs,
            "episodes_in_first_shard": n_episodes,
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
        },
    )


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def lerobot_to_worldmodel(
    dataset_path: str,
    output_path: str,
    output_format: str = "hdf5",
    image_size: Tuple[int, int] = _DEFAULT_IMAGE_SIZE,
    window_size: Optional[int] = None,
    stride: int = 1,
    image_keys: Optional[List[str]] = None,
    action_keys: Optional[List[str]] = None,
    state_keys: Optional[List[str]] = None,
    max_episodes: Optional[int] = None,
    normalize_actions: bool = False,
    fps: Optional[int] = None,
    shard_size: int = 256,
) -> OperationResult:
    """
    Convert a LeRobotDataset into a world-model training dataset.

    Args:
        dataset_path: LeRobotDataset root.
        output_path: Output file (hdf5/npz dir) or shard dir (webdataset).
        output_format: "hdf5" | "npz" | "webdataset".
        image_size: Target (H, W) for resized frames.
        window_size: If set, also emit sliding windows of this length.
        stride: Window stride (only used when window_size is set).
        image_keys: Camera video directories under videos/ (auto-detected if None).
        action_keys / state_keys: Parquet columns (auto-detected if None).
        max_episodes: Limit number of episodes (None = all).
        normalize_actions: Z-score actions and store mean/std in attrs.
        fps: Override video framerate (read from meta/info.json otherwise).
        shard_size: WebDataset samples per .tar shard.

    Returns:
        OperationResult with episode counts, output path, normalization stats.
    """
    if output_format not in {"hdf5", "npz", "webdataset"}:
        return OperationResult(
            success=False,
            error=f"Unknown output_format: {output_format!r}",
            suggestions=["Use 'hdf5', 'npz', or 'webdataset'"],
        )

    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return OperationResult(
            success=False,
            error="numpy + pandas required. Run: pip install numpy pandas pyarrow",
        )

    root = Path(dataset_path)
    parquet_files = sorted(root.glob("data/**/*.parquet"))
    if not parquet_files:
        return OperationResult(success=False, error=f"No parquet files under {root}/data/")

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    auto_actions, auto_states, _ = _autodetect_columns(list(df.columns))
    action_keys = action_keys or auto_actions
    state_keys = state_keys or auto_states

    if not action_keys:
        return OperationResult(
            success=False,
            error="No action columns detected; pass action_keys explicitly.",
            suggestions=[f"Available columns: {list(df.columns)[:20]}..."],
        )

    # Detect features stored as inline image bytes in parquet (dtype="image"),
    # in addition to MP4 video files under videos/ (dtype="video"). LeRobot v3.0
    # datasets may use either, and the bridge needs to handle both.
    info_path = root / "meta" / "info.json"
    info_features: Dict[str, Any] = {}
    if info_path.exists():
        try:
            info_features = json.loads(info_path.read_text()).get("features", {}) or {}
        except (json.JSONDecodeError, OSError):
            info_features = {}
    image_dtype_keys = sorted(
        k for k, spec in info_features.items()
        if isinstance(spec, dict)
        and spec.get("dtype") == "image"
        and k in df.columns
    )

    state_only = image_keys is not None and len(image_keys) == 0
    if image_keys is None:
        video_dir_keys = [p.name for p in (root / "videos").glob("*") if p.is_dir()]
        if video_dir_keys:
            image_keys = video_dir_keys
        elif image_dtype_keys:
            # Inline image bytes in parquet (dtype="image"); no videos/ dir needed.
            image_keys = image_dtype_keys
        else:
            return OperationResult(
                success=False,
                error=(
                    f"No video directories under {root}/videos/ and no inline "
                    f"image columns (dtype=image) detected in parquet"
                ),
                suggestions=[
                    "Pass image_keys=[] to convert without frames (state-only world model)",
                    "Check meta/info.json features for dtype: image or video",
                ],
            )

    if fps is None:
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            fps = int(json.loads(info_path.read_text()).get("fps", 30))
        else:
            fps = 30

    ep_col = "episode_index"
    if ep_col not in df.columns:
        return OperationResult(success=False, error="episode_index column missing")

    episode_ids = sorted(df[ep_col].unique().tolist())
    if max_episodes is not None:
        episode_ids = episode_ids[:max_episodes]

    # First pass — collect actions for normalization stats
    action_mean: Optional["np.ndarray"] = None
    action_std: Optional["np.ndarray"] = None
    if normalize_actions:
        all_actions = _columns_to_float_array(df[df[ep_col].isin(episode_ids)], action_keys)
        action_mean = all_actions.mean(axis=0)
        action_std = all_actions.std(axis=0) + 1e-6

    episodes_data: List[Dict[str, Any]] = []
    for ep_id in episode_ids:
        ep_df = df[df[ep_col] == ep_id].sort_values("frame_index")
        actions = _columns_to_float_array(ep_df, action_keys)
        if normalize_actions and action_mean is not None and action_std is not None:
            actions = (actions - action_mean) / action_std
        states = (
            _columns_to_float_array(ep_df, state_keys) if state_keys else None
        )
        # Reward/done column conventions: the SO-101 recorder + DR writers use bare
        # `reward` / `done` (sparse_terminal_success); gym-style datasets use
        # `next.reward` / `next.done`. Prefer the bare names, fall back to next.*.
        _reward_col = "reward" if "reward" in ep_df.columns else (
            "next.reward" if "next.reward" in ep_df.columns else None
        )
        _done_col = "done" if "done" in ep_df.columns else (
            "next.done" if "next.done" in ep_df.columns else None
        )
        rewards = (
            ep_df[_reward_col].to_numpy().astype("float32").reshape(-1)
            if _reward_col else None
        )
        dones = (
            ep_df[_done_col].to_numpy().astype("bool").reshape(-1)
            if _done_col else None
        )

        # Decode video frames per camera, resize, take first camera (most common case).
        # When state_only is True the caller explicitly opted out of video loading.
        frames = None
        if image_keys and not state_only:
            primary_key = image_keys[0]
            if primary_key in image_dtype_keys:
                frames = _load_episode_frames_from_parquet(
                    ep_df, primary_key, image_size,
                )
                err_hint = (
                    f"Failed to decode inline parquet image bytes for episode "
                    f"{ep_id}, column {primary_key}"
                )
                err_sugg = [
                    "Verify Pillow + opencv-python are installed",
                    f"Check parquet column {primary_key} contains struct<bytes,path>",
                ]
            else:
                frames = _load_episode_frames(
                    root, int(ep_id), primary_key, image_size, fps,
                )
                err_hint = (
                    f"Failed to decode video for episode {ep_id}, key {primary_key}"
                )
                err_sugg = ["Verify opencv-python is installed; check video file naming"]
            if frames is None:
                return OperationResult(
                    success=False,
                    error=err_hint,
                    suggestions=err_sugg,
                )

        episodes_data.append(pack_episode(
            actions=actions,
            states=states,
            frames=frames,
            rewards=rewards,
            dones=dones,
            episode_id=int(ep_id),
        ))

    # Feature WIDTH, not key count. A single LeRobot key (e.g. "observation.state")
    # holds the full vector (12-dim for SO-101), so len(state_keys)==1 would
    # mislabel state_dim as 1. Read the real last-axis width from the packed arrays;
    # fall back to key count only when no episodes were written.
    _first = episodes_data[0] if episodes_data else {}
    _fa = _first.get("actions")
    _fs = _first.get("states")
    _action_dim = int(_fa.shape[-1]) if getattr(_fa, "ndim", 0) >= 1 else len(action_keys)
    _state_dim = (
        int(_fs.shape[-1]) if getattr(_fs, "ndim", 0) >= 1
        else (len(state_keys) if state_keys else 0)
    )

    write_meta = {
        "image_size": list(image_size),
        "fps": fps,
        "action_dim": _action_dim,
        "state_dim": _state_dim,
        "n_episodes": len(episodes_data),
        "action_keys": action_keys,
        "state_keys": state_keys,
        "image_keys": image_keys,
        "source_dataset": str(root),
        "normalize_actions": normalize_actions,
        "action_mean": action_mean.tolist() if action_mean is not None else None,
        "action_std": action_std.tolist() if action_std is not None else None,
        "window_size": window_size,
        "stride": stride if window_size else None,
    }

    if output_format == "hdf5":
        write_result = _write_hdf5(output_path, episodes_data, write_meta, window_size, stride)
    elif output_format == "npz":
        write_result = _write_npz(output_path, episodes_data, write_meta)
    else:  # webdataset
        write_result = _write_webdataset(
            output_path, episodes_data, write_meta, window_size, stride, shard_size,
        )

    if not write_result.success:
        return write_result

    return OperationResult(
        success=True,
        data={
            "episodes_written": len(episodes_data),
            "total_frames": sum(len(e["actions"]) for e in episodes_data),
            "output_path": str(output_path),
            "output_format": output_format,
            "image_size": list(image_size),
            "action_dim": _action_dim,
            "state_dim": _state_dim,
            "windows_emitted": write_result.data.get("windows_emitted") if write_result.data else None,
        },
    )


# ---------------------------------------------------------------------------
# Episode packing & windowing
# ---------------------------------------------------------------------------

def pack_episode(
    actions,
    states=None,
    frames=None,
    rewards=None,
    dones=None,
    episode_id: int = 0,
) -> Dict[str, Any]:
    """Bundle an episode's per-frame arrays into the canonical dict layout."""
    ep: Dict[str, Any] = {"episode_id": int(episode_id), "actions": actions}
    if states is not None:
        ep["states"] = states
    if frames is not None:
        ep["frames"] = frames
    if rewards is not None:
        ep["rewards"] = rewards
    if dones is not None:
        ep["dones"] = dones
    return ep


def slice_windows(arr, window_size: int, stride: int = 1):
    """
    Slide a (T, ...) array into (W, window_size, ...) windows.

    Returns (windowed_array, start_indices). When numpy is missing this
    is a no-op returning (arr, []).
    """
    try:
        import numpy as np
    except ImportError:
        return arr, []

    if window_size <= 0 or stride <= 0:
        return arr, []

    arr = np.asarray(arr)
    T = arr.shape[0]
    if T < window_size:
        return np.empty((0,) + (window_size,) + arr.shape[1:], dtype=arr.dtype), []

    starts = list(range(0, T - window_size + 1, stride))
    windowed = np.stack([arr[s : s + window_size] for s in starts], axis=0)
    return windowed, starts


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_episode_frames(
    dataset_path: str,
    episode_idx: int,
    image_key: Optional[str] = None,
    target_size: Tuple[int, int] = _DEFAULT_IMAGE_SIZE,
    fps: int = 30,
) -> OperationResult:
    """Decode a single episode's MP4 frames, resize to target_size."""
    root = Path(dataset_path)
    if image_key is None:
        candidates = [p.name for p in (root / "videos").glob("*") if p.is_dir()]
        if not candidates:
            return OperationResult(success=False, error="No camera dirs under videos/")
        image_key = candidates[0]

    frames = _load_episode_frames(root, episode_idx, image_key, target_size, fps)
    if frames is None:
        return OperationResult(
            success=False,
            error=f"Could not decode frames for episode {episode_idx}, key {image_key}",
        )

    return OperationResult(
        success=True,
        data={
            "shape": list(frames.shape),
            "dtype": str(frames.dtype),
            "image_key": image_key,
            "n_frames": int(frames.shape[0]),
        },
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(output_path: str) -> OperationResult:
    """Sanity-check shapes and dtypes of a converter output."""
    p = Path(output_path)
    if not p.exists():
        return OperationResult(success=False, error=f"Output not found: {output_path}")

    try:
        import numpy as np
    except ImportError:
        return OperationResult(success=False, error="numpy required for validation")

    if p.is_file() and p.suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError:
            return OperationResult(success=False, error="h5py required to validate HDF5")

        with h5py.File(p, "r") as f:
            n_eps = len(f.get("episodes", {}))
            sample = next(iter(f["episodes"].values())) if n_eps else None
            shapes = {k: list(v.shape) for k, v in sample.items()} if sample else {}
            attrs = dict(f.attrs)
            for k in list(attrs):
                v = attrs[k]
                if isinstance(v, np.ndarray):
                    attrs[k] = v.tolist()
                elif isinstance(v, bytes):
                    attrs[k] = v.decode("utf-8", errors="replace")
        return OperationResult(
            success=True,
            data={
                "format": "hdf5",
                "n_episodes": n_eps,
                "sample_shapes": shapes,
                "attrs": attrs,
            },
        )

    if p.is_dir():
        npz_files = sorted(p.glob("ep_*.npz"))
        tar_files = sorted(p.glob("*.tar"))
        if npz_files:
            sample = np.load(npz_files[0])
            return OperationResult(
                success=True,
                data={
                    "format": "npz",
                    "n_episodes": len(npz_files),
                    "sample_keys": list(sample.keys()),
                    "sample_shapes": {k: list(sample[k].shape) for k in sample.keys()},
                },
            )
        if tar_files:
            return OperationResult(
                success=True,
                data={"format": "webdataset", "n_shards": len(tar_files)},
            )

    return OperationResult(success=False, error=f"Unrecognized output layout at {output_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _columns_to_float_array(df, keys, dtype="float32"):
    """Convert one or more columns to a (N, D) float ndarray.

    Handles both LeRobotDataset v2.x layout (one scalar column per feature
    dim, e.g. ``action.0``, ``action.1``) and v3.0 layout (a single column
    where each row is a length-D ``np.ndarray``).

    Args:
        df: pandas DataFrame (or sub-DataFrame).
        keys: List of column names. May be length-1 (v3.0) or length-D (v2.x).
        dtype: NumPy dtype string.

    Returns:
        np.ndarray of shape (N, D).
    """
    import numpy as np

    if not keys:
        return np.empty((len(df), 0), dtype=dtype)

    # Detect v3.0 single-column-of-arrays layout: one key, object dtype, first
    # element is array-like with ndim >= 1.
    if len(keys) == 1:
        col = df[keys[0]]
        sample = col.iloc[0] if len(col) else None
        if hasattr(sample, "ndim") and getattr(sample, "ndim", 0) >= 1:
            return np.stack(col.to_list()).astype(dtype)
        if isinstance(sample, (list, tuple)):
            return np.stack([np.asarray(x) for x in col.to_list()]).astype(dtype)

    # v2.x layout: D scalar columns. Falls back to (N, D) ndarray.
    return df[keys].to_numpy().astype(dtype)


def _autodetect_columns(columns: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """Return (action_keys, state_keys, other_keys)."""
    action_keys = sorted(c for c in columns if c.startswith("action.") or c == "action")
    state_keys = sorted(
        c for c in columns
        if c.startswith("observation.state.") or c == "observation.state"
    )
    other = [
        c for c in columns
        if c not in action_keys and c not in state_keys and c not in _NON_FEATURE_COLUMNS
    ]
    return action_keys, state_keys, other


def _load_episode_frames_from_parquet(
    ep_df,
    image_key: str,
    target_size: Tuple[int, int],
):
    """Decode an episode's frames from inline parquet bytes (dtype=image).

    Uses Pillow + numpy only (no cv2 dep) — keeps this path runnable in any
    pixi env without opencv-python installed.
    """
    import numpy as np
    from PIL import Image

    target_h, target_w = target_size
    frames = []
    for cell in ep_df[image_key]:
        if isinstance(cell, dict):
            b = cell.get("bytes")
        elif isinstance(cell, (bytes, bytearray)):
            b = bytes(cell)
        else:
            raise ValueError(
                f"Unexpected cell type for image column {image_key!r}: {type(cell)}"
            )
        if not b:
            raise ValueError(
                f"Empty image bytes for column {image_key!r} in episode row"
            )
        img = Image.open(io.BytesIO(b)).convert("RGB")
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.BILINEAR)
        frames.append(np.asarray(img))

    if not frames:
        return None
    return np.stack(frames).astype("uint8")


def _load_episode_frames(
    root: Path,
    ep_id: int,
    image_key: str,
    target_size: Tuple[int, int],
    fps: int,
):
    """Decode one episode's MP4 from videos/<image_key>/.../*.mp4 and resize."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    target_h, target_w = target_size
    cam_dir = root / "videos" / image_key
    if not cam_dir.exists():
        return None

    candidates: List[Path] = []
    for chunk in cam_dir.iterdir() if cam_dir.is_dir() else []:
        if chunk.is_dir():
            candidates.extend(sorted(chunk.glob("*.mp4")))
        elif chunk.suffix == ".mp4":
            candidates.append(chunk)

    patterns = (
        f"episode_{ep_id:06d}.mp4",
        f"ep_{ep_id:06d}.mp4",
        f"file-{ep_id:06d}.mp4",
        f"episode_{ep_id}.mp4",
    )
    chosen = next((c for c in candidates if c.name in patterns), None)
    if chosen is None and candidates:
        # v3.0 stores many episodes per file; fall back to the only/first shard
        # and let the caller slice externally if needed.
        chosen = candidates[0]
    if chosen is None:
        return None

    cap = cv2.VideoCapture(str(chosen))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[0], frame.shape[1]) != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    if not frames:
        return None
    return np.stack(frames).astype("uint8")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_hdf5(
    output_path: str,
    episodes: List[Dict[str, Any]],
    meta: Dict[str, Any],
    window_size: Optional[int],
    stride: int,
) -> OperationResult:
    try:
        import h5py
        import numpy as np
    except ImportError:
        return OperationResult(success=False, error="h5py + numpy required for HDF5 output")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as f:
        ep_grp = f.create_group("episodes")
        for ep in episodes:
            g = ep_grp.create_group(f"ep_{ep['episode_id']:06d}")
            for key in ("actions", "states", "frames", "rewards", "dones"):
                if key in ep and ep[key] is not None:
                    g.create_dataset(key, data=ep[key], compression="gzip")

        windows_emitted = 0
        if window_size:
            wgrp = f.create_group("windows")
            all_frames, all_actions, all_states, all_idx = [], [], [], []
            for ep in episodes:
                if "frames" not in ep:
                    continue
                fw, starts = slice_windows(ep["frames"], window_size, stride)
                if len(starts) == 0:
                    continue
                aw, _ = slice_windows(ep["actions"], window_size, stride)
                all_frames.append(fw)
                all_actions.append(aw)
                if "states" in ep:
                    sw, _ = slice_windows(ep["states"], window_size, stride)
                    all_states.append(sw)
                all_idx.extend((ep["episode_id"], s) for s in starts)
            if all_frames:
                wgrp.create_dataset("frames", data=np.concatenate(all_frames), compression="gzip")
                wgrp.create_dataset("actions", data=np.concatenate(all_actions))
                if all_states:
                    wgrp.create_dataset("states", data=np.concatenate(all_states))
                wgrp.create_dataset("window_index", data=np.array(all_idx, dtype="int32"))
                windows_emitted = len(all_idx)

        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, list) and v and not isinstance(v[0], (int, float, str, bool)):
                continue
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = json.dumps(v)

    return OperationResult(success=True, data={"windows_emitted": windows_emitted})


def _write_npz(
    output_path: str,
    episodes: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> OperationResult:
    try:
        import numpy as np
    except ImportError:
        return OperationResult(success=False, error="numpy required for npz output")

    out = Path(output_path)
    if out.exists() and out.is_file():
        return OperationResult(
            success=False,
            error="output_path must be a directory for npz format",
        )
    out.mkdir(parents=True, exist_ok=True)

    for ep in episodes:
        ep_clean = {k: v for k, v in ep.items() if k != "episode_id" and v is not None}
        np.savez_compressed(out / f"ep_{ep['episode_id']:06d}.npz", **ep_clean)

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return OperationResult(success=True, data={"windows_emitted": 0})


def _write_webdataset(
    output_path: str,
    episodes: List[Dict[str, Any]],
    meta: Dict[str, Any],
    window_size: Optional[int],
    stride: int,
    shard_size: int,
) -> OperationResult:
    try:
        import numpy as np
    except ImportError:
        return OperationResult(success=False, error="numpy required for webdataset output")

    import tarfile

    out = Path(output_path)
    if out.exists() and out.is_file():
        return OperationResult(
            success=False,
            error="output_path must be a directory for webdataset format",
        )
    out.mkdir(parents=True, exist_ok=True)

    def _emit_samples():
        for ep in episodes:
            if window_size and "frames" in ep:
                fw, starts = slice_windows(ep["frames"], window_size, stride)
                aw, _ = slice_windows(ep["actions"], window_size, stride)
                sw, _ = (slice_windows(ep["states"], window_size, stride)
                         if "states" in ep else (None, None))
                for i, start in enumerate(starts):
                    sample = {
                        "key": f"ep_{ep['episode_id']:06d}_w_{start:06d}",
                        "frames.npy": fw[i],
                        "actions.npy": aw[i],
                    }
                    if sw is not None:
                        sample["states.npy"] = sw[i]
                    yield sample
            else:
                sample = {
                    "key": f"ep_{ep['episode_id']:06d}",
                    "actions.npy": ep["actions"],
                }
                for k in ("frames", "states", "rewards", "dones"):
                    if k in ep and ep[k] is not None:
                        sample[f"{k}.npy"] = ep[k]
                yield sample

    samples = list(_emit_samples())
    n_shards = (len(samples) + shard_size - 1) // shard_size if samples else 0

    for shard_idx in range(n_shards):
        shard_path = out / f"shard-{shard_idx:06d}.tar"
        with tarfile.open(shard_path, "w") as tar:
            for sample in samples[shard_idx * shard_size : (shard_idx + 1) * shard_size]:
                key = sample.pop("key")
                for fname, arr in sample.items():
                    buf = io.BytesIO()
                    np.save(buf, arr)
                    info = tarfile.TarInfo(f"{key}.{fname}")
                    info.size = buf.tell()
                    buf.seek(0)
                    tar.addfile(info, buf)

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return OperationResult(success=True, data={"windows_emitted": len(samples)})
