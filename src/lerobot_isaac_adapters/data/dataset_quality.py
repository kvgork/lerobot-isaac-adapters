"""Dataset quality filtering — SAL (Spectral Arc Length) + TED (Trajectory-Envelope Distance).

Moved here from the ``lerobot_dataset_quality`` Claude-Code skill (2026-06-28), same rationale
as world_model_bridge: core data-pipeline code belongs in a versioned, testable package module,
not a skill (the skill forced quality.py into a two-tier sys.path-inject + subprocess reach).
The skill ``operations.py`` is now a thin shim re-exporting from this module. Public API
unchanged: ``compute_sal``, ``compute_ted``, ``score_dataset``, ``filter_dataset``, ``OperationResult``.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class OperationResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Core metric implementations
# ---------------------------------------------------------------------------

def compute_sal(trajectory: np.ndarray, fs: float = 1.0) -> float:
    """
    Compute Spectral Arc Length (SAL) for a 1-D trajectory.

    SAL measures smoothness in the frequency domain. Returns a value ≤ 0;
    closer to 0 means smoother. Based on Balasubramanian et al. and the
    RINSE paper (arXiv 2604.23000).

    Args:
        trajectory: 1-D array of joint positions or action values.
        fs: Sampling frequency in Hz (default 1.0 for frame-indexed data).

    Returns:
        SAL value (float, ≤ 0).
    """
    try:
        from scipy.fft import rfft, rfftfreq
    except ImportError:
        from numpy.fft import rfft, rfftfreq  # type: ignore[no-redef]

    traj = np.asarray(trajectory, dtype=float)
    N = len(traj)
    if N < 4:
        return 0.0

    amplitude = np.ptp(traj)
    if amplitude < 1e-8:
        return 0.0

    # Normalize to unit amplitude
    traj_norm = traj / amplitude

    freqs = rfftfreq(N, d=1.0 / fs)
    spectrum = np.abs(rfft(traj_norm)) * (2.0 / N)

    # Normalized frequency and magnitude
    omega_max = freqs[-1] if freqs[-1] > 0 else 1.0
    freqs_n = freqs / omega_max
    spectrum_n = spectrum / (spectrum[0] + 1e-12)

    # Arc length in (omega_n, X_n) space
    d_omega = np.diff(freqs_n)
    d_X = np.diff(spectrum_n)
    arc_elements = np.sqrt(d_omega**2 + d_X**2)
    sal = -float(np.sum(arc_elements))

    return sal


def compute_ted(episodes: List[np.ndarray]) -> List[float]:
    """
    Compute Trajectory-Envelope Distance (TED) for a list of episodes.

    TED is the normalized DTW distance between each episode and the
    median episode, measuring inter-episode consistency.

    Args:
        episodes: List of 2-D arrays, each shape (T_i, D) where D is
                  action/state dimensionality.

    Returns:
        List of TED values, one per episode (lower = more consistent).
    """
    if len(episodes) < 2:
        return [0.0] * len(episodes)

    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        # Fallback: use L2 distance without DTW warping
        return _ted_euclidean_fallback(episodes)

    # Use episode closest to mean length as reference
    lengths = [len(e) for e in episodes]
    ref_idx = int(np.argmin(np.abs(np.array(lengths) - np.median(lengths))))
    reference = episodes[ref_idx]

    teds = []
    for ep in episodes:
        # DTW via dynamic programming (O(T²) per episode)
        dist = _dtw_distance(ep, reference)
        # Normalize by combined length
        norm = len(ep) + len(reference)
        teds.append(dist / max(norm, 1))

    return teds


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute DTW distance between two multi-dimensional sequences."""
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _ted_euclidean_fallback(episodes: List[np.ndarray]) -> List[float]:
    """TED fallback using mean Euclidean distance (no DTW)."""
    max_len = max(len(e) for e in episodes)
    padded = []
    for ep in episodes:
        pad = np.zeros((max_len - len(ep), ep.shape[1]))
        padded.append(np.vstack([ep, pad]))
    stack = np.stack(padded)
    mean_ep = stack.mean(axis=0)
    return [float(np.linalg.norm(ep - mean_ep)) / max_len for ep in padded]


# ---------------------------------------------------------------------------
# Dataset-level operations
# ---------------------------------------------------------------------------

def _episode_array(group, keys: List[str]) -> np.ndarray:
    """Stack episode columns into a (T, dim) float array.

    Handles both layouts: lerobot v3 stores ``action`` / ``observation.state`` as a SINGLE
    array-valued column (each cell is a vector), while older datasets split them into per-dim
    scalar columns (``action_0`` ...). The v3 single-column case would break
    ``group[keys].values.astype(float)`` ("setting an array element with a sequence"), so detect
    array-valued cells and ``np.stack`` them instead.
    """
    if len(keys) == 1:
        col = group[keys[0]]
        first = col.iloc[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return np.stack([np.asarray(x, dtype=float) for x in col.to_numpy()])
    return group[keys].values.astype(float)


def score_dataset(
    dataset_path: str,
    action_keys: Optional[List[str]] = None,
    response_format: str = "concise",
) -> OperationResult:
    """
    Score all episodes in a LeRobotDataset by SAL and TED.

    Args:
        dataset_path: Path to LeRobotDataset root (contains data/ and meta/).
        action_keys: List of action column names to use. Auto-detected if None.
        response_format: "summary" | "concise" | "detailed"

    Returns:
        OperationResult with per-episode scores and distribution stats.
    """
    try:
        import pandas as pd
    except ImportError:
        return OperationResult(
            success=False,
            error="pandas not installed. Run: pip install pandas pyarrow",
            suggestions=["pip install pandas pyarrow"],
        )

    root = Path(dataset_path)
    if not root.exists():
        return OperationResult(
            success=False,
            error=f"Dataset not found: {dataset_path}",
            suggestions=["Check dataset path", "Run lerobot-record first"],
        )

    parquet_files = sorted(root.glob("data/**/*.parquet"))
    if not parquet_files:
        return OperationResult(
            success=False,
            error="No parquet files found in data/",
            suggestions=["Ensure dataset is in LeRobotDataset v3.0 format"],
        )

    # Load all data
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    # Auto-detect action columns
    if action_keys is None:
        action_keys = [c for c in df.columns if "action" in c.lower()]
    if not action_keys:
        action_keys = [c for c in df.columns if c not in ("timestamp", "frame_index",
                                                           "episode_index", "index",
                                                           "task_index", "next.done")]

    # Group by episode
    episode_col = "episode_index" if "episode_index" in df.columns else df.columns[0]
    episodes_df = df.groupby(episode_col)

    sal_scores: Dict[int, float] = {}
    ted_episodes: List[np.ndarray] = []
    episode_ids: List[int] = []
    episode_lengths: Dict[int, int] = {}

    for ep_id, group in episodes_df:
        actions = _episode_array(group, action_keys)
        # SAL: average across action dimensions
        sal_per_dim = [compute_sal(actions[:, d]) for d in range(actions.shape[1])]
        sal_scores[int(ep_id)] = float(np.mean(sal_per_dim))
        ted_episodes.append(actions)
        episode_ids.append(int(ep_id))
        episode_lengths[int(ep_id)] = len(group)

    ted_values = compute_ted(ted_episodes)
    ted_scores = {ep_id: ted for ep_id, ted in zip(episode_ids, ted_values)}

    # Composite rank score (lower = higher quality)
    sal_array = np.array([sal_scores[e] for e in episode_ids])
    ted_array = np.array([ted_scores[e] for e in episode_ids])

    sal_ranks = sal_array.argsort().argsort()  # rank: higher SAL (less negative) = better
    ted_ranks = (-ted_array).argsort().argsort()  # rank: lower TED = better
    composite_ranks = (sal_ranks + ted_ranks) / 2.0

    episode_data = [
        {
            "episode_id": ep_id,
            "sal": sal_scores[ep_id],
            "ted": ted_scores[ep_id],
            "composite_rank": float(composite_ranks[i]),
            "length": episode_lengths[ep_id],
        }
        for i, ep_id in enumerate(episode_ids)
    ]

    stats = {
        "total_episodes": len(episode_ids),
        "sal_mean": float(np.mean(sal_array)),
        "sal_p20": float(np.percentile(sal_array, 20)),
        "sal_p50": float(np.median(sal_array)),
        "ted_mean": float(np.mean(ted_array)),
        "ted_p80": float(np.percentile(ted_array, 80)),
        "action_keys_used": action_keys,
    }

    if response_format == "summary":
        return OperationResult(success=True, data=stats)

    if response_format == "concise":
        return OperationResult(
            success=True,
            data={**stats, "worst_10": sorted(episode_data, key=lambda x: x["composite_rank"])[-10:]},
        )

    return OperationResult(success=True, data={**stats, "episodes": episode_data})


def filter_dataset(
    dataset_path: str,
    output_path: str,
    filter_percentile: int = 20,
    strategy: str = "composite",
    dry_run: bool = False,
    response_format: str = "concise",
) -> OperationResult:
    """
    Filter low-quality episodes from a LeRobotDataset.

    Args:
        dataset_path: Input dataset root path.
        output_path: Output filtered dataset path.
        filter_percentile: Remove bottom N% episodes (default 20).
        strategy: "sal" | "ted" | "composite" (default "composite").
        dry_run: If True, only report what would be removed.
        response_format: "summary" | "concise" | "detailed"

    Returns:
        OperationResult with kept/removed episode counts and paths.
    """
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return OperationResult(
            success=False,
            error="pandas and pyarrow required. Run: pip install pandas pyarrow",
        )

    # Score first
    score_result = score_dataset(dataset_path, response_format="detailed")
    if not score_result.success:
        return score_result

    episodes: List[Dict] = score_result.data["episodes"]
    total = len(episodes)

    # Determine cutoff
    if strategy == "sal":
        values = [e["sal"] for e in episodes]
        threshold = np.percentile(values, filter_percentile)
        keep_mask = [e["sal"] >= threshold for e in episodes]
    elif strategy == "ted":
        values = [e["ted"] for e in episodes]
        threshold = np.percentile(values, 100 - filter_percentile)
        keep_mask = [e["ted"] <= threshold for e in episodes]
    else:  # composite
        values = [e["composite_rank"] for e in episodes]
        threshold = np.percentile(values, 100 - filter_percentile)
        keep_mask = [e["composite_rank"] <= threshold for e in episodes]

    kept_ids = {e["episode_id"] for e, keep in zip(episodes, keep_mask) if keep}
    removed_ids = {e["episode_id"] for e, keep in zip(episodes, keep_mask) if not keep}

    result_data = {
        "total": total,
        "kept": len(kept_ids),
        "removed": len(removed_ids),
        "kept_pct": round(100 * len(kept_ids) / total, 1),
        "strategy": strategy,
        "filter_percentile": filter_percentile,
        "removed_episode_ids": sorted(removed_ids),
    }

    if dry_run:
        return OperationResult(success=True, data={**result_data, "dry_run": True})

    # Write filtered dataset
    src = Path(dataset_path)
    dst = Path(output_path)
    if dst.exists():
        shutil.rmtree(dst)

    # Copy meta directory (info.json, stats.json, tasks.parquet)
    shutil.copytree(src / "meta", dst / "meta", dirs_exist_ok=True)

    # Filter parquet files
    (dst / "data").mkdir(parents=True, exist_ok=True)
    for pf in sorted(src.glob("data/**/*.parquet")):
        import pandas as pd
        df = pd.read_parquet(pf)
        ep_col = "episode_index" if "episode_index" in df.columns else df.columns[0]
        df_filtered = df[df[ep_col].isin(kept_ids)]
        if len(df_filtered) > 0:
            rel = pf.relative_to(src / "data")
            out_file = dst / "data" / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)
            df_filtered.to_parquet(out_file, index=False)

    # Copy video files for kept episodes only
    for vid_dir in (src / "videos").iterdir() if (src / "videos").exists() else []:
        dst_vid = dst / "videos" / vid_dir.name
        dst_vid.mkdir(parents=True, exist_ok=True)
        for chunk_dir in vid_dir.iterdir():
            for vid_file in chunk_dir.glob("*.mp4"):
                # Naming convention: episode_{id:06d}.mp4 or file-{id}.mp4
                # Keep only if episode id is in kept_ids
                try:
                    ep_id = int(vid_file.stem.split("-")[-1])
                except ValueError:
                    ep_id = -1
                if ep_id in kept_ids or ep_id == -1:
                    dst_chunk = dst_vid / chunk_dir.name
                    dst_chunk.mkdir(exist_ok=True)
                    shutil.copy2(vid_file, dst_chunk / vid_file.name)

    result_data["output_path"] = str(dst)

    if response_format == "summary":
        return OperationResult(success=True, data={
            "kept": result_data["kept"],
            "removed": result_data["removed"],
            "output_path": result_data["output_path"],
        })

    return OperationResult(success=True, data=result_data)
