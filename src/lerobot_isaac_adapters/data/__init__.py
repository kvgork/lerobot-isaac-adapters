"""In-RAM dataset caching utilities for LeRobot + world-model training.

See `plans/2026-05-15-dataloader-gpu-decode-plan.md` (approach A) for
rationale. The wrapper here is intentionally generic so it composes with
both LeRobotDataset (PNG/JPEG inline in parquet) and HDF5-based
world-model datasets.
"""

from lerobot_isaac_adapters.data.cached_dataset import CachedDatasetWrapper

__all__ = ["CachedDatasetWrapper"]
