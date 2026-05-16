"""Generic in-RAM cache wrapper for any torch.utils.data.Dataset.

Pre-decodes every row at construction time. For float32-normalized image
tensors (the LeRobotDataset default), the cache stores **uint8** copies
and lazy-casts back to float32 on read. This is approach A1 from
`plans/2026-05-15-dataloader-gpu-decode-plan.md`:

  raw 480x640 float32 image  ≈ 3.7 MB/row × 7491 rows  → 27.6 GB
  same image as uint8        ≈ 0.9 MB/row × 7491 rows  →  6.9 GB

The 4× compression makes the so101-pickplace1 cache fit in the default
8 GB ceiling. The cast-back costs ~3 ms per ``__getitem__`` (versus
~40 ms PNG decode) — still a net win.

Design notes
------------
- **Generic.** Accepts any base that exposes ``__len__`` + ``__getitem__``
  returning a dict. Tested with LeRobotDataset and HDF5 fixtures.
- **Image detection.** A column is treated as an image iff its first row
  is a 3-D tensor whose first or last dim is 3 (NCHW or NHWC). Override
  by passing ``image_keys`` explicitly.
- **uint8 compression.** Only applies to float32 image tensors whose max
  value is ≤ 1.5 (i.e. normalized to [0,1]). Already-uint8 images pass
  through unchanged.
- **RAM ceiling.** Default 8 GB. Raises ``MemoryError`` mid-warmup when
  exceeded so callers fall back to approach B/D cleanly.
- **Attribute proxy.** ``__getattr__`` forwards to the wrapped base.
- **Picklable.** Cache is a list of dicts; safe to fork under DataLoader
  workers (CoW).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


def _row_bytes(row: dict) -> int:
    """Best-effort byte-count for a dataset row (torch tensors + numpy arrays)."""
    total = 0
    for v in row.values():
        if hasattr(v, "element_size") and hasattr(v, "numel"):
            try:
                total += int(v.element_size()) * int(v.numel())
                continue
            except Exception:  # noqa: BLE001
                pass
        if hasattr(v, "nbytes"):
            try:
                total += int(v.nbytes)
                continue
            except Exception:  # noqa: BLE001
                pass
    return total


def _detect_image_keys(row: dict) -> list[str]:
    """Pick column names whose first-row value looks like an image tensor.

    Heuristic: 3-D or 4-D tensor with a channel dim of size 3 anywhere
    in its shape AND a total element count > 1024. Catches both
    ``[3, H, W]`` (single frame) and ``[T, 3, H, W]`` (delta-timestamp
    stack) shapes. Also catches by-name fallback (``image`` /
    ``observation.image`` / ``rgb`` substring) when shape inspection is
    inconclusive — LeRobotDataset always names its image columns
    ``observation.images.<cam>`` by convention.
    """
    keys: list[str] = []
    for k, v in row.items():
        if not (hasattr(v, "shape") and hasattr(v, "ndim")):
            continue
        try:
            ndim = int(v.ndim)
            shape = tuple(int(s) for s in v.shape)
            numel = 1
            for s in shape:
                numel *= s
        except Exception:  # noqa: BLE001
            continue
        if ndim in (3, 4) and 3 in shape and numel > 1024:
            keys.append(k)
            continue
        # By-name fallback: lerobot convention is
        # `observation.images.<cam>` or anything containing 'image'/'rgb'.
        kl = str(k).lower()
        if ("image" in kl or "rgb" in kl) and numel > 1024:
            keys.append(k)
    return keys


def _is_normalized_float(t: Any) -> bool:
    """True iff ``t`` is a float tensor with values plausibly in [0,1]."""
    try:
        import torch

        if not isinstance(t, torch.Tensor):
            return False
        if t.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            return False
        m = float(t.max())
        return m <= 1.5
    except Exception:  # noqa: BLE001
        return False


class CachedDatasetWrapper:
    """Wrap a Dataset, materialising every sample in RAM at construction.

    Parameters
    ----------
    base:
        Any object exposing ``__len__`` and ``__getitem__``.
    max_ram_gb:
        Hard ceiling on cumulative cached size (10^9 bytes). Default 8.
    progress_every:
        Stderr log every N pre-cached rows. ``0`` disables. Default 500.
    indices:
        Optional iterable of base indices to cache (default: all).
    image_keys:
        Optional explicit list of image columns. Auto-detected from the
        first row when ``None``.
    warmup_workers:
        Number of torch DataLoader workers for the one-time warmup decode.
        ``0`` runs serially. Default ``4`` parallelises the PNG-decode
        cost and typically brings 7.5k-row warmup from ~30 min to ~5 min.
    cache_disk_dir:
        Optional directory where the warmup result is persisted as a
        ``torch.save`` blob keyed by a signature hash. On a subsequent
        run with matching signature the cache loads in <30s instead of
        re-warming. ``None`` (default) disables disk caching. See
        ``plans/2026-05-15-cache-pickle-to-disk-plan.md``.
    """

    def __init__(
        self,
        base: Any,
        *,
        max_ram_gb: float = 8.0,
        progress_every: int = 500,
        indices: Iterable[int] | None = None,
        image_keys: list[str] | None = None,
        warmup_workers: int = 4,
        cache_disk_dir: Any = None,
    ) -> None:
        self._warmup_workers = max(0, int(warmup_workers))
        self._cache_disk_dir = cache_disk_dir
        self.base = base
        self._max_bytes = int(max_ram_gb * 1e9)
        self._progress_every = int(progress_every)

        if indices is None:
            self._index_map = list(range(len(base)))
        else:
            self._index_map = list(indices)

        # Probe a first row to pick image keys + decide compression.
        if not self._index_map:
            self._image_keys: list[str] = []
            self._compress_keys: set[str] = set()
            self._cache: list[dict] = []
            self._cache_bytes = 0
            return

        sample = base[self._index_map[0]]

        # Log first-row schema so future detection failures are diagnosable.
        try:
            schema = []
            for k, v in sample.items():
                if hasattr(v, "shape") and hasattr(v, "dtype"):
                    schema.append(f"{k}:{getattr(v, 'dtype', '?')}{tuple(getattr(v, 'shape', ()))}")
                else:
                    schema.append(f"{k}:{type(v).__name__}")
            print(
                f"[CachedDatasetWrapper] first-row schema: {schema}",
                flush=True,
            )
        except Exception:  # noqa: BLE001
            pass

        self._image_keys = list(image_keys) if image_keys else _detect_image_keys(sample)
        self._compress_keys = {k for k in self._image_keys if _is_normalized_float(sample[k])}
        print(
            f"[CachedDatasetWrapper] detected image_keys={self._image_keys}, "
            f"compress_keys={sorted(self._compress_keys)}",
            flush=True,
        )

        if self._compress_keys:
            logger.info(
                "[CachedDatasetWrapper] uint8 compression enabled for keys: %s",
                sorted(self._compress_keys),
            )
            print(
                f"[CachedDatasetWrapper] uint8 compression enabled for "
                f"keys: {sorted(self._compress_keys)}",
                flush=True,
            )

        self._cache = []
        self._cache_bytes = 0
        # Try disk-cache hit first; fall back to warmup + dump.
        if not self._try_load_from_disk():
            self._warmup()
            self._try_save_to_disk()

    # ----------------------------------------------------------------- #
    # Disk-cache (signature-keyed)
    # ----------------------------------------------------------------- #

    def _disk_path(self):
        if self._cache_disk_dir is None:
            return None
        from lerobot_isaac_adapters.data.cache_storage import (
            compute_signature,
            resolve_cache_path,
        )

        signature = compute_signature(
            self.base, len(self._index_map), self._image_keys
        )
        return resolve_cache_path(self._cache_disk_dir, signature), signature

    def _try_load_from_disk(self) -> bool:
        resolved = self._disk_path()
        if resolved is None:
            return False
        path, signature = resolved
        if path is None or not path.exists():
            return False
        try:
            from lerobot_isaac_adapters.data.cache_storage import load_cache

            cache, img_keys, compress_keys = load_cache(
                path, expected_signature=signature
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"[CachedDatasetWrapper] disk-cache load failed ({exc}); will re-warm"
            logger.warning(msg)
            print(msg, flush=True)
            return False
        # Restore state. We trust the disk image_keys / compress_keys.
        self._cache = cache
        self._image_keys = list(img_keys) if img_keys else self._image_keys
        self._compress_keys = set(compress_keys) if compress_keys else self._compress_keys
        # Recompute cumulative byte count for diagnostics.
        self._cache_bytes = sum(_row_bytes(r) for r in self._cache)
        msg = (
            f"[CachedDatasetWrapper] disk-cache HIT: {len(self._cache)} rows, "
            f"{self._cache_bytes / 1e9:.2f} GB (skipped warmup)"
        )
        logger.info(msg)
        print(msg, flush=True)
        return True

    def _try_save_to_disk(self) -> None:
        resolved = self._disk_path()
        if resolved is None:
            return
        path, signature = resolved
        if path is None:
            return
        try:
            from lerobot_isaac_adapters.data.cache_storage import save_cache

            save_cache(
                path,
                signature=signature,
                cache=self._cache,
                image_keys=self._image_keys,
                compress_keys=sorted(self._compress_keys),
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"[CachedDatasetWrapper] disk-cache save failed ({exc}); keeping in-RAM only"
            logger.warning(msg)
            print(msg, flush=True)

    # ----------------------------------------------------------------- #
    # Compression helpers (per-row)
    # ----------------------------------------------------------------- #

    def _compress_row(self, row: dict) -> dict:
        """Return a shallow copy of ``row`` with image columns cast to uint8."""
        if not self._compress_keys:
            return row  # nothing to compress
        import torch

        out = dict(row)
        for k in self._compress_keys:
            t = row[k]
            if isinstance(t, torch.Tensor):
                # Multiply, clamp, cast — saves 4x RAM vs float32.
                out[k] = (t * 255.0).clamp(0.0, 255.0).to(torch.uint8).contiguous()
        return out

    def _decompress_row(self, row: dict) -> dict:
        """Cast cached uint8 image columns back to float32 / 255."""
        if not self._compress_keys:
            return row
        import torch

        out = dict(row)
        for k in self._compress_keys:
            t = row[k]
            if isinstance(t, torch.Tensor) and t.dtype == torch.uint8:
                out[k] = t.to(torch.float32) / 255.0
        return out

    # ----------------------------------------------------------------- #
    # Cache population
    # ----------------------------------------------------------------- #

    def _warmup(self) -> None:
        """Materialise every row in the index_map into the cache.

        Uses a torch DataLoader with ``warmup_workers`` workers when set
        > 0 — the per-row PNG decode parallelises cleanly. ``num_workers``
        captures the env's effective CPU count (clamped at 8) to avoid
        oversubscription on small-core machines.
        """
        t0 = time.time()
        n = len(self._index_map)

        if self._warmup_workers > 0:
            self._warmup_parallel(n)
        else:
            for i, base_idx in enumerate(self._index_map):
                row = self.base[base_idx]
                self._append(i, n, row)

        elapsed = time.time() - t0
        rate = n / elapsed if elapsed > 0 else float("inf")
        msg = (
            f"[CachedDatasetWrapper] warmup complete: {n} rows, "
            f"{self._cache_bytes / 1e9:.2f} GB cached in {elapsed:.1f}s "
            f"({rate:.0f} rows/s)"
        )
        logger.info(msg)
        print(msg, flush=True)

    def _warmup_parallel(self, n: int) -> None:
        """Pre-cache rows via a parallel DataLoader.

        The trick: a tiny ``_BaseSubsetDataset`` yields ``(i, base[idx])``
        pairs so we can reassemble them in original order without an
        extra sort pass.
        """
        try:
            from torch.utils.data import DataLoader, Dataset
        except ImportError as exc:
            print(
                f"[CachedDatasetWrapper] torch unavailable for parallel warmup ({exc}); "
                f"falling back to serial.",
                flush=True,
            )
            for i, base_idx in enumerate(self._index_map):
                row = self.base[base_idx]
                self._append(i, n, row)
            return

        index_map = self._index_map
        base = self.base

        class _Adapter(Dataset):
            def __len__(self_):  # noqa: N805
                return len(index_map)

            def __getitem__(self_, i):  # noqa: N805
                return i, base[index_map[i]]

        def _collate(batch):
            # Batch is just a list of (i, row) tuples — keep as-is.
            return batch

        loader = DataLoader(
            _Adapter(),
            batch_size=8,
            num_workers=self._warmup_workers,
            collate_fn=_collate,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=2,
        )

        # Pre-size the cache so out-of-order arrivals can land in place.
        self._cache = [None] * n  # type: ignore[list-item]
        seen = 0
        for batch in loader:
            for i, row in batch:
                compressed = self._compress_row(row)
                self._cache[int(i)] = compressed
                self._cache_bytes += _row_bytes(compressed)
                seen += 1
                if self._cache_bytes > self._max_bytes:
                    cached_gb = self._cache_bytes / 1e9
                    raise MemoryError(
                        f"CachedDatasetWrapper exceeded {self._max_bytes / 1e9:.1f}GB "
                        f"cap at sample {seen}/{n} ({cached_gb:.2f}GB used). "
                        f"Either raise the cap with --cache_ram_gb, drop a camera "
                        f"key, or switch to approach B/D from "
                        f"plans/2026-05-15-dataloader-gpu-decode-plan.md."
                    )
                if self._progress_every and seen % self._progress_every == 0:
                    pct = 100.0 * seen / n
                    msg = (
                        f"[CachedDatasetWrapper] preload {seen}/{n} ({pct:.0f}%) "
                        f"{self._cache_bytes / 1e9:.2f} GB"
                    )
                    logger.info(msg)
                    print(msg, flush=True)

    def _append(self, i: int, n: int, row: dict) -> None:
        """Serial-path helper: compress + append + ram-check + progress log."""
        compressed = self._compress_row(row)
        self._cache.append(compressed)
        self._cache_bytes += _row_bytes(compressed)
        if self._cache_bytes > self._max_bytes:
            cached_gb = self._cache_bytes / 1e9
            raise MemoryError(
                f"CachedDatasetWrapper exceeded {self._max_bytes / 1e9:.1f}GB "
                f"cap at sample {i + 1}/{n} ({cached_gb:.2f}GB used). "
                f"Either raise the cap with --cache_ram_gb, drop a camera "
                f"key, or switch to approach B/D from "
                f"plans/2026-05-15-dataloader-gpu-decode-plan.md."
            )
        if self._progress_every and (i + 1) % self._progress_every == 0:
            pct = 100.0 * (i + 1) / n
            msg = (
                f"[CachedDatasetWrapper] preload {i + 1}/{n} ({pct:.0f}%) "
                f"{self._cache_bytes / 1e9:.2f} GB"
            )
            logger.info(msg)
            print(msg, flush=True)

    # ----------------------------------------------------------------- #
    # Dataset interface
    # ----------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict:
        return self._decompress_row(self._cache[idx])

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __repr__(self) -> str:
        return (
            f"CachedDatasetWrapper(base={self.base!r}, "
            f"n={len(self._cache)}, ram={self._cache_bytes / 1e9:.2f}GB, "
            f"compressed_keys={sorted(self._compress_keys)})"
        )

    @property
    def cached_bytes(self) -> int:
        return self._cache_bytes
