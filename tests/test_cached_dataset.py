"""Unit tests for CachedDatasetWrapper.

Generic over any base ``Dataset``. Does not require lerobot / image codecs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot_isaac_adapters.data.cached_dataset import (
    CachedDatasetWrapper,
    _detect_image_keys,
    _is_normalized_float,
)


class _NormalizedFloatDataset:
    """Mock dataset returning float32 [0,1] image tensors (LeRobot default).

    Sets explicit ``repo_id`` / ``root`` so disk-cache signatures are
    stable across instances with the same n.
    """

    def __init__(self, n: int = 8, image_shape: tuple = (3, 64, 64)) -> None:
        rng = np.random.default_rng(seed=42)
        self._rows = []
        for i in range(n):
            img = rng.integers(0, 256, image_shape, dtype=np.uint8).astype(np.float32) / 255.0
            self._rows.append(
                {
                    "observation.images.d435_rgb": torch.from_numpy(img),
                    "observation.state": torch.from_numpy(
                        rng.standard_normal(6).astype(np.float32)
                    ),
                    "action": torch.from_numpy(rng.standard_normal(6).astype(np.float32)),
                    "idx": i,
                }
            )
        self.features = ("observation.images.d435_rgb", "observation.state", "action")
        self.num_episodes = 2
        # Signature inputs — explicit so the disk-cache hash is stable.
        self.repo_id = "test-fixture/normalized-float"
        self.root = ""

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


class _Uint8Dataset:
    """Mock dataset returning uint8 image tensors (no compression needed)."""

    def __init__(self, n: int = 4, image_shape: tuple = (3, 32, 32)) -> None:
        rng = np.random.default_rng(seed=7)
        self._rows = []
        for i in range(n):
            self._rows.append(
                {
                    "image": torch.from_numpy(
                        rng.integers(0, 256, image_shape, dtype=np.uint8)
                    ),
                    "state": torch.from_numpy(rng.standard_normal(4).astype(np.float32)),
                    "idx": i,
                }
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


# ------------------------------------------------------------------------- #
# Detection
# ------------------------------------------------------------------------- #


def test_detect_image_keys_finds_nchw_tensors() -> None:
    sample = _NormalizedFloatDataset(n=1)[0]
    keys = _detect_image_keys(sample)
    assert keys == ["observation.images.d435_rgb"]


def test_is_normalized_float_recognizes_unit_interval() -> None:
    t = torch.tensor([[[0.0, 0.5], [0.2, 1.0]]])
    assert _is_normalized_float(t) is True


def test_is_normalized_float_rejects_uint8() -> None:
    t = torch.zeros((3, 4, 4), dtype=torch.uint8)
    assert _is_normalized_float(t) is False


def test_is_normalized_float_rejects_unnormalized() -> None:
    t = torch.tensor([[[0.0, 255.0]]])  # max > 1.5
    assert _is_normalized_float(t) is False


# ------------------------------------------------------------------------- #
# Wrapper basics
# ------------------------------------------------------------------------- #


def test_wrapper_preserves_len() -> None:
    base = _NormalizedFloatDataset(n=8)
    cached = CachedDatasetWrapper(base, max_ram_gb=1.0, progress_every=0)
    assert len(cached) == 8


def test_wrapper_proxies_base_attrs() -> None:
    base = _NormalizedFloatDataset(n=4)
    cached = CachedDatasetWrapper(base, max_ram_gb=1.0, progress_every=0)
    assert cached.features == ("observation.images.d435_rgb", "observation.state", "action")
    assert cached.num_episodes == 2


def test_wrapper_supports_index_subset() -> None:
    base = _NormalizedFloatDataset(n=8)
    cached = CachedDatasetWrapper(
        base,
        max_ram_gb=1.0,
        progress_every=0,
        indices=[0, 2, 4],
    )
    assert len(cached) == 3
    # Compare via roundtrip-equivalence (uint8 compress + decompress)
    assert torch.allclose(
        cached[0]["observation.images.d435_rgb"],
        base[0]["observation.images.d435_rgb"],
        atol=1.0 / 255,
    )


# ------------------------------------------------------------------------- #
# Compression path
# ------------------------------------------------------------------------- #


def test_uint8_compression_engaged_on_float_images() -> None:
    base = _NormalizedFloatDataset(n=4)
    cached = CachedDatasetWrapper(base, max_ram_gb=1.0, progress_every=0)
    assert "observation.images.d435_rgb" in cached._compress_keys

    # Cache should store uint8 internally
    stored = cached._cache[0]["observation.images.d435_rgb"]
    assert stored.dtype == torch.uint8

    # __getitem__ should return float32 again (roundtripped)
    out = cached[0]["observation.images.d435_rgb"]
    assert out.dtype == torch.float32


def test_uint8_compression_skipped_on_uint8_dataset() -> None:
    base = _Uint8Dataset(n=4)
    cached = CachedDatasetWrapper(base, max_ram_gb=1.0, progress_every=0)
    # Image key detected, but compression set is empty because dtype is already uint8
    assert cached._image_keys == ["image"]
    assert cached._compress_keys == set()


def test_uint8_roundtrip_equivalence_within_quantization() -> None:
    """Float→uint8→float roundtrip must stay within 1/255 of the original."""
    base = _NormalizedFloatDataset(n=4)
    cached = CachedDatasetWrapper(base, max_ram_gb=1.0, progress_every=0)
    for i in range(4):
        orig = base[i]["observation.images.d435_rgb"]
        out = cached[i]["observation.images.d435_rgb"]
        assert orig.shape == out.shape
        assert out.dtype == torch.float32
        # Max quantization error after *255→clamp→/255 is bounded by 1/255.
        diff = (out - orig).abs().max().item()
        assert diff <= 1.0 / 255 + 1e-6, f"row {i}: diff {diff}"


def test_uint8_compression_shrinks_cache_4x() -> None:
    """uint8 cache must be roughly 1/4 of float32 raw size."""
    base = _NormalizedFloatDataset(n=4, image_shape=(3, 64, 64))
    cached = CachedDatasetWrapper(base, max_ram_gb=1.0, progress_every=0)
    # raw image bytes per row = 3*64*64*4 = 49152 (float32)
    # uint8 image bytes per row = 3*64*64 = 12288
    # 4 rows uint8 image alone = 49152 bytes; plus state+action ~ 80 bytes/row
    assert 30_000 < cached.cached_bytes < 60_000


# ------------------------------------------------------------------------- #
# RAM cap
# ------------------------------------------------------------------------- #


def test_wrapper_raises_when_ram_cap_exceeded() -> None:
    base = _NormalizedFloatDataset(n=4, image_shape=(3, 64, 64))
    with pytest.raises(MemoryError, match=r"exceeded"):
        CachedDatasetWrapper(base, max_ram_gb=1e-5, progress_every=0)


# ------------------------------------------------------------------------- #
# Warmup paths — serial vs parallel
# ------------------------------------------------------------------------- #


def test_serial_warmup_path_produces_correct_cache() -> None:
    base = _NormalizedFloatDataset(n=8)
    cached = CachedDatasetWrapper(
        base, max_ram_gb=1.0, progress_every=0, warmup_workers=0
    )
    assert len(cached) == 8
    for i in range(8):
        assert torch.allclose(
            cached[i]["observation.images.d435_rgb"],
            base[i]["observation.images.d435_rgb"],
            atol=1.0 / 255,
        )


def test_parallel_warmup_path_produces_correct_cache() -> None:
    base = _NormalizedFloatDataset(n=8)
    cached = CachedDatasetWrapper(
        base, max_ram_gb=1.0, progress_every=0, warmup_workers=2
    )
    assert len(cached) == 8
    # Parallel path uses DataLoader workers — verify order preserved.
    for i in range(8):
        assert torch.allclose(
            cached[i]["observation.images.d435_rgb"],
            base[i]["observation.images.d435_rgb"],
            atol=1.0 / 255,
        )


def test_parallel_warmup_matches_serial() -> None:
    """Both paths must produce byte-identical caches after roundtrip."""
    base = _NormalizedFloatDataset(n=12)
    serial = CachedDatasetWrapper(base, max_ram_gb=1.0, progress_every=0, warmup_workers=0)
    parallel = CachedDatasetWrapper(base, max_ram_gb=1.0, progress_every=0, warmup_workers=2)
    assert len(serial) == len(parallel) == 12
    for i in range(12):
        torch.testing.assert_close(
            serial[i]["observation.images.d435_rgb"],
            parallel[i]["observation.images.d435_rgb"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            serial[i]["observation.state"],
            parallel[i]["observation.state"],
        )


def test_parallel_warmup_preserves_index_order() -> None:
    """Out-of-order DataLoader arrivals must still write to the right slot."""
    base = _NormalizedFloatDataset(n=16)
    cached = CachedDatasetWrapper(
        base, max_ram_gb=1.0, progress_every=0, warmup_workers=4
    )
    for i in range(16):
        # idx field is the original sample index — must match position.
        assert cached[i]["idx"] == i


# ------------------------------------------------------------------------- #
# Disk cache (signature-keyed pickle-to-disk)
# ------------------------------------------------------------------------- #


def test_disk_cache_save_and_load_roundtrip(tmp_path) -> None:
    """First wrapper writes the .pt; second wrapper loads it without warmup."""
    base = _NormalizedFloatDataset(n=6)
    disk_dir = tmp_path / "cache_storage"

    # First instantiation: warmup + dump.
    w1 = CachedDatasetWrapper(
        base, max_ram_gb=1.0, progress_every=0, warmup_workers=0,
        cache_disk_dir=disk_dir,
    )
    assert len(w1) == 6
    pt_files = list(disk_dir.glob("*.pt"))
    assert len(pt_files) == 1, f"expected 1 disk cache file, got {pt_files}"

    # Second instantiation: load from disk; verify warmup was SKIPPED by
    # mocking the base to raise on access.
    class _ExplodingBase:
        # All metadata still available so signature can be computed.
        def __init__(self, src):
            self._first_row = src[0]
            self._n = len(src)
            self.repo_id = getattr(src, "repo_id", "")
            self.root = getattr(src, "root", "")
            self.access_count = 0

        def __len__(self):
            return self._n

        def __getitem__(self, idx):
            # The wrapper probes index_map[0] once to detect image keys,
            # then should hit the disk cache and never touch the base again.
            self.access_count += 1
            return self._first_row

    exploding = _ExplodingBase(base)
    w2 = CachedDatasetWrapper(
        exploding, max_ram_gb=1.0, progress_every=0, warmup_workers=0,
        cache_disk_dir=disk_dir,
    )
    assert len(w2) == 6
    # Detection probes the first row exactly once; warmup must not run.
    assert exploding.access_count == 1, (
        f"warmup should have been skipped; got {exploding.access_count} accesses"
    )

    # Roundtrip equivalence to original base.
    for i in range(6):
        assert torch.allclose(
            w2[i]["observation.images.d435_rgb"],
            base[i]["observation.images.d435_rgb"],
            atol=1.0 / 255,
        )


def test_disk_cache_signature_invalidates_on_row_count_change(tmp_path) -> None:
    """Signature mismatch must trigger a fresh warmup (not return stale rows)."""
    disk_dir = tmp_path / "cache_storage"

    # Warm with n=4 → file dumped for n=4 signature.
    base_a = _NormalizedFloatDataset(n=4)
    _ = CachedDatasetWrapper(
        base_a, max_ram_gb=1.0, progress_every=0, warmup_workers=0,
        cache_disk_dir=disk_dir,
    )
    n_after_first = len(list(disk_dir.glob("*.pt")))
    assert n_after_first == 1

    # New base with n=8 → different signature → fresh warmup + new file.
    base_b = _NormalizedFloatDataset(n=8)
    w = CachedDatasetWrapper(
        base_b, max_ram_gb=1.0, progress_every=0, warmup_workers=0,
        cache_disk_dir=disk_dir,
    )
    assert len(w) == 8
    n_after_second = len(list(disk_dir.glob("*.pt")))
    assert n_after_second == 2, f"expected 2 distinct .pt files, got {n_after_second}"


def test_disk_cache_disabled_when_dir_none(tmp_path) -> None:
    """cache_disk_dir=None means no file written, no load attempt."""
    base = _NormalizedFloatDataset(n=4)
    w = CachedDatasetWrapper(
        base, max_ram_gb=1.0, progress_every=0, warmup_workers=0,
        cache_disk_dir=None,
    )
    assert len(w) == 4
    # No spurious files should exist.
    assert not any(tmp_path.glob("*.pt"))


def test_disk_cache_corrupt_file_falls_back_to_warmup(tmp_path) -> None:
    """A .pt file that fails signature/load check must not block training."""
    from lerobot_isaac_adapters.data.cache_storage import compute_signature

    base = _NormalizedFloatDataset(n=4)
    disk_dir = tmp_path / "cache_storage"
    disk_dir.mkdir()
    sig = compute_signature(base, n_rows=4, image_keys=["observation.images.d435_rgb"])
    bad = disk_dir / f"{sig}.pt"
    bad.write_bytes(b"not-a-torch-blob")

    # Should warn, fall back to warmup, and overwrite the broken file.
    w = CachedDatasetWrapper(
        base, max_ram_gb=1.0, progress_every=0, warmup_workers=0,
        cache_disk_dir=disk_dir,
    )
    assert len(w) == 4
    # Overwrite should have replaced the garbage with a valid blob.
    assert bad.stat().st_size > len(b"not-a-torch-blob")
