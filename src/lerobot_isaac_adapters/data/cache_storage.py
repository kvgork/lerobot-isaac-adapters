"""On-disk persistence for ``CachedDatasetWrapper``.

After the first warmup, dump the uint8-compressed cache to a single
``torch.save`` blob. Subsequent runs with a matching signature load it
in ~20 s (SSD read at ~350 MB/s for 6.9 GB) instead of re-decoding for
~15 min. See ``plans/2026-05-15-cache-pickle-to-disk-plan.md``.

On-disk format
--------------
A single ``torch.save({...}, path)`` payload containing:

    {
        "version": 1,
        "signature": "<sha1 hex>",
        "n_rows": int,
        "image_keys": list[str],
        "compress_keys": list[str],
        "cache": list[dict],
    }

Signature
---------
sha1 of ``"{repo_id}|{root}|{n_rows}|{sorted_image_keys}|{lerobot_version}"``.
Any change in any field invalidates the on-disk cache. The wrapper
then re-warms and overwrites the file.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1


class CacheSignatureMismatch(RuntimeError):
    """Raised when an on-disk cache exists but its signature doesn't match."""


def compute_signature(
    base: Any,
    n_rows: int,
    image_keys: list[str],
) -> str:
    """Stable sha1 over the inputs that materially affect the cache content.

    ``root`` is normalised via ``Path.resolve()`` so that relative paths
    passed through the lerobot-train CLI and absolute paths passed
    directly to LeRobotDataset produce the SAME signature for the same
    on-disk dataset.
    """
    repo_id = str(getattr(base, "repo_id", "")) or type(base).__name__
    raw_root = getattr(base, "root", "")
    if raw_root:
        try:
            root = str(Path(raw_root).resolve())
        except Exception:  # noqa: BLE001
            root = str(raw_root)
    else:
        root = ""
    try:
        import lerobot  # type: ignore[import-untyped]

        lerobot_version = getattr(lerobot, "__version__", "?")
    except ImportError:
        lerobot_version = "no-lerobot"

    payload = "|".join(
        [
            repo_id,
            root,
            str(int(n_rows)),
            ",".join(sorted(image_keys)),
            str(lerobot_version),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def resolve_cache_path(
    cache_disk_dir: Path | str | None,
    signature: str,
) -> Path | None:
    """Return ``<cache_disk_dir>/<signature>.pt`` or ``None`` when disabled."""
    if cache_disk_dir is None:
        return None
    p = Path(cache_disk_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{signature}.pt"


def save_cache(
    path: Path | str,
    *,
    signature: str,
    cache: list[dict],
    image_keys: list[str],
    compress_keys: list[str],
) -> None:
    """Persist the cache to ``path`` as a single ``torch.save`` blob."""
    import torch

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "signature": signature,
        "n_rows": len(cache),
        "image_keys": list(image_keys),
        "compress_keys": list(compress_keys),
        "cache": cache,
    }
    t0 = time.time()
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on POSIX
    msg = (
        f"[cache_storage] saved {len(cache)} rows to {path} "
        f"({path.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.1f}s"
    )
    logger.info(msg)
    print(msg, flush=True)


def load_cache(
    path: Path | str,
    *,
    expected_signature: str,
) -> tuple[list[dict], list[str], list[str]]:
    """Load + signature-check. Returns ``(cache, image_keys, compress_keys)``."""
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    t0 = time.time()
    payload: dict = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != CACHE_FORMAT_VERSION:
        raise CacheSignatureMismatch(
            f"cache version mismatch: file={payload.get('version')} "
            f"expected={CACHE_FORMAT_VERSION}"
        )
    if payload.get("signature") != expected_signature:
        raise CacheSignatureMismatch(
            f"signature mismatch: file={payload.get('signature')!r} "
            f"expected={expected_signature!r}"
        )
    cache = payload["cache"]
    image_keys = payload.get("image_keys", []) or []
    compress_keys = payload.get("compress_keys", []) or []
    msg = (
        f"[cache_storage] loaded {len(cache)} rows from {path} "
        f"({path.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.1f}s"
    )
    logger.info(msg)
    print(msg, flush=True)
    return cache, image_keys, compress_keys
