"""Cached-dataset entry point for lerobot policy training.

Imports lerobot's ``lerobot_train`` CLI and monkey-patches its
``make_dataset`` call so the returned LeRobotDataset is wrapped in
``CachedDatasetWrapper`` before training begins. All other CLI flags
are forwarded unchanged via ``sys.argv``.

Why a separate module
---------------------
The lerobot training entry imports ``make_dataset`` into its module
namespace at import time::

    from lerobot.datasets.factory import make_dataset

so patching ``lerobot.datasets.factory.make_dataset`` *after* import is
a no-op for that script. We import the script, replace the *local*
binding inside ``lerobot.scripts.lerobot_train``, and only then call
``main()``.

The cli_train_cached entry runs as a normal lerobot-train subprocess.
The adapter's ``policy_lerobot.run()`` swaps the command string between
the two entry points based on the ``--cache-frames`` flag.

Environment knobs
-----------------
- ``LEROBOT_ISAAC_CACHE_RAM_GB``: float, default 8.0. Hard ceiling on
  the cache. Raises ``MemoryError`` mid-warmup when exceeded.
- ``LEROBOT_ISAAC_CACHE_PROGRESS_EVERY``: int, default 500. Stderr
  progress every N rows. ``0`` disables.
- ``LEROBOT_ISAAC_CACHE_DISK_DIR``: Path, default
  ``<workspace>/outputs/cache_storage`` (when CWD looks like a workspace
  with that layout) or ``None`` (disable). On match, the wrapper writes
  the post-warmup cache to ``<dir>/<signature>.pt`` and reloads it on
  subsequent runs in ~20 s instead of re-warming.

Usage
-----
::

    python -m lerobot_isaac_adapters.cli_train_cached \\
        --policy.type=smolvla \\
        --dataset.repo_id=kvgork/so101-pickplace1 \\
        --batch_size=4 \\
        ...

CLI args are forwarded to lerobot's parser verbatim — this module owns
no flags of its own.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def _patched_make_dataset_factory(orig_make_dataset):
    """Return a ``make_dataset`` replacement that wraps the result in the cache."""
    from lerobot_isaac_adapters.data import CachedDatasetWrapper

    max_ram_gb = float(os.environ.get("LEROBOT_ISAAC_CACHE_RAM_GB", "8.0"))
    progress_every = int(os.environ.get("LEROBOT_ISAAC_CACHE_PROGRESS_EVERY", "500"))

    # Disk cache: opt-in via env var. When unset, default to
    # `<cwd>/outputs/cache_storage` if that path is creatable (handy for
    # the lerobot-isaac-training workspace), otherwise disable.
    raw_disk_dir = os.environ.get("LEROBOT_ISAAC_CACHE_DISK_DIR")
    if raw_disk_dir is None:
        from pathlib import Path

        candidate = Path.cwd() / "outputs" / "cache_storage"
        cache_disk_dir = candidate  # mkdir runs lazily in resolve_cache_path
    elif raw_disk_dir.strip().lower() in ("", "none", "off", "0", "false"):
        cache_disk_dir = None
    else:
        from pathlib import Path

        cache_disk_dir = Path(raw_disk_dir)

    def _patched_make_dataset(cfg):
        ds = orig_make_dataset(cfg)
        logger.info(
            "[cli_train_cached] wrapping %s with CachedDatasetWrapper "
            "(max_ram_gb=%.1f, disk_dir=%s)",
            type(ds).__name__,
            max_ram_gb,
            cache_disk_dir,
        )
        print(
            f"[cli_train_cached] wrapping {type(ds).__name__} with "
            f"CachedDatasetWrapper (max_ram_gb={max_ram_gb:.1f}, "
            f"disk_dir={cache_disk_dir})",
            flush=True,
        )
        return CachedDatasetWrapper(
            ds,
            max_ram_gb=max_ram_gb,
            progress_every=progress_every,
            cache_disk_dir=cache_disk_dir,
        )

    _patched_make_dataset.__wrapped__ = orig_make_dataset  # type: ignore[attr-defined]
    return _patched_make_dataset


def main() -> int:
    """Patch make_dataset then dispatch to lerobot's training main."""
    try:
        # Import the lerobot training script. After this import, the
        # script's own `make_dataset` binding is the bound name we have
        # to override — patching only `lerobot.datasets.factory.make_dataset`
        # is too late (lerobot_train already imported it).
        from lerobot.scripts import lerobot_train as _lerobot_train_module
    except ImportError as exc:
        print(
            f"[cli_train_cached] ERROR: lerobot not importable: {exc}\n"
            "Install LeRobot 0.5+: pip install lerobot",
            file=sys.stderr,
        )
        return 127

    orig = getattr(_lerobot_train_module, "make_dataset", None)
    if orig is None:
        print(
            "[cli_train_cached] ERROR: lerobot.scripts.lerobot_train has no "
            "make_dataset symbol. Upstream API may have changed; falling "
            "back to the uncached path is recommended.",
            file=sys.stderr,
        )
        return 2

    _lerobot_train_module.make_dataset = _patched_make_dataset_factory(orig)
    # Belt-and-suspenders: also patch the canonical location so any other
    # call sites pick up the wrapped version.
    try:
        from lerobot.datasets import factory as _factory_module

        _factory_module.make_dataset = _lerobot_train_module.make_dataset
    except ImportError:
        pass

    return _lerobot_train_module.main()


if __name__ == "__main__":
    sys.exit(main())
