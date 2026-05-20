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
- ``LEROBOT_ISAAC_USE_LORA``: ``"1"`` to enable LoRA monkey-patch on
  ``make_policy``. Requires ``LEROBOT_ISAAC_LORA_RANK``,
  ``LEROBOT_ISAAC_LORA_ALPHA``, ``LEROBOT_ISAAC_LORA_DROPOUT``, and
  ``LEROBOT_ISAAC_LORA_TARGET_MODULES`` to also be set.

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
    """Patch make_dataset (and optionally make_policy for LoRA) then dispatch to lerobot's training main."""
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

    # LoRA monkey-patch: when LEROBOT_ISAAC_USE_LORA=1, wrap the policy with
    # PEFT LoRA adapters at policy-construction time (same in-process approach
    # as the dataset cache patch above).
    if os.environ.get("LEROBOT_ISAAC_USE_LORA") == "1":
        import warnings

        from lerobot_isaac_adapters.targets._lora import (
            LoraSpec,
            wrap_smolvla_policy,
        )

        # Locate make_policy in lerobot.scripts.train (preferred) or fall back
        # to any make_policy attribute on lerobot.scripts.lerobot_train.
        _train_module_for_policy = None
        _orig_make_policy = None

        try:
            import lerobot.scripts.train as _lerobot_scripts_train

            _orig_make_policy = getattr(_lerobot_scripts_train, "make_policy", None)
            if _orig_make_policy is not None:
                _train_module_for_policy = _lerobot_scripts_train
        except ImportError:
            pass

        if _orig_make_policy is None:
            # Fallback: search lerobot.scripts.lerobot_train for make_policy.
            _orig_make_policy = getattr(_lerobot_train_module, "make_policy", None)
            if _orig_make_policy is not None:
                _train_module_for_policy = _lerobot_train_module
            else:
                warnings.warn(
                    "[cli_train_cached] LEROBOT_ISAAC_USE_LORA=1 but could not "
                    "find make_policy in lerobot.scripts.train or "
                    "lerobot.scripts.lerobot_train. LoRA wrap will NOT be applied. "
                    "Check that lerobot 0.5+ is installed.",
                    stacklevel=1,
                )

        if _orig_make_policy is not None and _train_module_for_policy is not None:
            def _patched_make_policy(*a, **kw):
                policy = _orig_make_policy(*a, **kw)
                spec = LoraSpec.from_args(
                    rank=int(os.environ["LEROBOT_ISAAC_LORA_RANK"]),
                    alpha=int(os.environ["LEROBOT_ISAAC_LORA_ALPHA"]),
                    dropout=float(os.environ["LEROBOT_ISAAC_LORA_DROPOUT"]),
                    target_modules_spec=os.environ["LEROBOT_ISAAC_LORA_TARGET_MODULES"],
                )
                return wrap_smolvla_policy(policy, spec)

            setattr(_train_module_for_policy, "make_policy", _patched_make_policy)

    return _lerobot_train_module.main()


if __name__ == "__main__":
    sys.exit(main())
