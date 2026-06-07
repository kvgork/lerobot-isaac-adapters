# lerobot-isaac-adapters — Package Orientation

**Phase:** 2 (training adapter)
**Role:** Single entrypoint for all training runs. Dispatches by `--target_arch` to policy
(smolvla/act/diffusion) or world-model (dreamerv3/le_world_model) backends.
**Status:** All three backends wired with subprocess dispatch + metric extraction.
Dry-run smoke passes for `smolvla` / `act` / `diffusion` / `dreamerv3` / `le_world_model`.
Real training requires `lerobot-train` (policy) or `sheeprl` (DreamerV3) installed.

---

## What This Package Does

Three responsibilities:
1. **Dispatch router** (`train.py`) — parses `--target_arch` and delegates to the
   correct `targets/*.py` module. All args forwarded uniformly regardless of backend.
2. **Metric emitter** (`metric_extractor.py`) — canonical stdout format `name=value`
   consumed by `autoresearch-ml-executor-worker`. Every backend must use this module.
3. **Data recorder** (`isaac_data_recorder.py`) — records Isaac Lab rollout episodes
   into LeRobotDataset Parquet format with the standard SO-101 column schema.

---

## Public API

```python
from lerobot_isaac_adapters import metric_extractor  # emit() + MetricEmitter
from lerobot_isaac_adapters.train import main as train_main  # CLI entry
from lerobot_isaac_adapters import isaac_data_recorder  # record_episodes()
```

Note: `train` is NOT eagerly re-exported from the package `__init__.py`.
This keeps `python -m lerobot_isaac_adapters.train` from triggering a
`RuntimeWarning: <module> found in sys.modules` on every invocation.

Console script: `lerobot-isaac-train` → `lerobot_isaac_adapters.train:main`

---

## Key Files

| File | Purpose |
|------|---------|
| `src/lerobot_isaac_adapters/train.py` | argparse + dispatch router |
| `src/lerobot_isaac_adapters/targets/policy_lerobot.py` | smolvla/act/diffusion — spawns `lerobot-train` subprocess. `--successes_only` reads the recorder's `meta/episode_labels.json` sidecar (inline JSON; NO `robot_data_recorder` import — coupling rule) and forwards successful indices via `--dataset.episodes`. Single local dataset only. |
| `src/lerobot_isaac_adapters/targets/wm_dreamerv3.py` | DreamerV3 — `lerobot_world_model_bridge` Parquet→HDF5 (64×64) + `sheeprl exp=dreamer_v3` subprocess; parses `recon_loss=` |
| `src/lerobot_isaac_adapters/targets/wm_leworldmodel.py` | HF LeWorldModel — bridge Parquet→HDF5 (96×96, win=16) + `python -m lerobot.scripts.train_world_model`; parses `pred_loss=` |
| `src/lerobot_isaac_adapters/metric_extractor.py` | canonical stdout metric emitter |
| `src/lerobot_isaac_adapters/isaac_data_recorder.py` | Isaac rollout -> LeRobotDataset Parquet |
| `tests/test_train_argparse.py` | argparse smoke tests (all archs, dry_run) |
| `tests/test_metric_extractor.py` | metric format contract tests |
| `tests/test_targets_subprocess.py` | subprocess wiring tests |

---

## Metric Contract

Every eval step emits exactly: `<name>=<float>` on stdout.
Parsed by `autoresearch-ml-executor-worker` regex: `(\w+)[=:\s]+([0-9.eE+-]+)`.

| arch | metric name | direction |
|------|-------------|-----------|
| smolvla/act/diffusion | `pc_success` | maximize |
| dreamerv3 | `recon_loss` | minimize |
| le_world_model | `pred_loss` | minimize |

---

## Coupling Rules (plan §11.6)

- Does NOT import `lerobot-isaac-env` at module load; `isaac_data_recorder.py`
  soft-imports `isaaclab` and `lerobot` only at call time.
- May import `lerobot-isaac-configs` for default YAML config paths (optional dep).
- `lerobot`, `sheeprl`, `transformers` are soft-imported in `targets/*.py`.
- No imports from `lerobot-isaac-meta` (that would create a circular dep).

---

## Wiring a Backend

To wire a real backend, replace `raise NotImplementedError(...)` in the relevant
`targets/*.py` with the actual subprocess call or import. No changes to `train.py`
or `metric_extractor.py` needed.

Example stub → real for `wm_dreamerv3.py`:
```python
def run(args):
    # Old: raise NotImplementedError(...)
    # New:
    import subprocess, sys
    cmd = ["python", "-m", "dreamer", "--config", args.config, ...]
    proc = subprocess.run(cmd)
    return proc.returncode
```

---

## Source-of-Truth Paths (repo)

- World model bridge skill: `${CLAUDE_CODE_ROOT}/skills/lerobot_world_model_bridge/`
- Training orchestrator: `~/.claude/agents/orchestrators/lerobot-training-orchestrator.md`
- Autoresearch executor: `~/.claude/agents/workers/autoresearch-ml-executor-worker.md`
- LeRobot specialist: `~/.claude/agents/lerobot-specialist.md`

---

## Testing Notes

All tests pass without lerobot, sheeprl, Isaac Lab, or transformers installed.

- `test_train_argparse.py` — verifies all 5 archs accepted; `--dry_run` works; unknown arch rejected
- `test_metric_extractor.py` — `emit()` format, name validation, `MetricEmitter` prefix
- `test_targets_subprocess.py` — `policy_lerobot.run()` returns 127 when `lerobot-train` not found

---

## WM-Isaac sheeprl_plugin Pitfalls (GPU-verified 2026-05-31)

The DreamerV3-on-Isaac path (`sheeprl_plugin/isaac_env.py` + `scripts/_wm_isaac_entry.py`)
had a "training stall" that was really a **masked crash + Isaac teardown hang**:

- **`IsaacSO101Env.close()` must be a NO-OP on the backing env.** The backing
  `ManagerBasedRLEnv` is a process-global singleton (`_GLOBAL_BACKING_ISAAC_ENV`).
  sheeprl `dreamer_v3.py:765` calls `envs.close()` then `:767 test()`, which builds
  a fresh wrapper that reuses the singleton. Closing it deletes `.scene` → eval
  reset crashes `'ManagerBasedRLEnv' object has no attribute 'scene'`.
- **`_wm_isaac_entry.py` must `os._exit(code)`** after `run()`. Isaac's atexit
  `SimulationApp.close()` hangs in `render()`; without the hard exit a finished or
  crashed trial holds ~1.6 GB VRAM forever and looks frozen (`metric=-9999`).
- **`learning_starts ≥ per_rank_sequence_length (64)`** or the first grad update
  raises `ValueError: Cannot sample a sequence of length 64` (`buffers.py:432`).
  Keep `seq_len ≤ learning_starts < total_steps`. Prod default 1024 is fine.
- **Camera RGB is now wired (2026-05-31).** `make_env(enable_cameras=True)` adds the
  `d435_rgb` term (3,480,640); the wrapper (`camera_key="d435_rgb"`, default
  `enable_cameras=True`) resizes it to `image_size`² via `_resize_chw` before the
  CNN. Verified: `Loss/observation_loss≈85` on real frames (vs ~0 for the old
  all-zero RGB). Toggle off with `env.enable_cameras=False` for a state-only WM.
- **Diagnose with `faulthandler.dump_traceback_later()`**, not py-spy — host has
  `ptrace_scope=1` + no passwordless sudo, so cross-shell `py-spy dump` is blocked.

## Spinout

```bash
git subtree split -P packages/lerobot-isaac-adapters -b spinout-adapters
```

After spinout, add `lerobot-isaac-configs` and `lerobot-isaac-env` as PyPI deps in
the extracted `pyproject.toml`.
