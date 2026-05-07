# lerobot-isaac-adapters — Internals

---

## File Structure Walk-through

```
packages/lerobot-isaac-adapters/
├── pyproject.toml          — deps: pyyaml; optional: lerobot, sheeprl, transformers
├── pixi.toml
├── README.md / CLAUDE.md / docs/
├── src/
│   └── lerobot_isaac_adapters/
│       ├── __init__.py              — package marker
│       ├── train.py                 — argparse + dispatch router
│       ├── metric_extractor.py      — stdout metric emitter
│       ├── isaac_data_recorder.py   — Isaac rollout -> LeRobotDataset Parquet
│       └── targets/
│           ├── __init__.py
│           ├── policy_lerobot.py    — smolvla/act/diffusion: spawns lerobot-train
│           ├── wm_dreamerv3.py      — DreamerV3 stub
│           └── wm_leworldmodel.py   — HF LeWorldModel stub
└── tests/
    ├── test_train_argparse.py       — argparse smoke tests
    ├── test_metric_extractor.py     — metric format contract
    └── test_targets_subprocess.py   — subprocess dispatch smoke tests
```

---

## Key Data Structures

### `train._POLICY_ARCHS` and `train._WM_ARCHS`

```python
_POLICY_ARCHS = ("smolvla", "act", "diffusion")
_WM_ARCHS = ("dreamerv3", "le_world_model")
```

`argparse` uses `_ALL_ARCHS = _POLICY_ARCHS + _WM_ARCHS` as the `choices` list.

### Dispatch logic in `train._dispatch()`

```python
if arch in _POLICY_ARCHS:
    from lerobot_isaac_adapters.targets import policy_lerobot as backend
elif arch == "dreamerv3":
    from lerobot_isaac_adapters.targets import wm_dreamerv3 as backend
elif arch == "le_world_model":
    from lerobot_isaac_adapters.targets import wm_leworldmodel as backend
rc = backend.run(args)
```

All backends expose a single `run(args: argparse.Namespace) -> int` function.
Lazy imports keep startup fast when only some backends are installed.

### `metric_extractor.emit()` format

```python
f"{name}={value:.6g}"              # without step
f"{name}={value:.6g}  # step={step}"  # with step
```

The `:.6g` format (6 significant digits) was chosen to avoid representation
surprises (`0.73` vs `0.7300000000000001`). The `# step=N` comment is outside
the regex match pattern so the executor only sees `name=value`.

---

## Soft-Import Strategy

| Module | Soft-imported symbol | Location of import |
|--------|---------------------|-------------------|
| `policy_lerobot.py` | `lerobot-train` CLI (subprocess) | inside `run()` — `FileNotFoundError` if missing |
| `wm_dreamerv3.py` | `sheeprl` | never reached (NotImplementedError stub) |
| `wm_leworldmodel.py` | `transformers` | never reached (NotImplementedError stub) |
| `isaac_data_recorder.py` | `gymnasium`, `lerobot.common.datasets.lerobot_dataset` | inside `record_episodes()` |

`train.py` and `metric_extractor.py` have no soft imports — they only use stdlib.

---

## `policy_lerobot.py` — Subprocess Wiring

The policy backend spawns `lerobot-train` as a subprocess using `subprocess.Popen`,
streaming stdout line-by-line. It searches each line for `eval/pc_success=<float>`
using `_PC_SUCCESS_RE` and re-emits using `metric_extractor.emit()` (stripping the
`eval/` prefix for autoresearch compatibility).

If `lerobot-train` is not in PATH, `FileNotFoundError` is caught and the function
returns `127` (standard "command not found" exit code).

Extra args in `args.remainder` (after `--` on the CLI) are appended to the command
verbatim, minus the leading `--` separator token.

---

## `isaac_data_recorder.py` — Feature Schema

The `_FEATURES` dict defines the LeRobotDataset v3.0 schema for SO-101 recordings:

```python
_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (12,), ...},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), ...},
    "observation.images.overhead": {"dtype": "video", "shape": (480, 640, 3), ...},
    "action": {"dtype": "float32", "shape": (6,), ...},
}
```

This schema is passed to `LeRobotDataset.create(features=_FEATURES)` ensuring all
recordings are compatible with policies trained on real SO-101 data.

---

## Test Architecture

Three test files, all no external deps:

- `test_train_argparse.py` — verifies all 5 archs accepted; `--dry_run` flag works;
  unknown arch rejected; extra remainder args captured.
- `test_metric_extractor.py` — verifies `emit()` output format (stdout capture),
  `ValueError` on bad names, `MetricEmitter` prefix behavior, `metric_scope` alias.
- `test_targets_subprocess.py` — `policy_lerobot.run()` returns 127 when `lerobot-train`
  not in PATH (uses mock or subprocess that immediately fails).

---

## Known Limitations

1. **DreamerV3 and LeWorldModel backends are stubs** — `wm_dreamerv3.run()` and
   `wm_leworldmodel.run()` both raise `NotImplementedError`. Wire them in when the
   corresponding packages are installed.

2. **`policy_lerobot.py` does not normalise `pc_success`** — the raw `eval/pc_success`
   value from `lerobot-train` is re-emitted as-is. If LeRobot reports it as a
   percentage (e.g. `73.0` instead of `0.73`), the `train_wrapper` in autoresearch
   must handle normalisation.

3. **`isaac_data_recorder.py` observation keys** — the recorder writes
   `obs["joint_pos_vel"]` and `obs["wrist_cam_rgb"]`. These must match the actual
   observation dict keys from `lerobot_isaac_env`. Verify after Isaac Lab install.

---

## Future Un-stubbing Plan

| Stub | Action |
|------|--------|
| `wm_dreamerv3.run()` | Replace `NotImplementedError` with subprocess call to sheeprl/dreamer |
| `wm_leworldmodel.run()` | Replace with HuggingFace `transformers` training loop |
| World model bridge | `wm_dreamerv3.py` should call `lerobot_world_model_bridge` skill for Parquet→HDF5 |
