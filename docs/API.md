# lerobot-isaac-adapters — Public API Reference

---

## Module: `lerobot_isaac_adapters.train`

Unified training entrypoint. Parses `--target_arch` and routes to the appropriate backend.

### `main(argv: list[str] | None = None) -> None`

Console script entrypoint. Calls `sys.exit(rc)` with the backend return code.

| Parameter | Type | Description |
|-----------|------|-------------|
| `argv` | `list[str] \| None` | Argument list. If `None`, reads `sys.argv[1:]`. |

**CLI Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--target_arch` | Yes | — | One of `smolvla`, `act`, `diffusion`, `dreamerv3`, `le_world_model`. |
| `--dataset` | No | `None` | Local path or HuggingFace repo id. |
| `--config` | No | `None` | Path to backend-specific YAML config. |
| `--output_dir` | No | `"outputs/run"` | Checkpoint and log directory. |
| `--steps` | No | `50000` | Total training steps or world-model iterations. |
| `--batch_size` | No | `32` | Training batch size. |
| `--lr` | No | `1e-4` | Learning rate. |
| `--seed` | No | `42` | Random seed. |
| `--dry_run` | No | `False` | Print resolved command and exit without training. |
| `-- <extra>` | No | `[]` | Extra args forwarded verbatim to the backend. |

**Example:**
```bash
lerobot-isaac-train --target_arch smolvla --dataset /data --steps 10000 --dry_run
```

---

## Module: `lerobot_isaac_adapters.metric_extractor`

Canonical stdout metric emitter.

### `emit(name, value, step=None) -> None`

Print a single metric line in autoresearch-compatible format.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Metric name. Must match `\w+` (letters, digits, underscores only). |
| `value` | `float` | Numeric metric value. |
| `step` | `int \| None` | Optional training step; appended as comment (`# step=N`). |

**Returns:** `None`

**Raises:** `ValueError` if `name` contains non-alphanumeric/underscore characters.

**Output format:**
```
<name>=<value>         # without step
<name>=<value>  # step=<step>  # with step
```

Value is formatted as `:.6g` (6 significant digits).

**Example:**
```python
from lerobot_isaac_adapters.metric_extractor import emit

emit("pc_success", 0.73)          # stdout: pc_success=0.73
emit("recon_loss", 3.17e-2, 1000) # stdout: recon_loss=0.0317  # step=1000
```

---

### `MetricEmitter`

Context manager that wraps `emit()` with an optional prefix and flushes stdout on exit.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | `str` | `""` | Prepended to metric names with `_` separator. |

**Method:** `emit(name, value, step=None)` — same as module-level `emit()` but prepends prefix.

**Example:**
```python
with MetricEmitter(prefix="eval") as me:
    me.emit("pred_loss", 0.021, step=500)
# stdout: eval_pred_loss=0.021  # step=500
```

---

### `metric_scope(prefix="") -> Generator[MetricEmitter, None, None]`

Convenience alias for `MetricEmitter` as a context manager.

```python
with metric_scope("train") as m:
    m.emit("loss", 0.5)
# stdout: train_loss=0.5
```

---

## Module: `lerobot_isaac_adapters.isaac_data_recorder`

Records Isaac Lab rollout episodes to LeRobotDataset Parquet format.

### `record_episodes(env_id, output_dir, num_episodes, policy_fn, policy_checkpoint, seed) -> Path`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env_id` | `str` | required | Gym env ID, e.g. `"Isaac-SO101-PickPlace-v0"`. |
| `output_dir` | `str \| Path` | required | Output directory for the dataset. |
| `num_episodes` | `int` | `10` | Number of rollout episodes to record. |
| `policy_fn` | `callable \| None` | `None` | `(obs_dict) -> action_array`. Random actions if `None`. |
| `policy_checkpoint` | `str \| None` | `None` | Path to a LeRobot `.pt` checkpoint. |
| `seed` | `int` | `42` | Random seed. |

**Returns:** `Path` — root of the created LeRobotDataset.

**Raises:**
- `ImportError` — if `gymnasium` or `lerobot` not installed.

**LeRobotDataset schema produced:**

| Column | dtype | shape |
|--------|-------|-------|
| `observation.state` | float32 | (12,) — joint pos (6) + joint vel (6) |
| `observation.images.wrist` | uint8 | (480, 640, 3) |
| `observation.images.overhead` | uint8 | (480, 640, 3) |
| `action` | float32 | (6,) |

**Example:**
```python
from lerobot_isaac_adapters.isaac_data_recorder import record_episodes

path = record_episodes(
    env_id="Isaac-SO101-PickPlace-v0",
    output_dir="/data/dr_episodes",
    num_episodes=50,
    seed=0,
)
print(f"Dataset at: {path}")
```

---

## Module: `lerobot_isaac_adapters.targets.policy_lerobot`

### `run(args: argparse.Namespace) -> int`

Dispatches a LeRobot policy training run by spawning a `lerobot-train` subprocess.

**Returns:** `0` on success, `127` if `lerobot-train` not found, or the subprocess exit code.

---

## Cross-Package References

- Metric format consumed by `../../lerobot-isaac-autoresearch/docs/API.md` — `train_wrapper`
- Dataset schema matches `../../lerobot-isaac-env/docs/API.md` — observation conventions
- Config files from `../../lerobot-isaac-configs/docs/API.md` — `load_config()`
