# lerobot-isaac-adapters

Modular training adapter for the LeRobot + Isaac Lab workspace. Provides a
single entrypoint, `lerobot-isaac-train` (or `python -m lerobot_isaac_adapters.train`),
that dispatches to the correct backend based on `--target_arch`.

---

## Purpose

Decouple the "what to train" decision (policy vs world model, which architecture)
from the rest of the workspace. `autoresearch-ml-executor-worker` points at this
package as its sole training target; downstream components never need to know
which backend is active.

The package also provides:
- **`metric_extractor`** — canonical stdout metric emitter ensuring autoresearch
  can parse results via a stable regex.
- **`isaac_data_recorder`** — records Isaac Lab rollout episodes to LeRobotDataset
  Parquet format.

---

## Status

**Phase 2 — training adapter scaffolded.** All `--target_arch` backends are stubs
that raise `NotImplementedError`. Real backends are wired once the corresponding
training package is installed.

| Component | Status |
|-----------|--------|
| `train.py` (dispatch router) | Implemented |
| `metric_extractor.py` | Implemented |
| `isaac_data_recorder.py` | Implemented (soft-imports) |
| `targets/policy_lerobot.py` | Implemented — spawns `lerobot-train` subprocess |
| `targets/wm_dreamerv3.py` | Stub — raises `NotImplementedError` |
| `targets/wm_leworldmodel.py` | Stub — raises `NotImplementedError` |

---

## Installation

### Monorepo mode (pixi)

```bash
pixi install   # from workspace root
```

### Standalone mode

```bash
cd packages/lerobot-isaac-adapters
pixi install
```

### Direct pip install

```bash
# Base install (no heavy ML deps):
pip install -e packages/lerobot-isaac-adapters/

# With LeRobot policy support:
pip install -e "packages/lerobot-isaac-adapters[lerobot]"

# With DreamerV3 support:
pip install -e "packages/lerobot-isaac-adapters[dreamerv3]"

# With HF LeWorldModel support:
pip install -e "packages/lerobot-isaac-adapters[leworldmodel]"
```

---

## Quick Example

```bash
# Dry-run: verify args without actually training
lerobot-isaac-train \
  --target_arch smolvla \
  --dataset /data/real \
  --output_dir outputs/smolvla_run1 \
  --steps 50000 \
  --dry_run
```

```python
# Emit a metric line (autoresearch-compatible format)
from lerobot_isaac_adapters.metric_extractor import emit, MetricEmitter

emit("pc_success", 0.73)
# stdout: pc_success=0.73

emit("pc_success", 0.73, step=1000)
# stdout: pc_success=0.73  # step=1000

with MetricEmitter(prefix="eval") as me:
    me.emit("pred_loss", 0.021, step=500)
# stdout: eval_pred_loss=0.021  # step=500
```

---

## Supported `--target_arch` values

| Value | Backend | Primary metric emitted |
|-------|---------|------------------------|
| `smolvla` | `lerobot-train --policy.type smolvla` subprocess | `pc_success=<float>` |
| `act` | `lerobot-train --policy.type act` subprocess | `pc_success=<float>` |
| `diffusion` | `lerobot-train --policy.type diffusion` subprocess | `pc_success=<float>` |
| `dreamerv3` | sheeprl / nm-wu DreamerV3 stub | `recon_loss=<float>` |
| `le_world_model` | HF LeWorldModel stub | `pred_loss=<float>` |

---

## Public API

- **`lerobot_isaac_adapters.train.main(argv=None)`** — CLI entrypoint. Registered
  as `lerobot-isaac-train` console script.
- **`lerobot_isaac_adapters.metric_extractor.emit(name, value, step=None)`** —
  print a metric line to stdout in autoresearch-compatible format.
- **`lerobot_isaac_adapters.metric_extractor.MetricEmitter`** — context manager
  wrapping `emit()` with optional prefix and flush-on-exit.
- **`lerobot_isaac_adapters.metric_extractor.metric_scope(prefix="")`** — context
  manager convenience alias.
- **`lerobot_isaac_adapters.isaac_data_recorder.record_episodes(...)`** — record
  Isaac Lab rollout episodes to a LeRobotDataset Parquet directory.

---

## Metric Output Format

Each eval step emits exactly one line on stdout matching:

```
<metric_name>=<value>
```

Examples:
```
pc_success=0.73
recon_loss=0.0317
pred_loss=0.0214
```

Parsed by `autoresearch-ml-executor-worker` via regex:
```
(\w+)[=:\s]+([0-9.eE+-]+)
```

---

## Full CLI Usage

```bash
# Policy training (lerobot-train must be installed)
lerobot-isaac-train \
  --target_arch smolvla \
  --dataset /data/my_dataset \
  --config /path/to/configs/policy_smolvla.yaml \
  --output_dir outputs/smolvla_run1 \
  --steps 50000 \
  --batch_size 8 \
  --lr 1e-4 \
  --seed 42

# World model training (stub — raises NotImplementedError)
lerobot-isaac-train \
  --target_arch dreamerv3 \
  --dataset /data/my_dataset \
  --config /path/to/configs/wm_dreamerv3.yaml \
  --output_dir outputs/dreamerv3_run1

# Extra args forwarded to backend verbatim (after --)
lerobot-isaac-train --target_arch act --dataset ... -- --policy.n_action_steps=100

# Dry-run: print resolved command, exit 0
lerobot-isaac-train --target_arch smolvla --dataset /data --dry_run
```

---

## Dependencies

### Python (pyproject.toml)

```
pyyaml
```

Optional extras:
- `[lerobot]` — `lerobot` (heavy; not installed by default)
- `[dreamerv3]` — `sheeprl`
- `[leworldmodel]` — `transformers`, `datasets`

### Sibling package dependencies

- `lerobot-isaac-configs` (optional; provides default YAML config paths)
- `lerobot-isaac-env` (soft-import inside `isaac_data_recorder.py` at call time)

### Heavy/external dependencies

| Dependency | Optional extra | Used by |
|------------|---------------|---------|
| `lerobot` | `[lerobot]` | `targets/policy_lerobot.py` — `lerobot-train` CLI |
| `sheeprl` | `[dreamerv3]` | `targets/wm_dreamerv3.py` (stub) |
| `transformers` | `[leworldmodel]` | `targets/wm_leworldmodel.py` (stub) |
| Isaac Lab | system-wide | `isaac_data_recorder.py` |

All heavy deps are soft-imported (try/except) so the package imports cleanly without them.

---

## Configuration

All backends accept a `--config PATH` pointing to a YAML file. If omitted, defaults
from `lerobot-isaac-configs` are used. Config keys are backend-specific.

`--dry_run` flag is supported globally and by each backend; it prints the resolved command
without executing it.

---

## Running Tests

```bash
cd packages/lerobot-isaac-adapters
pytest tests/ -v
```

All tests pass without lerobot, sheeprl, or Isaac Lab installed.

---

## Extraction to Standalone Repo

```bash
# From workspace root:
git subtree split -P packages/lerobot-isaac-adapters -b spinout-adapters
git checkout spinout-adapters
# Remove workspace deps from pyproject.toml, pin sibling-pkg versions
git remote add origin git@github.com:user/lerobot-isaac-adapters.git
git push -u origin main
```
