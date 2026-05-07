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

---

## Supported `--target_arch` values

| Value           | Backend                                      | Primary metric emitted |
|-----------------|----------------------------------------------|------------------------|
| `smolvla`       | `lerobot-train --policy.type smolvla`        | `pc_success=<float>`   |
| `act`           | `lerobot-train --policy.type act`            | `pc_success=<float>`   |
| `diffusion`     | `lerobot-train --policy.type diffusion`      | `pc_success=<float>`   |
| `dreamerv3`     | sheeprl DreamerV3 (or nm-wu/dreamer-v3-pytorch) | `recon_loss=<float>` |
| `le_world_model`| HF LeWorldModel                              | `pred_loss=<float>`    |

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

This format is parsed by `autoresearch-ml-executor-worker` using the regex:

```
(\w+)[=:\s]+([0-9.eE+-]+)
```

---

## Usage

```bash
# Policy training (stub — Phase X wires the real backend)
lerobot-isaac-train \
  --target_arch smolvla \
  --dataset /path/to/dataset \
  --config /path/to/configs/policy_smolvla.yaml \
  --output_dir outputs/smolvla_run1 \
  --steps 50000 \
  --seed 42

# World model training
lerobot-isaac-train \
  --target_arch dreamerv3 \
  --dataset /path/to/dataset \
  --config /path/to/configs/wm_dreamerv3.yaml \
  --output_dir outputs/dreamerv3_run1 \
  --steps 100000

# Extra args are forwarded to the backend unchanged
lerobot-isaac-train --target_arch act --dataset ... -- --policy.n_action_steps=100
```

---

## Cross-Package Dependencies

- **`lerobot-isaac-configs`** (sibling) — provides YAML config files; optional, configs may
  be supplied by path.
- **`lerobot_world_model_bridge`** skill (repo) — called inside `wm_dreamerv3.py` and
  `wm_leworldmodel.py` to auto-convert a Parquet dataset to HDF5 if a Parquet path is
  supplied. Reference path:
  `/home/koen/tools/claude_code/skills/lerobot_world_model_bridge/`

This package does NOT import `lerobot-isaac-env` directly; the data recorder
(`isaac_data_recorder.py`) soft-imports `isaaclab` and `lerobot` so it can be used
without either installed.

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

---

## Development

```bash
pip install -e "packages/lerobot-isaac-adapters[lerobot,dreamerv3]"
pytest packages/lerobot-isaac-adapters/tests/
lerobot-isaac-train --help
```
