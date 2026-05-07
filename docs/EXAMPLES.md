# lerobot-isaac-adapters — Usage Examples

---

## Example 1 — Import without any ML deps

```python
import lerobot_isaac_adapters
from lerobot_isaac_adapters import train, metric_extractor
print("import ok")
```

Expected output:
```
import ok
```

No external deps required.

---

## Example 2 — Emit a metric line

```python
from lerobot_isaac_adapters.metric_extractor import emit

emit("pc_success", 0.73)
emit("recon_loss", 0.0317)
emit("pc_success", 0.85, step=10000)
```

Expected stdout:
```
pc_success=0.73
recon_loss=0.0317
pc_success=0.85  # step=10000
```

---

## Example 3 — Use MetricEmitter with prefix

```python
from lerobot_isaac_adapters.metric_extractor import MetricEmitter

with MetricEmitter(prefix="eval") as me:
    me.emit("pc_success", 0.72, step=5000)
    me.emit("loss", 0.143, step=5000)
```

Expected stdout:
```
eval_pc_success=0.72  # step=5000
eval_loss=0.143  # step=5000
```

---

## Example 4 — Dry-run a training command

```bash
lerobot-isaac-train \
  --target_arch smolvla \
  --dataset /data/real \
  --output_dir outputs/smolvla_run1 \
  --steps 50000 \
  --batch_size 8 \
  --lr 3e-4 \
  --dry_run
```

Expected stdout:
```
[dry_run] target_arch=smolvla dataset=/data/real output_dir=outputs/smolvla_run1 steps=50000 batch_size=8 lr=0.0003 seed=42
lerobot-train --policy.type=smolvla --dataset.repo_id=/data/real ...
```

---

## Example 5 — Dispatch through Python API

```python
import sys
from lerobot_isaac_adapters.train import main

# Dry-run: prints command without executing
main(["--target_arch", "act", "--dataset", "/data", "--dry_run"])
```

---

## Example 6 — Record episodes from Isaac Lab

Requires: Isaac Lab, lerobot.

```python
from lerobot_isaac_adapters.isaac_data_recorder import record_episodes

path = record_episodes(
    env_id="Isaac-SO101-PickPlace-v0",
    output_dir="/data/dr_episodes",
    num_episodes=50,
    seed=42,
)
print(f"Wrote dataset to: {path}")
```

Expected output:
```
[isaac_data_recorder] episode 1/50 (300 steps)
...
[isaac_data_recorder] Dataset written to /data/dr_episodes
Wrote dataset to: /data/dr_episodes
```

---

## Example 7 — Use a policy checkpoint for recording

Requires: Isaac Lab, lerobot, torch.

```python
from lerobot_isaac_adapters.isaac_data_recorder import record_episodes

path = record_episodes(
    env_id="Isaac-SO101-PickPlace-v0",
    output_dir="/data/policy_rollouts",
    num_episodes=100,
    policy_checkpoint="outputs/smolvla_run1/checkpoints/last.pt",
    seed=0,
)
```

---

## Example 8 — Integration: train with autoresearch wrapper

Demonstrates how `lerobot-isaac-autoresearch` calls this package.

```bash
# The autoresearch executor calls train_wrapper.py which calls this package:
python -m lerobot_isaac_autoresearch.train_wrapper \
  --target_arch smolvla \
  --dataset /data/real \
  --output_dir /tmp/run_001 \
  --steps 20000

# Expected last stdout line:
# pc_success=0.73    (or pc_success=0.0 sentinel if no metric found)
```

The wrapper guarantees the last stdout line is `<metric>=<float>` as required by
`autoresearch-ml-executor-worker`. See `../../lerobot-isaac-autoresearch/docs/API.md`.
