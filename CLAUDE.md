# lerobot-isaac-adapters — Package Orientation

**Phase:** 2 (training adapter)
**Role:** Single entrypoint for all training runs in the workspace. Dispatches by
`--target_arch` to policy (smolvla/act/diffusion) or world-model (dreamerv3/le_world_model)
backends. All targets are stubs until wired in Phase X.

---

## Public API

```python
from lerobot_isaac_adapters import train        # module with main()
from lerobot_isaac_adapters import metric_extractor  # emit() + MetricEmitter
```

Entry point: `lerobot-isaac-train` (console script) or
`python -m lerobot_isaac_adapters.train`.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/lerobot_isaac_adapters/train.py` | argparse + dispatch router |
| `src/lerobot_isaac_adapters/targets/policy_lerobot.py` | smolvla / act / diffusion stub |
| `src/lerobot_isaac_adapters/targets/wm_dreamerv3.py` | DreamerV3 stub |
| `src/lerobot_isaac_adapters/targets/wm_leworldmodel.py` | HF LeWorldModel stub |
| `src/lerobot_isaac_adapters/metric_extractor.py` | canonical stdout metric emitter |
| `src/lerobot_isaac_adapters/isaac_data_recorder.py` | Isaac rollout -> LeRobotDataset Parquet stub |
| `tests/test_train_argparse.py` | argparse smoke tests |
| `tests/test_metric_extractor.py` | metric format contract tests |

---

## Metric Contract

Every eval step emits exactly: `<name>=<float>` on stdout.
Parsed by `autoresearch-ml-executor-worker` regex `(\w+)[=:\s]+([0-9.eE+-]+)`.

---

## Coupling Rules (Section 11.6)

- Does NOT cross-import `lerobot-isaac-env` directly; env is accessed via
  `isaac_data_recorder.py` which soft-imports `isaaclab`.
- May import `lerobot-isaac-configs` for default config paths.
- `lerobot`, `sheeprl`, `transformers` are soft-imported (try/except ImportError).

---

## Source-of-Truth Paths (repo)

- World model bridge skill: `/home/koen/tools/claude_code/skills/lerobot_world_model_bridge/`
- Training orchestrator: `/home/koen/tools/claude_code/agents/orchestrators/lerobot-training-orchestrator.md`
- Autoresearch executor: `/home/koen/tools/claude_code/agents/workers/autoresearch-ml-executor-worker.md`
- LeRobot specialist: `/home/koen/tools/claude_code/agents/lerobot-specialist.md`

---

## Wiring in Next Phase

To wire a real backend, replace `raise NotImplementedError(...)` in the
relevant `targets/*.py` with the actual subprocess call or import.
No changes to `train.py` or `metric_extractor.py` are needed.

---

## Spinout

```bash
git subtree split -P packages/lerobot-isaac-adapters -b spinout-adapters
```

After spinout, add `lerobot-isaac-configs` and `lerobot-isaac-env` (or equivalents) as PyPI deps in the extracted `pyproject.toml`.
