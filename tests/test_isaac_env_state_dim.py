"""test_isaac_env_state_dim — verify IsaacSO101Env state_dim expands with object_pose flag.

Tests:
  1. _state_dim == 6 when LEROBOT_ISAAC_INCLUDE_OBJECT_POSE is unset.
  2. _state_dim == 13 when LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1.
  3. observation_space["state"] shape matches _state_dim.

These tests do NOT boot Isaac Lab — they only check the spaces declared in
__init__, which are computed from the module-level _INCLUDE_OBJECT_POSE flag
without any Isaac Lab imports.
"""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager

import pytest


_ADAPTER_MODULES = (
    "lerobot_isaac_adapters.sheeprl_plugin.isaac_env",
    "lerobot_isaac_adapters.sheeprl_plugin",
)


@contextmanager
def _isolated_import(*module_names: str):
    """Save/restore sys.modules so module-level flags are re-evaluated."""
    saved = {k: v for k, v in sys.modules.items()}
    try:
        for name in list(sys.modules):
            for mod in module_names:
                if name == mod or name.startswith(mod + "."):
                    sys.modules.pop(name, None)
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


def test_isaac_env_state_dim_default(monkeypatch):
    """IsaacSO101Env._state_dim must be 6 when env var is absent."""
    with _isolated_import(*_ADAPTER_MODULES):
        monkeypatch.delenv("LEROBOT_ISAAC_INCLUDE_OBJECT_POSE", raising=False)
        mod = importlib.import_module(
            "lerobot_isaac_adapters.sheeprl_plugin.isaac_env"
        )
        env = mod.IsaacSO101Env()
        assert env._state_dim == 6, f"expected _state_dim=6, got {env._state_dim}"
        assert env.observation_space["state"].shape == (6,), (
            f"expected obs_space state shape (6,), "
            f"got {env.observation_space['state'].shape}"
        )


def test_isaac_env_state_dim_with_object_pose(monkeypatch):
    """IsaacSO101Env._state_dim must be 13 when LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1."""
    with _isolated_import(*_ADAPTER_MODULES):
        monkeypatch.setenv("LEROBOT_ISAAC_INCLUDE_OBJECT_POSE", "1")
        mod = importlib.import_module(
            "lerobot_isaac_adapters.sheeprl_plugin.isaac_env"
        )
        env = mod.IsaacSO101Env()
        assert env._state_dim == 13, f"expected _state_dim=13, got {env._state_dim}"
        assert env.observation_space["state"].shape == (13,), (
            f"expected obs_space state shape (13,), "
            f"got {env.observation_space['state'].shape}"
        )


def test_isaac_env_obs_space_consistent(monkeypatch):
    """observation_space['state'].shape[0] must equal _state_dim regardless of flag."""
    for flag_val in ("0", "1"):
        with _isolated_import(*_ADAPTER_MODULES):
            monkeypatch.setenv("LEROBOT_ISAAC_INCLUDE_OBJECT_POSE", flag_val)
            mod = importlib.import_module(
                "lerobot_isaac_adapters.sheeprl_plugin.isaac_env"
            )
            env = mod.IsaacSO101Env()
            declared = env.observation_space["state"].shape[0]
            assert declared == env._state_dim, (
                f"flag={flag_val}: obs_space dim {declared} != _state_dim {env._state_dim}"
            )
