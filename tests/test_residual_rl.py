"""Residual-RL-on-scripted-grasp tests.

Covers the CPU-testable contract of the residual-RL feature (lever 1):
  * `IsaacSO101Env.compute_scripted_action()` hardware/no-scene fallback → None
    (the sim-only boundary — on hardware the residual must silently become a no-op).
  * `_wm_isaac_entry._patch_residual_rl_action()` is a no-op when OFF (default).
  * The blend/schedule/eval-skip/self-actions-consistency of the get_actions patch
    (torch-guarded — runs in the sim/train-dreamer env where the real run lives).

The GPU run validates the reactive controller's grasp behaviour; these guard the
contract (hardware fallback, default-OFF, buffer/executed-action consistency).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Workspace scripts/ dir (holds _wm_isaac_entry.py). Skip the patch tests gracefully
# if absent (e.g. after a standalone spinout of this sibling).
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
_HAS_ENTRY = (_SCRIPTS / "_wm_isaac_entry.py").is_file()


def _load_entry():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import _wm_isaac_entry  # noqa: PLC0415

    return _wm_isaac_entry


# --------------------------------------------------------------------------- #
# compute_scripted_action — hardware / no-scene fallback (torch-free)
# --------------------------------------------------------------------------- #


def test_compute_scripted_action_returns_none_without_scene():
    """No Isaac scene (e.g. hardware deploy, or pre-boot) → None, no crash.

    This is the sim-only boundary: the residual patch treats None as 'use the pure
    policy action', so the residual is automatically OFF when there is no sim scene.
    """
    pytest.importorskip("gymnasium")
    from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import IsaacSO101Env

    env = IsaacSO101Env()
    assert env._isaac_env is None  # not booted
    assert env.compute_scripted_action() is None


def test_compute_scripted_action_none_when_backing_has_no_scene():
    """A backing env object lacking `.scene` (degenerate) → None, not an exception."""
    pytest.importorskip("gymnasium")
    from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import IsaacSO101Env

    env = IsaacSO101Env()
    env._isaac_env = object()  # has no `.scene`
    assert env.compute_scripted_action() is None


# --------------------------------------------------------------------------- #
# patch gating — OFF by default (torch-free: returns before importing sheeprl)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _HAS_ENTRY, reason="workspace scripts/_wm_isaac_entry.py not present")
def test_residual_patch_off_by_default(monkeypatch):
    """Weight unset (default 0.0) → patch is a no-op, does not even import sheeprl."""
    monkeypatch.delenv("LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT", raising=False)
    entry = _load_entry()
    # Inject a fake PlayerDV3 so we can prove it stays unpatched.
    fake_player_cls = _install_fake_player(monkeypatch)
    entry._patch_residual_rl_action()
    assert not getattr(fake_player_cls.get_actions, "_lerobot_residual_patched", False)


@pytest.mark.skipif(not _HAS_ENTRY, reason="workspace scripts/_wm_isaac_entry.py not present")
def test_residual_patch_off_when_weight_zero(monkeypatch):
    monkeypatch.setenv("LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT", "0.0")
    entry = _load_entry()
    fake_player_cls = _install_fake_player(monkeypatch)
    entry._patch_residual_rl_action()
    assert not getattr(fake_player_cls.get_actions, "_lerobot_residual_patched", False)


# --------------------------------------------------------------------------- #
# patch blend / schedule / eval-skip (torch-guarded)
# --------------------------------------------------------------------------- #


def _install_fake_player(monkeypatch, policy_val: float = 0.0):
    """Register a fake sheeprl.algos.dreamer_v3.agent.PlayerDV3 in sys.modules.

    Uses monkeypatch.setitem so the fake is torn down after the test — otherwise it would
    clobber the REAL sheeprl module for later tests in the same process (the torch-guarded
    tests run in the sim/train-dreamer env where real sheeprl is installed).

    The fake get_actions mimics the real one: returns [policy_tensor] AND sets
    self.actions = cat(actions, -1). `policy_val` sets the (constant) policy action so the
    blend's policy term is observable (not just zeros).
    """
    for name in ("sheeprl", "sheeprl.algos", "sheeprl.algos.dreamer_v3"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    agent_mod = types.ModuleType("sheeprl.algos.dreamer_v3.agent")

    class FakePlayerDV3:
        def __init__(self):
            self.actions = None

        def get_actions(self, obs, greedy=False, mask=None):
            import torch  # noqa: PLC0415

            pol = torch.full((1, 6), float(policy_val), dtype=torch.float32)
            self.actions = torch.cat([pol], -1)
            return [pol]

    agent_mod.PlayerDV3 = FakePlayerDV3
    monkeypatch.setitem(sys.modules, "sheeprl.algos.dreamer_v3.agent", agent_mod)
    return FakePlayerDV3


@pytest.mark.skipif(not _HAS_ENTRY, reason="workspace scripts/_wm_isaac_entry.py not present")
def test_residual_patch_blends_and_respects_eval(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT", "1.0")  # full script at step 0
    monkeypatch.setenv("LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS", "100")
    entry = _load_entry()
    fake_player_cls = _install_fake_player(monkeypatch)

    # Fake wrapper whose scripted action is a known constant (clipped to [-1,1]).
    import numpy as np

    from lerobot_isaac_adapters.sheeprl_plugin import isaac_env as ienv

    class _FakeWrapper:
        def compute_scripted_action(self):
            return np.array([0.5, -0.5, 1.5, -1.5, 0.0, -1.0], dtype=np.float32)

    monkeypatch.setattr(ienv, "_LAST_WRAPPER", _FakeWrapper(), raising=False)

    entry._patch_residual_rl_action()
    assert getattr(fake_player_cls.get_actions, "_lerobot_residual_patched", False)

    player = fake_player_cls()

    # Training step 0 with w0=1.0 → script_frac=1.0 → blended == clip(script,-1,1).
    out = player.get_actions({}, greedy=False)
    blended = out[0].detach().cpu().numpy().reshape(-1)
    expected = np.clip([0.5, -0.5, 1.5, -1.5, 0.0, -1.0], -1.0, 1.0)
    assert np.allclose(blended, expected, atol=1e-5)
    # player.actions must equal the EXECUTED (blended) action — latent consistency.
    assert np.allclose(player.actions.detach().cpu().numpy().reshape(-1), expected, atol=1e-5)

    # Eval (greedy=True) → pure policy (zeros), NEVER blended.
    out_eval = player.get_actions({}, greedy=True)
    assert np.allclose(out_eval[0].detach().cpu().numpy().reshape(-1), 0.0, atol=1e-5)


@pytest.mark.skipif(not _HAS_ENTRY, reason="workspace scripts/_wm_isaac_entry.py not present")
def test_residual_patch_handsoff_after_decay(monkeypatch):
    """Past decay_steps the script fraction is 0 → pure policy (handoff complete)."""
    pytest.importorskip("torch")
    monkeypatch.setenv("LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT", "1.0")
    monkeypatch.setenv("LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS", "10")
    entry = _load_entry()
    fake_player_cls = _install_fake_player(monkeypatch)

    import numpy as np

    from lerobot_isaac_adapters.sheeprl_plugin import isaac_env as ienv

    class _FakeWrapper:
        def compute_scripted_action(self):
            return np.ones(6, dtype=np.float32)

    monkeypatch.setattr(ienv, "_LAST_WRAPPER", _FakeWrapper(), raising=False)
    entry._patch_residual_rl_action()
    player = fake_player_cls()

    # Drive past decay_steps; eventually script_frac → 0 → blended == policy (zeros).
    last = None
    for _ in range(30):
        last = player.get_actions({}, greedy=False)[0].detach().cpu().numpy().reshape(-1)
    assert np.allclose(last, 0.0, atol=1e-5)


@pytest.mark.skipif(not _HAS_ENTRY, reason="workspace scripts/_wm_isaac_entry.py not present")
def test_residual_patch_intermediate_blend_uses_both_terms(monkeypatch):
    """At an intermediate script_frac the blend is a TRUE convex mix of BOTH the script
    and the (non-zero) policy action — guards against the policy term being dropped."""
    pytest.importorskip("torch")
    import numpy as np

    # w0=1.0, decay=2 → step0 frac=1.0, step1 frac=0.5.
    monkeypatch.setenv("LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT", "1.0")
    monkeypatch.setenv("LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS", "2")
    entry = _load_entry()
    # Policy outputs a constant 1.0; script outputs 0.2 → at frac=0.5 blend = 0.6.
    fake_player_cls = _install_fake_player(monkeypatch, policy_val=1.0)

    from lerobot_isaac_adapters.sheeprl_plugin import isaac_env as ienv

    class _FakeWrapper:
        def compute_scripted_action(self):
            return np.full(6, 0.2, dtype=np.float32)

    monkeypatch.setattr(ienv, "_LAST_WRAPPER", _FakeWrapper(), raising=False)
    entry._patch_residual_rl_action()
    player = fake_player_cls()

    # step 0 → frac=1.0 → pure script (0.2).
    out0 = player.get_actions({}, greedy=False)[0].detach().cpu().numpy().reshape(-1)
    assert np.allclose(out0, 0.2, atol=1e-5)
    # step 1 → frac=0.5 → 0.5*0.2 + 0.5*1.0 = 0.6 (BOTH terms contribute).
    out1 = player.get_actions({}, greedy=False)[0].detach().cpu().numpy().reshape(-1)
    assert np.allclose(out1, 0.6, atol=1e-5)
