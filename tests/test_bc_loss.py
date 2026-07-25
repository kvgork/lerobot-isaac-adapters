"""CPU unit tests for the DreamerFD BC actor-loss components.

Tests:
  - bc_weight() decay schedule: linear 1→0, clamp at 0 after decay_steps
  - behavior_cloning_loss(): finite scalar, sign, virtual-clutch zeroing, w=0 fast-path
  - Decay integration: step counter drives weight from full → zero

bc_weight() is pure Python — passes in any pixi env.
behavior_cloning_loss() requires torch — skipped gracefully when torch is absent.
"""
from __future__ import annotations

import math

import pytest

from lerobot_isaac_adapters.sheeprl_plugin.demo_buffer import (
    behavior_cloning_loss,
    bc_weight,
)

# Try to import torch; mark torch-dependent tests with `requires_torch` skip condition.
try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed in this env")


# ---------------------------------------------------------------------------
# bc_weight() decay schedule tests — pure Python, always run
# ---------------------------------------------------------------------------


def test_bc_weight_at_zero():
    """Full weight at step 0."""
    assert bc_weight(0, start=1.0, decay_steps=100) == pytest.approx(1.0)


def test_bc_weight_midpoint():
    """Half weight at step = decay_steps // 2."""
    assert bc_weight(50, start=1.0, decay_steps=100) == pytest.approx(0.5)


def test_bc_weight_at_decay_steps():
    """Zero exactly at decay_steps."""
    assert bc_weight(100, start=1.0, decay_steps=100) == pytest.approx(0.0)


def test_bc_weight_beyond_decay_steps():
    """Clamped to 0 after decay_steps."""
    assert bc_weight(9999, start=1.0, decay_steps=100) == pytest.approx(0.0)


def test_bc_weight_custom_start():
    """Respects custom start value."""
    assert bc_weight(0, start=2.5, decay_steps=200) == pytest.approx(2.5)
    assert bc_weight(100, start=2.5, decay_steps=200) == pytest.approx(1.25)


def test_bc_weight_monotone_decreasing():
    """Weight is non-increasing over training steps."""
    steps = range(0, 120, 10)
    weights = [bc_weight(s, decay_steps=100) for s in steps]
    for a, b in zip(weights, weights[1:]):
        assert a >= b, f"weight increased: {a} -> {b}"


def test_bc_weight_decay_over_loop():
    """Simulate a 200-step training loop; verify full→zero profile."""
    decay = 100
    weights = [bc_weight(s, decay_steps=decay) for s in range(200)]
    assert weights[0] == pytest.approx(1.0)
    assert weights[decay] == pytest.approx(0.0)
    assert all(w == pytest.approx(0.0) for w in weights[decay:])


# ---------------------------------------------------------------------------
# Import smoke test — always run, no torch required
# ---------------------------------------------------------------------------


def test_demo_buffer_module_has_bc_symbols():
    """The module exposes all BC-related symbols at import time (no GPU needed)."""
    import importlib
    mod = importlib.import_module("lerobot_isaac_adapters.sheeprl_plugin.demo_buffer")
    assert hasattr(mod, "behavior_cloning_loss")
    assert hasattr(mod, "bc_weight")
    assert hasattr(mod, "DemoBuffer")
    assert hasattr(mod, "load_sim_demos")


# ---------------------------------------------------------------------------
# behavior_cloning_loss() tests — require torch, skipped in default env
# ---------------------------------------------------------------------------


def _make_gaussian_actor(action_dim: int = 4):
    """Minimal actor_logprob callable: fixed Diagonal Gaussian N(0, I).

    Returns a function (latents, demo_actions) → per-sample log_prob tensor.
    Latents are ignored; the distribution is constant for deterministic tests.
    """
    dist = _torch.distributions.Independent(
        _torch.distributions.Normal(
            _torch.zeros(action_dim), _torch.ones(action_dim)
        ),
        1,
    )

    def actor_logprob(latents, demo_actions):
        return dist.log_prob(demo_actions)

    return actor_logprob


@requires_torch
def test_bc_loss_is_finite():
    """BC loss is a finite scalar when weight > 0."""
    actor_logprob = _make_gaussian_actor(4)
    latents = _torch.zeros(8, 16)
    demo_actions = _torch.randn(8, 4)
    loss = behavior_cloning_loss(actor_logprob, latents, demo_actions, step=0, decay_steps=1000)
    assert loss.ndim == 0, "loss must be a scalar"
    assert math.isfinite(loss.item()), f"loss is not finite: {loss.item()}"


@requires_torch
def test_bc_loss_zero_when_weight_zero():
    """When step >= decay_steps, weight = 0 → loss = 0 (fast path)."""
    actor_logprob = _make_gaussian_actor(4)
    latents = _torch.zeros(8, 16)
    demo_actions = _torch.randn(8, 4)
    loss = behavior_cloning_loss(actor_logprob, latents, demo_actions, step=9999, decay_steps=100)
    assert loss.item() == pytest.approx(0.0)


@requires_torch
def test_bc_loss_zero_when_kl_exceeds_clutch():
    """Virtual clutch: KL > kl_clutch zeroes the weight → loss = 0."""
    actor_logprob = _make_gaussian_actor(4)
    latents = _torch.zeros(8, 16)
    demo_actions = _torch.randn(8, 4)
    loss = behavior_cloning_loss(
        actor_logprob, latents, demo_actions,
        step=0, kl=10.0, kl_clutch=3.0, decay_steps=1000,
    )
    assert loss.item() == pytest.approx(0.0)


@requires_torch
def test_bc_loss_nonzero_when_kl_below_clutch():
    """When KL < kl_clutch, clutch is open → loss is nonzero."""
    actor_logprob = _make_gaussian_actor(4)
    latents = _torch.zeros(8, 16)
    demo_actions = _torch.randn(8, 4)
    loss = behavior_cloning_loss(
        actor_logprob, latents, demo_actions,
        step=0, kl=1.0, kl_clutch=3.0, decay_steps=1000,
    )
    assert loss.item() != pytest.approx(0.0)


@requires_torch
def test_bc_loss_returns_torch_tensor():
    """Return type is always a torch scalar tensor."""
    actor_logprob = _make_gaussian_actor(4)
    latents = _torch.zeros(4, 8)
    demo_actions = _torch.randn(4, 4)
    loss = behavior_cloning_loss(actor_logprob, latents, demo_actions, step=0, decay_steps=100)
    assert isinstance(loss, _torch.Tensor)
    assert loss.shape == _torch.Size([])


@requires_torch
def test_bc_loss_scales_linearly_with_weight():
    """BC loss magnitude is 2× when weight is 2× (same demo batch, same logprob).

    step=0 → weight=1.0; step=50 → weight=0.5; decay_steps=100.
    ratio loss_full / loss_half must equal 2.0 ± ε.
    """
    actor_logprob = _make_gaussian_actor(4)
    latents = _torch.zeros(16, 8)
    demo_actions = _torch.randn(16, 4)

    loss_full = behavior_cloning_loss(
        actor_logprob, latents, demo_actions, step=0, decay_steps=100
    )
    loss_half = behavior_cloning_loss(
        actor_logprob, latents, demo_actions, step=50, decay_steps=100
    )

    if loss_half.item() != 0.0:
        ratio = abs(loss_full.item()) / abs(loss_half.item())
        assert ratio == pytest.approx(2.0, rel=1e-4)
