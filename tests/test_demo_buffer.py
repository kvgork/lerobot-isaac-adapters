"""Unit tests for the demo-buffer foundation (schema-agnostic, no GPU/lerobot)."""

import numpy as np
import pytest

from lerobot_isaac_adapters.sheeprl_plugin.demo_buffer import (
    DemoBuffer,
    bc_weight,
    _to_chw_uint8,
)


def _fake_episode(T, image_size=8, state_dim=18, action_dim=6):
    return {
        "rgb": np.zeros((T, 3, image_size, image_size), dtype=np.uint8),
        "state": np.zeros((T, state_dim), dtype=np.float32),
        "actions": np.zeros((T, action_dim), dtype=np.float32),
        "rewards": np.zeros((T, 1), dtype=np.float32),
        "terminated": np.zeros((T, 1), dtype=bool),
        "truncated": np.zeros((T, 1), dtype=bool),
        "is_first": np.zeros((T, 1), dtype=bool),
    }


def test_demo_buffer_counts():
    buf = DemoBuffer(episodes=[_fake_episode(10), _fake_episode(20)])
    assert buf.n_episodes == 2
    assert buf.n_transitions == 30


def test_demo_buffer_sample_shape():
    buf = DemoBuffer(episodes=[_fake_episode(30), _fake_episode(40)], _rng=np.random.default_rng(1))
    batch = buf.sample(batch_size=4, seq_len=8)
    # sheeprl layout: (seq, batch, ...)
    assert batch["rgb"].shape == (8, 4, 3, 8, 8)
    assert batch["state"].shape == (8, 4, 18)
    assert batch["actions"].shape == (8, 4, 6)
    assert batch["is_first"].shape == (8, 4, 1)


def test_demo_buffer_rejects_too_long_seq():
    buf = DemoBuffer(episodes=[_fake_episode(5)])
    with pytest.raises(ValueError):
        buf.sample(batch_size=2, seq_len=8)


def test_demo_buffer_drops_empty_episodes():
    buf = DemoBuffer(episodes=[_fake_episode(0), _fake_episode(10)])
    assert buf.n_episodes == 1


def test_to_chw_uint8_nonsquare_float_hwc():
    # real-cam-style: (480,640,3) float[0,1] HWC → (3,64,64) uint8, bilinear (no crash)
    img = np.random.default_rng(0).random((480, 640, 3)).astype(np.float32)
    out = _to_chw_uint8(img, 64)
    assert out.shape == (3, 64, 64)
    assert out.dtype == np.uint8
    assert out.max() <= 255 and out.min() >= 0


def test_to_chw_uint8_already_chw_uint8():
    img = np.zeros((3, 64, 64), dtype=np.uint8)
    out = _to_chw_uint8(img, 64)
    assert out.shape == (3, 64, 64) and out.dtype == np.uint8


def test_bc_weight_schedule():
    assert bc_weight(0, decay_steps=100) == pytest.approx(1.0)
    assert bc_weight(50, decay_steps=100) == pytest.approx(0.5)
    assert bc_weight(100, decay_steps=100) == 0.0
    assert bc_weight(200, decay_steps=100) == 0.0
