"""
test_loggers.py
===============
Tests for the loggers package (Bundle E scaffold, 2026-05-21).
Verifies:
- StdoutLogger always works.
- WandbLogger import raises ImportError when wandb is unavailable.
- Factory falls back to stdout when wandb is missing.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# StdoutLogger
# ---------------------------------------------------------------------------


def test_stdout_logger_log():
    from lerobot_isaac_adapters.loggers import StdoutLogger

    buf = io.StringIO()
    with redirect_stdout(buf):
        log = StdoutLogger(run_name="test-run")
        log.log(step=1, metrics={"loss": 0.5, "pc_success": 0.7})
        log.finish()

    out = buf.getvalue()
    assert "test-run" in out
    assert "loss" in out
    assert "pc_success" in out


def test_stdout_logger_log_image_shape():
    from lerobot_isaac_adapters.loggers import StdoutLogger

    img = MagicMock()
    img.shape = (224, 224, 3)
    buf = io.StringIO()
    with redirect_stdout(buf):
        log = StdoutLogger()
        log.log_image(step=1, key="camera", image=img)

    assert "shape=(224, 224, 3)" in buf.getvalue()


def test_stdout_logger_log_artifact():
    from lerobot_isaac_adapters.loggers import StdoutLogger

    buf = io.StringIO()
    with redirect_stdout(buf):
        log = StdoutLogger()
        log.log_artifact(step=5, name="ckpt", path=Path("/tmp/x.bin"))

    assert "artifact=ckpt" in buf.getvalue()


# ---------------------------------------------------------------------------
# WandbLogger
# ---------------------------------------------------------------------------


def test_wandb_logger_import_error_when_sdk_missing(monkeypatch):
    """If wandb SDK is not installed, instantiating WandbLogger raises ImportError."""
    # Force wandb import failure
    monkeypatch.setitem(sys.modules, "wandb", None)

    from lerobot_isaac_adapters.loggers import WandbLogger

    with pytest.raises(ImportError, match="wandb"):
        WandbLogger(run_name="test")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_returns_stdout_by_default():
    from lerobot_isaac_adapters.loggers.factory import get_logger

    log = get_logger("stdout")
    assert log.name == "stdout"


def test_factory_unknown_backend_raises():
    from lerobot_isaac_adapters.loggers.factory import get_logger

    with pytest.raises(ValueError, match="Unknown logger"):
        get_logger("nonexistent-backend")


def test_factory_wandb_falls_back_to_stdout_when_sdk_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    from lerobot_isaac_adapters.loggers.factory import get_logger

    log = get_logger("wandb", run_name="x")
    # Should fall back to stdout
    assert log.name == "stdout"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_stdout_logger_satisfies_protocol():
    from lerobot_isaac_adapters.loggers import LoggerBase, StdoutLogger

    log = StdoutLogger()
    assert isinstance(log, LoggerBase)
