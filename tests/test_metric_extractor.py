"""
test_metric_extractor.py
========================

Verify that metric_extractor:
- emit("pc_success", 0.73) prints exactly 'pc_success=0.73'
- Output is matched by the autoresearch regex  (\\w+)[=:\\s]+([0-9.eE+-]+)
- MetricEmitter context manager emits correctly with prefix
- Invalid metric names raise ValueError
- emit() with step appends comment but regex still matches
- Values use :.6g format (stable, no trailing repr surprises)
"""

from __future__ import annotations

import re
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from lerobot_isaac_adapters.metric_extractor import (
    MetricEmitter,
    emit,
    metric_scope,
)

# The autoresearch-ml-executor-worker regex pattern
AUTORESEARCH_REGEX = re.compile(r"(\w+)[=:\s]+([0-9.eE+-]+)")


def capture_emit(*args, **kwargs) -> str:
    """Call emit() and return the captured stdout line (stripped)."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        emit(*args, **kwargs)
    return buf.getvalue().strip()


class TestEmitFormat:
    """emit() must produce 'name=value' lines parseable by autoresearch regex."""

    def test_pc_success_format(self) -> None:
        line = capture_emit("pc_success", 0.73)
        assert line == "pc_success=0.73", f"Got: {line!r}"

    def test_recon_loss_format(self) -> None:
        line = capture_emit("recon_loss", 0.0317)
        assert line.startswith("recon_loss=")

    def test_pred_loss_format(self) -> None:
        line = capture_emit("pred_loss", 0.0214)
        assert line.startswith("pred_loss=")

    def test_integer_value(self) -> None:
        line = capture_emit("step_count", 1000)
        assert line.startswith("step_count=")

    def test_scientific_notation_value(self) -> None:
        # :.6g on 1e-5 produces '1e-05' which is regex-matchable
        line = capture_emit("val_loss", 1e-5)
        match = AUTORESEARCH_REGEX.search(line)
        assert match is not None, f"Regex did not match: {line!r}"
        assert match.group(1) == "val_loss"

    def test_with_step_still_regex_matches(self) -> None:
        line = capture_emit("pc_success", 0.5, step=1000)
        match = AUTORESEARCH_REGEX.search(line)
        assert match is not None, f"Regex did not match line with step: {line!r}"
        assert match.group(1) == "pc_success"
        assert float(match.group(2)) == pytest.approx(0.5)

    def test_6g_format_stable(self) -> None:
        """:.6g must not produce trailing zeros or repr-style suffixes."""
        line = capture_emit("pc_success", 0.73)
        # Should be exactly '0.73', not '0.7300000000000001' or '0.73!'
        assert line == "pc_success=0.73", f"Got: {line!r}"

    def test_6g_format_small_float(self) -> None:
        """Small floats should use scientific notation via :.6g."""
        line = capture_emit("loss", 3.17e-2)
        # :.6g -> '0.0317'
        assert line == "loss=0.0317", f"Got: {line!r}"


class TestAutoresearchRegex:
    """Verify the canonical autoresearch regex matches expected formats."""

    @pytest.mark.parametrize("line,expected_name,expected_value", [
        ("pc_success=0.73", "pc_success", 0.73),
        ("recon_loss=0.0317", "recon_loss", 0.0317),
        ("pred_loss=0.0214", "pred_loss", 0.0214),
        ("val_loss=1e-5", "val_loss", 1e-5),
        ("pc_success=0.73  # step=1000", "pc_success", 0.73),
    ])
    def test_regex_matches(
        self, line: str, expected_name: str, expected_value: float
    ) -> None:
        match = AUTORESEARCH_REGEX.search(line)
        assert match is not None, f"Regex did not match: {line!r}"
        assert match.group(1) == expected_name
        assert float(match.group(2)) == pytest.approx(expected_value, rel=1e-4)


class TestEmitValidation:
    """Invalid metric names must raise ValueError."""

    def test_space_in_name_raises(self) -> None:
        with pytest.raises(ValueError):
            emit("bad name", 0.5)

    def test_hyphen_in_name_raises(self) -> None:
        with pytest.raises(ValueError):
            emit("bad-name", 0.5)

    def test_dot_in_name_raises(self) -> None:
        with pytest.raises(ValueError):
            emit("bad.name", 0.5)

    def test_underscore_and_alnum_accepted(self) -> None:
        # Should not raise
        buf = StringIO()
        with patch("sys.stdout", buf):
            emit("pc_success_v2", 1.0)
        assert "pc_success_v2" in buf.getvalue()


class TestMetricEmitter:
    """MetricEmitter context manager must prefix names and flush stdout."""

    def test_prefix_prepended(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            with MetricEmitter(prefix="eval") as me:
                me.emit("pc_success", 0.9)
        line = buf.getvalue().strip()
        assert line.startswith("eval_pc_success="), f"Got: {line!r}"

    def test_no_prefix(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            with MetricEmitter() as me:
                me.emit("recon_loss", 0.03)
        line = buf.getvalue().strip()
        assert line.startswith("recon_loss=")

    def test_metric_scope_alias(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            with metric_scope("train") as m:
                m.emit("loss", 0.5)
        line = buf.getvalue().strip()
        assert line.startswith("train_loss=")

    def test_emitter_step_forwarded(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            with MetricEmitter(prefix="eval") as me:
                me.emit("pc_success", 0.8, step=500)
        line = buf.getvalue().strip()
        assert "step=500" in line
        match = AUTORESEARCH_REGEX.search(line)
        assert match is not None
