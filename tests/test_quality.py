"""
test_quality.py — Tests for lerobot_isaac_adapters.quality

Tests:
  - Module and function importability.
  - apply_quality_filter signature matches spec.
  - OperationResult structure.
  - Missing dataset returns success=False.
  - dry_run=True returns 0 without touching filesystem.
  - Tier 1 soft-import path (mocked).
  - Tier 2 subprocess fallback path (mocked).

Plan reference: §13.1 Bundle A, deliverable A6
"""

from __future__ import annotations

import importlib
from inspect import signature
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import smoke
# ---------------------------------------------------------------------------


class TestImport:
    def test_module_importable(self):
        """quality module imports without error."""
        import lerobot_isaac_adapters.quality  # noqa: F401

    def test_apply_quality_filter_importable(self):
        from lerobot_isaac_adapters.quality import apply_quality_filter

        assert callable(apply_quality_filter)

    def test_operation_result_importable(self):
        from lerobot_isaac_adapters.quality import OperationResult

        assert OperationResult is not None


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------


class TestSignature:
    def test_signature_has_required_dataset_param(self):
        from lerobot_isaac_adapters.quality import apply_quality_filter

        sig = signature(apply_quality_filter)
        assert "dataset_path" in sig.parameters

    def test_signature_defaults(self):
        from lerobot_isaac_adapters.quality import apply_quality_filter

        sig = signature(apply_quality_filter)
        params = sig.parameters
        assert params["sal_threshold"].default == pytest.approx(0.2)
        assert params["ted_threshold"].default == pytest.approx(2.0)
        assert params["min_episode_length"].default == 50

    def test_signature_has_dry_run(self):
        from lerobot_isaac_adapters.quality import apply_quality_filter

        sig = signature(apply_quality_filter)
        assert "dry_run" in sig.parameters
        assert sig.parameters["dry_run"].default is False

    def test_signature_has_output_path(self):
        from lerobot_isaac_adapters.quality import apply_quality_filter

        sig = signature(apply_quality_filter)
        assert "output_path" in sig.parameters


# ---------------------------------------------------------------------------
# Error path: missing dataset
# ---------------------------------------------------------------------------


class TestMissingDataset:
    def test_returns_failure_for_nonexistent_path(self, tmp_path: Path):
        from lerobot_isaac_adapters.quality import apply_quality_filter

        nonexistent = tmp_path / "no_such_dataset"
        result = apply_quality_filter(dataset_path=nonexistent)
        assert not result.success
        assert result.error is not None
        assert (
            "not found" in result.error.lower()
            or "nonexistent" in result.error.lower()
            or str(nonexistent) in result.error
        )

    def test_suggestions_provided_on_error(self, tmp_path: Path):
        from lerobot_isaac_adapters.quality import apply_quality_filter

        result = apply_quality_filter(dataset_path=tmp_path / "missing")
        assert not result.success
        assert result.suggestions is not None
        assert len(result.suggestions) > 0


# ---------------------------------------------------------------------------
# dry_run tests
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_on_nonexistent_path_returns_error(self, tmp_path: Path):
        """dry_run=True still returns an error if the dataset doesn't exist."""
        from lerobot_isaac_adapters.quality import apply_quality_filter

        result = apply_quality_filter(
            dataset_path=tmp_path / "missing",
            dry_run=True,
        )
        assert not result.success  # path check comes first

    def test_dry_run_on_existing_empty_dir(self, tmp_path: Path):
        """dry_run on an existing path (even with no parquet) won't write output."""
        from lerobot_isaac_adapters.quality import apply_quality_filter

        # Create a fake dataset root
        ds = tmp_path / "my_dataset"
        ds.mkdir()
        output_path = tmp_path / "my_dataset_filtered"

        # Mock _import_skill to fail so we exercise the subprocess path
        with (
            patch("lerobot_isaac_adapters.quality._import_skill", return_value=None),
            patch(
                "lerobot_isaac_adapters.quality._invoke_skill_subprocess"
            ) as mock_sub,
        ):
            from lerobot_isaac_adapters.quality import OperationResult

            mock_sub.return_value = OperationResult(
                success=True,
                data={"kept": 5, "removed": 2, "dry_run": True},
            )
            result = apply_quality_filter(
                dataset_path=ds,
                output_path=output_path,
                dry_run=True,
            )

        assert result.success
        # Output path should NOT have been created (dry_run)
        assert not output_path.exists()
        mock_sub.assert_called_once()


# ---------------------------------------------------------------------------
# Tier 1 import mocking
# ---------------------------------------------------------------------------


class TestTier1Import:
    def test_tier1_success_calls_skill(self, tmp_path: Path):
        """When _import_skill succeeds, apply_quality_filter calls it."""
        from lerobot_isaac_adapters.quality import apply_quality_filter

        ds = tmp_path / "dataset"
        ds.mkdir()

        mock_filter = MagicMock(
            return_value=MagicMock(
                success=True,
                data={"kept": 8, "removed": 2},
                error=None,
                suggestions=None,
            )
        )

        with patch(
            "lerobot_isaac_adapters.quality._import_skill", return_value=mock_filter
        ):
            result = apply_quality_filter(dataset_path=ds)

        assert result.success
        mock_filter.assert_called_once()

    def test_tier1_exception_falls_back_to_tier2(self, tmp_path: Path):
        """If _import_skill returns callable but calling it raises, Tier 2 is used."""
        from lerobot_isaac_adapters.quality import apply_quality_filter, OperationResult

        ds = tmp_path / "dataset"
        ds.mkdir()

        def raise_on_call(*args, **kwargs):
            raise RuntimeError("simulated skill failure")

        with (
            patch(
                "lerobot_isaac_adapters.quality._import_skill",
                return_value=raise_on_call,
            ),
            patch(
                "lerobot_isaac_adapters.quality._invoke_skill_subprocess"
            ) as mock_sub,
        ):
            mock_sub.return_value = OperationResult(success=True, data={"kept": 5})
            result = apply_quality_filter(dataset_path=ds)

        mock_sub.assert_called_once()
        assert result.success


# ---------------------------------------------------------------------------
# Tier 2 subprocess mocking
# ---------------------------------------------------------------------------


class TestTier2Subprocess:
    def test_tier2_called_when_tier1_fails(self, tmp_path: Path):
        from lerobot_isaac_adapters.quality import apply_quality_filter, OperationResult

        ds = tmp_path / "dataset"
        ds.mkdir()

        with (
            patch("lerobot_isaac_adapters.quality._import_skill", return_value=None),
            patch(
                "lerobot_isaac_adapters.quality._invoke_skill_subprocess"
            ) as mock_sub,
        ):
            mock_sub.return_value = OperationResult(
                success=True,
                data={"kept": 10, "removed": 3},
            )
            result = apply_quality_filter(dataset_path=ds)

        mock_sub.assert_called_once()
        assert result.success
        assert result.data["kept"] == 10

    def test_tier2_failure_propagates(self, tmp_path: Path):
        from lerobot_isaac_adapters.quality import apply_quality_filter, OperationResult

        ds = tmp_path / "dataset"
        ds.mkdir()

        with (
            patch("lerobot_isaac_adapters.quality._import_skill", return_value=None),
            patch(
                "lerobot_isaac_adapters.quality._invoke_skill_subprocess"
            ) as mock_sub,
        ):
            mock_sub.return_value = OperationResult(
                success=False,
                error="subprocess failed",
                suggestions=["check CLAUDE_CODE_ROOT"],
            )
            result = apply_quality_filter(dataset_path=ds)

        assert not result.success
        assert "subprocess" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# CLAUDE_CODE_ROOT constant
# ---------------------------------------------------------------------------


class TestClaudeCodeRoot:
    def test_claude_code_root_is_path(self):
        from lerobot_isaac_adapters.quality import CLAUDE_CODE_ROOT

        assert isinstance(CLAUDE_CODE_ROOT, Path)

    def test_env_var_overrides_root(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("LEROBOT_CLAUDE_CODE_ROOT", str(tmp_path))
        import lerobot_isaac_adapters.quality as qmod

        importlib.reload(qmod)
        assert tmp_path == qmod.CLAUDE_CODE_ROOT
        # Cleanup: reload with original value
        monkeypatch.delenv("LEROBOT_CLAUDE_CODE_ROOT", raising=False)
        importlib.reload(qmod)
