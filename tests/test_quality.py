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

        result = apply_quality_filter(
            dataset_path=ds, output_path=output_path, dry_run=True,
        )
        # No parquet in the dir -> the in-process filter returns a graceful failure;
        # dry_run never writes the output dir either way.
        assert result.success is False
        assert not output_path.exists()


# ---------------------------------------------------------------------------
# In-process delegation to the package quality module (replaces the old
# CLAUDE_CODE_ROOT sys.path-inject + python -c subprocess two-tier bridge,
# which was removed when the SAL/TED logic moved into
# lerobot_isaac_adapters.data.dataset_quality).
# ---------------------------------------------------------------------------


class TestInProcessDelegation:
    def test_delegates_to_package_filter_dataset(self, tmp_path: Path):
        """apply_quality_filter calls the in-package filter_dataset directly."""
        import lerobot_isaac_adapters.quality as qmod
        from lerobot_isaac_adapters.quality import apply_quality_filter, OperationResult

        ds = tmp_path / "dataset"
        ds.mkdir()
        with patch.object(qmod, "_filter_dataset") as mock_filter:
            mock_filter.return_value = OperationResult(
                success=True, data={"kept": 8, "removed": 2}
            )
            result = apply_quality_filter(dataset_path=ds, sal_threshold=0.2)

        mock_filter.assert_called_once()
        _, kwargs = mock_filter.call_args
        # sal_threshold 0.2 -> filter_percentile 20, composite SAL+TED strategy
        assert kwargs.get("filter_percentile") == 20
        assert kwargs.get("strategy") == "composite"
        assert result.success and result.data["kept"] == 8

    def test_operation_result_is_package_type(self):
        """OperationResult is sourced from the package quality module (single type)."""
        from lerobot_isaac_adapters.quality import OperationResult as QResult
        from lerobot_isaac_adapters.data.dataset_quality import OperationResult as PResult

        assert QResult is PResult

    def test_legacy_bridge_internals_removed(self):
        """The old CLAUDE_CODE_ROOT / subprocess bridge is gone."""
        import lerobot_isaac_adapters.quality as qmod

        assert not hasattr(qmod, "CLAUDE_CODE_ROOT")
        assert not hasattr(qmod, "_invoke_skill_subprocess")
        assert not hasattr(qmod, "_import_skill")
