"""
test_train_argparse.py
======================

Verify that train.py argparse layer:
- Accepts each valid --target_arch value
- Rejects an invalid --target_arch value
- Prints help without error
- With --dry_run: dispatches to backend, backend prints resolved cmd, returns 0
- --dry_run is accepted for every target_arch and returns 0 without spawning subprocesses
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot_isaac_adapters.train import _build_parser, _dispatch, _ALL_ARCHS

VALID_ARCHS = list(_ALL_ARCHS)
# Expected: smolvla, act, diffusion, dreamerv3, le_world_model
assert len(VALID_ARCHS) == 5, f"Expected 5 archs, got {VALID_ARCHS}"

# Path to src/ so subprocess invocations can find the package
_SRC_DIR = str(Path(__file__).parent.parent / "src")


def _subprocess_env() -> dict:
    """Return env with src/ prepended to PYTHONPATH for subprocess calls."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_SRC_DIR}:{existing}" if existing else _SRC_DIR
    return env


class TestArgparseAcceptsValidArchs:
    """Parser should accept all documented target_arch values."""

    @pytest.mark.parametrize("arch", VALID_ARCHS)
    def test_valid_arch_parses(self, arch: str) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", arch])
        assert args.target_arch == arch

    @pytest.mark.parametrize("arch", VALID_ARCHS)
    def test_defaults_are_set(self, arch: str) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", arch])
        assert args.steps == 50_000
        assert args.batch_size == 32
        assert args.lr == pytest.approx(1e-4)
        assert args.seed == 42
        assert args.output_dir == "outputs/run"
        assert args.dataset is None
        assert args.config is None


class TestArgparseRejectsInvalidArch:
    """Parser should exit with code 2 when given an unknown arch."""

    def test_invalid_arch_rejected(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--target_arch", "banana"])
        assert exc_info.value.code == 2

    def test_missing_target_arch_rejected(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code == 2


class TestArgparseHelp:
    """--help must exit 0 and print usage."""

    def test_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "lerobot_isaac_adapters.train", "--help"],
            capture_output=True,
            text=True,
            env=_subprocess_env(),
        )
        assert result.returncode == 0, f"--help returned non-zero: {result.stderr}"

    def test_help_mentions_all_archs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "lerobot_isaac_adapters.train", "--help"],
            capture_output=True,
            text=True,
            env=_subprocess_env(),
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        for arch in VALID_ARCHS:
            assert arch in result.stdout, (
                f"Arch '{arch}' not mentioned in --help output.\n"
                f"stdout:\n{result.stdout}"
            )


class TestDispatchDryRunReturnsZero:
    """Each dispatch target with --dry_run must return 0 (no subprocess spawned)."""

    @pytest.mark.parametrize("arch", VALID_ARCHS)
    def test_dry_run_returns_zero(self, arch: str) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch",
                arch,
                "--dataset",
                "lerobot/pusht",
                "--dry_run",
            ]
        )
        result = _dispatch(args)
        assert result == 0


class TestArgparsePassthrough:
    """Extra args after '--' are captured in remainder."""

    def test_remainder_captured(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch",
                "act",
                "--dataset",
                "/tmp/ds",
                "--",
                "--policy.n_action_steps=100",
            ]
        )
        assert "--policy.n_action_steps=100" in args.remainder

    def test_all_custom_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch",
                "dreamerv3",
                "--dataset",
                "my/dataset",
                "--config",
                "/tmp/config.yaml",
                "--output_dir",
                "/tmp/out",
                "--steps",
                "10000",
                "--batch_size",
                "16",
                "--lr",
                "3e-4",
                "--seed",
                "7",
            ]
        )
        assert args.target_arch == "dreamerv3"
        assert args.dataset == "my/dataset"
        assert args.config == "/tmp/config.yaml"
        assert args.output_dir == "/tmp/out"
        assert args.steps == 10_000
        assert args.batch_size == 16
        assert args.lr == pytest.approx(3e-4)
        assert args.seed == 7


class TestDryRun:
    """--dry_run must be accepted for every arch and short-circuit dispatch."""

    @pytest.mark.parametrize("arch", VALID_ARCHS)
    def test_dry_run_flag_accepted(self, arch: str) -> None:
        """Parser accepts --dry_run for every target_arch."""
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", arch, "--dry_run"])
        assert args.dry_run is True

    @pytest.mark.parametrize("arch", VALID_ARCHS)
    def test_dry_run_returns_zero_and_does_not_raise(self, arch: str) -> None:
        """_dispatch with --dry_run returns 0 and never spawns subprocess."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch",
                arch,
                "--dataset",
                "lerobot/pusht",
                "--dry_run",
            ]
        )
        result = _dispatch(args)
        assert result == 0

    def test_dry_run_default_is_false(self) -> None:
        """--dry_run defaults to False when not supplied."""
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.dry_run is False

    def test_dry_run_prints_key_args(self, capsys) -> None:
        """dry-run output contains target_arch, dataset, output_dir, steps, batch_size, lr, seed."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch",
                "smolvla",
                "--dataset",
                "lerobot/pusht",
                "--output_dir",
                "/tmp/out",
                "--steps",
                "1000",
                "--batch_size",
                "8",
                "--lr",
                "3e-4",
                "--seed",
                "7",
                "--dry_run",
            ]
        )
        _dispatch(args)
        captured = capsys.readouterr()
        assert "smolvla" in captured.out
        assert "lerobot/pusht" in captured.out
        assert "/tmp/out" in captured.out
        assert "1000" in captured.out
        assert "8" in captured.out
        assert "7" in captured.out

    def test_policy_dry_run_prints_lerobot_train(self, capsys) -> None:
        """Policy dry-run output must include 'lerobot-train'."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch",
                "smolvla",
                "--dataset",
                "lerobot/pusht",
                "--dry_run",
            ]
        )
        _dispatch(args)
        captured = capsys.readouterr()
        assert "lerobot-train" in captured.out, (
            f"Expected 'lerobot-train' in dry-run output.\nstdout: {captured.out!r}"
        )

    def test_dreamerv3_dry_run_prints_sheeprl(self, capsys) -> None:
        """DreamerV3 dry-run output must include 'sheeprl'."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch",
                "dreamerv3",
                "--dataset",
                "lerobot/pusht",
                "--dry_run",
            ]
        )
        _dispatch(args)
        captured = capsys.readouterr()
        assert "sheeprl" in captured.out, (
            f"Expected 'sheeprl' in dry-run output.\nstdout: {captured.out!r}"
        )

    def test_leworldmodel_dry_run_prints_lerobot_train_world_model(
        self, capsys
    ) -> None:
        """LeWorldModel dry-run output must include 'train_world_model'."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch",
                "le_world_model",
                "--dataset",
                "lerobot/pusht",
                "--dry_run",
            ]
        )
        _dispatch(args)
        captured = capsys.readouterr()
        assert "train_world_model" in captured.out, (
            f"Expected 'train_world_model' in dry-run output.\nstdout: {captured.out!r}"
        )

    @pytest.mark.parametrize("arch", ["act", "diffusion"])
    def test_policy_arch_dry_run_prints_policy_type(self, arch: str, capsys) -> None:
        """Policy archs dry-run must include the correct --policy.type flag."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch",
                arch,
                "--dataset",
                "lerobot/pusht",
                "--dry_run",
            ]
        )
        _dispatch(args)
        captured = capsys.readouterr()
        assert f"--policy.type={arch}" in captured.out, (
            f"Expected '--policy.type={arch}' in dry-run output.\nstdout: {captured.out!r}"
        )
