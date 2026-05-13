"""
test_targets_subprocess.py
==========================

Unit tests for training target modules.
Uses ``unittest.mock.patch("subprocess.Popen")`` to verify each target's
``run()`` builds the correct subprocess command without actually spawning
a process.

Covers:
- policy_lerobot: correct argv, policy.type flag, passthrough args
- wm_dreamerv3: correct sheeprl argv (dry_run path)
- wm_leworldmodel: correct lerobot.scripts.train_world_model argv (dry_run path)
- returncode propagation from subprocess
- FileNotFoundError → 127 for policy_lerobot
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch


from lerobot_isaac_adapters.targets import policy_lerobot, wm_dreamerv3, wm_leworldmodel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal Namespace for target run() calls."""
    defaults = dict(
        target_arch="smolvla",
        dataset="lerobot/pusht",
        config=None,
        output_dir="/tmp/test_output",
        steps=1000,
        batch_size=8,
        lr=1e-4,
        seed=42,
        dry_run=False,
        remainder=[],
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _mock_popen(returncode: int = 0, stdout_lines: list[str] | None = None):
    """Create a mock Popen that yields given stdout lines and exits with returncode."""
    mock_proc = MagicMock()
    mock_proc.stdout = iter(stdout_lines or [])
    mock_proc.returncode = returncode
    mock_proc.wait.return_value = returncode
    mock_popen = MagicMock(return_value=mock_proc)
    return mock_popen, mock_proc


# ---------------------------------------------------------------------------
# policy_lerobot
# ---------------------------------------------------------------------------


class TestPolicyLerobotSubprocess:
    """policy_lerobot.run() must build the correct lerobot-train command."""

    def test_smolvla_command(self) -> None:
        mock_popen, _ = _mock_popen()
        args = _make_args(target_arch="smolvla")

        with patch("subprocess.Popen", mock_popen):
            rc = policy_lerobot.run(args)

        assert rc == 0
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "lerobot-train"
        assert "--policy.type=smolvla" in cmd
        assert f"--dataset.repo_id={args.dataset}" in cmd
        # lerobot >= 0.5 CLI: --batch_size / --steps / --optimizer.lr
        # (was --training.batch_size / --training.num_steps / --training.lr in
        # older releases).
        assert f"--batch_size={args.batch_size}" in cmd
        assert f"--steps={args.steps}" in cmd
        assert f"--optimizer.lr={args.lr}" in cmd
        assert f"--seed={args.seed}" in cmd
        assert f"--output_dir={args.output_dir}" in cmd

    def test_act_policy_type(self) -> None:
        mock_popen, _ = _mock_popen()
        args = _make_args(target_arch="act")

        with patch("subprocess.Popen", mock_popen):
            policy_lerobot.run(args)

        cmd = mock_popen.call_args[0][0]
        assert "--policy.type=act" in cmd

    def test_diffusion_policy_type(self) -> None:
        mock_popen, _ = _mock_popen()
        args = _make_args(target_arch="diffusion")

        with patch("subprocess.Popen", mock_popen):
            policy_lerobot.run(args)

        cmd = mock_popen.call_args[0][0]
        assert "--policy.type=diffusion" in cmd

    def test_passthrough_args_appended(self) -> None:
        mock_popen, _ = _mock_popen()
        args = _make_args(
            target_arch="act",
            remainder=["--", "--policy.n_action_steps=100"],
        )

        with patch("subprocess.Popen", mock_popen):
            policy_lerobot.run(args)

        cmd = mock_popen.call_args[0][0]
        assert "--policy.n_action_steps=100" in cmd
        # The separator '--' should be stripped
        assert "--" not in cmd

    def test_config_arg_included(self) -> None:
        mock_popen, _ = _mock_popen()
        args = _make_args(target_arch="act", config="/tmp/my_config.yaml")

        with patch("subprocess.Popen", mock_popen):
            policy_lerobot.run(args)

        cmd = mock_popen.call_args[0][0]
        # lerobot >= 0.5 CLI: --config_path (was --config in older releases).
        assert "--config_path=/tmp/my_config.yaml" in cmd

    def test_nonzero_returncode_propagated(self) -> None:
        mock_popen, _ = _mock_popen(returncode=1)
        args = _make_args(target_arch="smolvla")

        with patch("subprocess.Popen", mock_popen):
            rc = policy_lerobot.run(args)

        assert rc == 1

    def test_file_not_found_returns_127(self) -> None:
        """FileNotFoundError (lerobot-train not in PATH) must return 127."""
        args = _make_args(target_arch="smolvla")

        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            rc = policy_lerobot.run(args)

        assert rc == 127

    def test_metric_re_emitted(self, capsys) -> None:
        """pc_success lines from stdout are re-emitted via metric_extractor."""
        mock_popen, _ = _mock_popen(
            returncode=0,
            stdout_lines=["eval/pc_success=0.75\n", "other line\n"],
        )
        args = _make_args(target_arch="smolvla")

        with patch("subprocess.Popen", mock_popen):
            policy_lerobot.run(args)

        captured = capsys.readouterr()
        assert "pc_success=0.75" in captured.out

    def test_dry_run_prints_lerobot_train_no_popen(self, capsys) -> None:
        """dry_run must print the command and NOT call Popen."""
        mock_popen = MagicMock()
        args = _make_args(target_arch="smolvla", dry_run=True)

        with patch("subprocess.Popen", mock_popen):
            rc = policy_lerobot.run(args)

        assert rc == 0
        mock_popen.assert_not_called()
        captured = capsys.readouterr()
        assert "lerobot-train" in captured.out

    def test_video_backend_defaults_to_pyav(self) -> None:
        """video_backend default must be pyav (avoid torchcodec libavutil mismatch)."""
        mock_popen, _ = _mock_popen()
        args = _make_args(target_arch="smolvla")

        with patch("subprocess.Popen", mock_popen):
            policy_lerobot.run(args)

        cmd = mock_popen.call_args[0][0]
        assert "--dataset.video_backend=pyav" in cmd

    def test_video_backend_override_via_namespace(self) -> None:
        """Caller can override default by setting args.video_backend."""
        mock_popen, _ = _mock_popen()
        args = _make_args(target_arch="smolvla", video_backend="torchcodec")

        with patch("subprocess.Popen", mock_popen):
            policy_lerobot.run(args)

        cmd = mock_popen.call_args[0][0]
        assert "--dataset.video_backend=torchcodec" in cmd
        assert "--dataset.video_backend=pyav" not in cmd


# ---------------------------------------------------------------------------
# wm_dreamerv3 — dry_run path (no conversion needed)
# ---------------------------------------------------------------------------


class TestWmDreamerv3DryRun:
    """wm_dreamerv3.run() dry_run must print both steps without spawning processes."""

    def test_dry_run_shows_sheeprl(self, capsys) -> None:
        args = _make_args(target_arch="dreamerv3", dry_run=True)
        rc = wm_dreamerv3.run(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "sheeprl" in captured.out

    def test_dry_run_shows_conversion_step(self, capsys) -> None:
        args = _make_args(
            target_arch="dreamerv3",
            dataset="lerobot/pusht",
            dry_run=True,
        )
        rc = wm_dreamerv3.run(args)

        assert rc == 0
        captured = capsys.readouterr()
        # Should mention dataset conversion
        assert "Step 1" in captured.out or "convert" in captured.out.lower()

    def test_dry_run_hdf5_skips_conversion_message(self, capsys) -> None:
        """If dataset is already an HDF5, dry_run shows pre-converted note."""
        args = _make_args(
            target_arch="dreamerv3",
            dataset="/tmp/data.hdf5",
            dry_run=True,
        )
        rc = wm_dreamerv3.run(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "pre-converted" in captured.out.lower() or "hdf5" in captured.out.lower()

    def test_dry_run_train_cmd_includes_correct_args(self, capsys) -> None:
        args = _make_args(
            target_arch="dreamerv3",
            dataset="lerobot/pusht",
            steps=500,
            batch_size=4,
            lr=3e-4,
            seed=7,
            output_dir="/tmp/dv3_out",
            dry_run=True,
        )
        rc = wm_dreamerv3.run(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "exp=dreamer_v3" in captured.out
        assert "500" in captured.out
        assert "/tmp/dv3_out" in captured.out


# ---------------------------------------------------------------------------
# wm_leworldmodel — dry_run path
# ---------------------------------------------------------------------------


class TestWmLeWorldModelDryRun:
    """wm_leworldmodel.run() dry_run must print both steps without spawning processes."""

    def test_dry_run_shows_train_world_model(self, capsys) -> None:
        args = _make_args(target_arch="le_world_model", dry_run=True)
        rc = wm_leworldmodel.run(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "train_world_model" in captured.out

    def test_dry_run_shows_conversion_step(self, capsys) -> None:
        args = _make_args(
            target_arch="le_world_model",
            dataset="lerobot/pusht",
            dry_run=True,
        )
        rc = wm_leworldmodel.run(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "Step 1" in captured.out or "convert" in captured.out.lower()

    def test_dry_run_hdf5_skips_conversion_message(self, capsys) -> None:
        args = _make_args(
            target_arch="le_world_model",
            dataset="/tmp/data.h5",
            dry_run=True,
        )
        rc = wm_leworldmodel.run(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "pre-converted" in captured.out.lower() or "hdf5" in captured.out.lower()

    def test_dry_run_train_cmd_includes_steps_and_output(self, capsys) -> None:
        args = _make_args(
            target_arch="le_world_model",
            dataset="lerobot/pusht",
            steps=200,
            batch_size=4,
            output_dir="/tmp/lwm_out",
            dry_run=True,
        )
        rc = wm_leworldmodel.run(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "200" in captured.out
        assert "/tmp/lwm_out" in captured.out


# ---------------------------------------------------------------------------
# wm_dreamerv3 — real run subprocess wiring
# ---------------------------------------------------------------------------


class TestWmDreamerv3Subprocess:
    """wm_dreamerv3.run() real path must call sheeprl with correct args."""

    def test_sheeprl_command_built_correctly(self, tmp_path) -> None:
        hdf5_path = tmp_path / "dreamerv3_data.hdf5"
        # Create a fake HDF5 so conversion is skipped
        hdf5_path.touch()

        mock_popen, _ = _mock_popen()
        args = _make_args(
            target_arch="dreamerv3",
            dataset=str(hdf5_path),  # pre-converted HDF5
            output_dir=str(tmp_path),
            steps=100,
            batch_size=4,
            lr=1e-3,
            seed=0,
        )

        with patch("subprocess.Popen", mock_popen):
            rc = wm_dreamerv3.run(args)

        assert rc == 0
        cmd = mock_popen.call_args[0][0]
        # Verify sheeprl is invoked via `python -m sheeprl` (not `sheeprl.cli`,
        # which exits silently without dispatching the hydra-decorated run()).
        assert "sheeprl" in cmd, f"sheeprl module not in cmd: {cmd}"
        assert "sheeprl.cli" not in cmd, f"must use `-m sheeprl`, got cmd: {cmd}"
        assert "exp=dreamer_v3" in cmd
        assert "algo.total_steps=100" in cmd

    def test_sheeprl_nonzero_returncode_propagated(self, tmp_path) -> None:
        hdf5_path = tmp_path / "dreamerv3_data.hdf5"
        hdf5_path.touch()
        mock_popen, _ = _mock_popen(returncode=2)
        args = _make_args(
            target_arch="dreamerv3",
            dataset=str(hdf5_path),
            output_dir=str(tmp_path),
        )

        with patch("subprocess.Popen", mock_popen):
            rc = wm_dreamerv3.run(args)

        assert rc == 2

    def test_file_not_found_returns_127(self, tmp_path) -> None:
        hdf5_path = tmp_path / "dreamerv3_data.hdf5"
        hdf5_path.touch()
        args = _make_args(
            target_arch="dreamerv3",
            dataset=str(hdf5_path),
            output_dir=str(tmp_path),
        )

        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            rc = wm_dreamerv3.run(args)

        assert rc == 127

    def test_recon_loss_re_emitted(self, tmp_path, capsys) -> None:
        """recon_loss lines from sheeprl stdout are re-emitted."""
        hdf5_path = tmp_path / "dreamerv3_data.hdf5"
        hdf5_path.touch()
        mock_popen, _ = _mock_popen(
            returncode=0,
            stdout_lines=["recon_loss=0.042\n", "step=100\n"],
        )
        args = _make_args(
            target_arch="dreamerv3",
            dataset=str(hdf5_path),
            output_dir=str(tmp_path),
        )

        with patch("subprocess.Popen", mock_popen):
            wm_dreamerv3.run(args)

        captured = capsys.readouterr()
        assert "recon_loss=0.042" in captured.out


# ---------------------------------------------------------------------------
# wm_leworldmodel — real run subprocess wiring
# ---------------------------------------------------------------------------


class TestWmLeWorldModelSubprocess:
    """wm_leworldmodel.run() real path must call lerobot.scripts.train_world_model."""

    def test_train_world_model_command_built(self, tmp_path) -> None:
        hdf5_path = tmp_path / "leworldmodel_data.hdf5"
        hdf5_path.touch()
        mock_popen, _ = _mock_popen()
        args = _make_args(
            target_arch="le_world_model",
            dataset=str(hdf5_path),
            output_dir=str(tmp_path),
            steps=50,
        )

        with patch("subprocess.Popen", mock_popen):
            rc = wm_leworldmodel.run(args)

        assert rc == 0
        cmd = mock_popen.call_args[0][0]
        assert "lerobot.scripts.train_world_model" in " ".join(cmd)
        assert any("50" in part for part in cmd)

    def test_nonzero_returncode_propagated(self, tmp_path) -> None:
        hdf5_path = tmp_path / "leworldmodel_data.hdf5"
        hdf5_path.touch()
        mock_popen, _ = _mock_popen(returncode=3)
        args = _make_args(
            target_arch="le_world_model",
            dataset=str(hdf5_path),
            output_dir=str(tmp_path),
        )

        with patch("subprocess.Popen", mock_popen):
            rc = wm_leworldmodel.run(args)

        assert rc == 3

    def test_file_not_found_returns_127(self, tmp_path) -> None:
        hdf5_path = tmp_path / "leworldmodel_data.hdf5"
        hdf5_path.touch()
        args = _make_args(
            target_arch="le_world_model",
            dataset=str(hdf5_path),
            output_dir=str(tmp_path),
        )

        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            rc = wm_leworldmodel.run(args)

        assert rc == 127

    def test_pred_loss_re_emitted(self, tmp_path, capsys) -> None:
        """pred_loss lines from stdout are re-emitted."""
        hdf5_path = tmp_path / "leworldmodel_data.hdf5"
        hdf5_path.touch()
        mock_popen, _ = _mock_popen(
            returncode=0,
            stdout_lines=["pred_loss=0.018\n", "epoch=5\n"],
        )
        args = _make_args(
            target_arch="le_world_model",
            dataset=str(hdf5_path),
            output_dir=str(tmp_path),
        )

        with patch("subprocess.Popen", mock_popen):
            wm_leworldmodel.run(args)

        captured = capsys.readouterr()
        assert "pred_loss=0.018" in captured.out
