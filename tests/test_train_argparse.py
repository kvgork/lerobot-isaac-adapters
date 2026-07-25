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
# Expected: smolvla, act, diffusion (plain policies),
#           vla_jepa, fastwam, lingbot_va (lerobot >=0.6.0 world-model policies),
#           dreamerv3, le_world_model (predictive world-model backends).
assert len(VALID_ARCHS) == 8, f"Expected 8 archs, got {VALID_ARCHS}"

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
        self, capsys, monkeypatch
    ) -> None:
        """LeWorldModel HF-backend dry-run output must include 'train_world_model'.
        (The default backend is the in-process _lewm_minimal trainer — lerobot
        0.6.0 still does not ship a standalone train_world_model CLI.)"""
        monkeypatch.setenv("LEROBOT_ISAAC_LEWM_BACKEND", "hf")
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

    @pytest.mark.parametrize(
        "arch", ["act", "diffusion", "vla_jepa", "fastwam", "lingbot_va"]
    )
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


class TestWorldModelPolicies:
    """lerobot >=0.6.0 world-model policies (vla_jepa / fastwam / lingbot_va).

    They dispatch through policy_lerobot (metric pc_success), so on the CLI they
    behave exactly like the plain policies — only --policy.type differs.
    """

    _WM_POLICIES = ["vla_jepa", "fastwam", "lingbot_va"]

    @pytest.mark.parametrize("arch", _WM_POLICIES)
    def test_wm_policy_dry_run_prints_lerobot_train_and_type(
        self, arch: str, capsys
    ) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            ["--target_arch", arch, "--dataset", "lerobot/pusht", "--dry_run"]
        )
        rc = _dispatch(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "lerobot-train" in out
        assert f"--policy.type={arch}" in out

    @pytest.mark.parametrize("arch", _WM_POLICIES)
    def test_policy_path_omits_policy_type(self, arch: str, capsys) -> None:
        """A pretrained checkpoint via --policy.path must suppress the auto
        --policy.type (passing both is a draccus conflict in lerobot 0.6.0)."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", arch, "--dataset", "lerobot/pusht", "--dry_run",
                "--", "--policy.path=lerobot/VLA-JEPA-Pretrain",
            ]
        )
        rc = _dispatch(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "--policy.path=lerobot/VLA-JEPA-Pretrain" in out
        assert "--policy.type=" not in out, (
            f"--policy.type must be omitted when --policy.path is set.\nstdout: {out!r}"
        )

    def test_vla_jepa_policy_path_auto_injects_recipe(self, capsys) -> None:
        """vla_jepa + a pretrained --policy.path auto-adds freeze_qwen + reinit
        (the GPU-verified RTX-3080 fine-tune recipe)."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "vla_jepa", "--dataset", "lerobot/pusht", "--dry_run",
                "--", "--policy.path=lerobot/VLA-JEPA-Pretrain",
            ]
        )
        assert _dispatch(args) == 0
        out = capsys.readouterr().out
        assert "--policy.freeze_qwen=true" in out
        assert "--policy.reinit_modules=[" in out

    def test_vla_jepa_from_scratch_no_recipe(self, capsys) -> None:
        """No --policy.path (train from scratch) => recipe not injected."""
        parser = _build_parser()
        args = parser.parse_args(
            ["--target_arch", "vla_jepa", "--dataset", "lerobot/pusht", "--dry_run"]
        )
        assert _dispatch(args) == 0
        out = capsys.readouterr().out
        assert "--policy.freeze_qwen" not in out
        assert "--policy.reinit_modules" not in out
        assert "--policy.type=vla_jepa" in out

    def test_fastwam_policy_path_no_freeze_qwen(self, capsys) -> None:
        """freeze_qwen is a vla_jepa attribute; must not leak to fastwam."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "fastwam", "--dataset", "lerobot/pusht", "--dry_run",
                "--", "--policy.path=some/ckpt",
            ]
        )
        assert _dispatch(args) == 0
        out = capsys.readouterr().out
        assert "--policy.freeze_qwen" not in out
        assert "--policy.reinit_modules" not in out

    def test_vla_jepa_user_freeze_qwen_not_overridden(self, capsys) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "vla_jepa", "--dataset", "lerobot/pusht", "--dry_run",
                "--", "--policy.path=X", "--policy.freeze_qwen=false",
            ]
        )
        assert _dispatch(args) == 0
        out = capsys.readouterr().out
        assert "--policy.freeze_qwen=false" in out
        assert "--policy.freeze_qwen=true" not in out


class TestLoraFlags:
    """LoRA / PEFT flag passthrough — Phase 1.4 contract."""

    def test_use_lora_default_false(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.use_lora is False

    def test_use_lora_flag_sets_true(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla", "--use_lora"])
        assert args.use_lora is True

    def test_lora_rank_default_is_8(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.lora_rank == 8

    def test_lora_rank_parsed_as_int(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla", "--lora_rank", "16"])
        assert args.lora_rank == 16

    def test_lora_alpha_default_is_16(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.lora_alpha == 16

    def test_lora_dropout_default_is_zero(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.lora_dropout == pytest.approx(0.0)

    def test_lora_target_modules_default_attn_qv(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.lora_target_modules == "attn_qv"

    def test_lora_dry_run_prints_lora_config(self, capsys):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "smolvla", "--dataset", "lerobot/pusht",
                "--use_lora", "--lora_rank", "16", "--lora_alpha", "32",
                "--lora_dropout", "0.05", "--lora_target_modules", "attn_qkvo",
                "--dry_run",
            ]
        )
        _dispatch(args)
        captured = capsys.readouterr()
        out = captured.out
        assert ("LoRA enabled" in out) or ("use_lora=True" in out), (
            f"Expected LoRA banner in dry-run output. stdout: {out!r}"
        )
        assert ("r=16" in out) or ("lora_rank=16" in out), (
            f"Expected rank=16 in dry-run output. stdout: {out!r}"
        )
        assert "attn_qkvo" in out

    @pytest.mark.parametrize("arch", ["act", "diffusion"])
    def test_use_lora_on_non_smolvla_warns(self, arch, capsys):
        """Phase 1.4 contract: --use_lora on non-smolvla warns, does not raise."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", arch, "--dataset", "lerobot/pusht",
                "--use_lora", "--dry_run",
            ]
        )
        rc = _dispatch(args)
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert "lora" in combined and "smolvla" in combined
        assert rc == 0


class TestSuccessesOnlyFlag:
    """--successes_only is an opt-in store_true flag, default False."""

    def test_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.successes_only is False

    def test_flag_sets_true(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            ["--target_arch", "smolvla", "--successes_only"]
        )
        assert args.successes_only is True


class TestP2eExpSelection:
    """Plan2Explore exp= selection — p2e_dv3 MVP plumbing (spec 2026-06-22)."""

    # ------------------------------------------------------------------
    # (a) --exp p2e_dv3_exploration + --dry_run
    # ------------------------------------------------------------------
    def test_exp_p2e_exploration_in_cmd(self, capsys) -> None:
        """--exp p2e_dv3_exploration → cmd contains exactly one exp=p2e_dv3_exploration,
        zero exp=dreamer_v3."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "dreamerv3",
                "--dataset", "lerobot/pusht",
                "--exp", "p2e_dv3_exploration",
                "--dry_run",
            ]
        )
        rc = _dispatch(args)
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "exp=p2e_dv3_exploration" in out, (
            f"Expected 'exp=p2e_dv3_exploration' in dry-run output.\nstdout: {out!r}"
        )
        assert "exp=dreamer_v3" not in out, (
            f"Unexpected 'exp=dreamer_v3' in dry-run output when --exp p2e_dv3_exploration set.\nstdout: {out!r}"
        )
        # Exactly one exp= token in the sheeprl cmd line
        sheeprl_line = next(
            (line for line in out.splitlines() if "sheeprl" in line), ""
        )
        exp_count = sheeprl_line.count("exp=")
        assert exp_count == 1, (
            f"Expected exactly 1 exp= in sheeprl cmd, got {exp_count}.\nLine: {sheeprl_line!r}"
        )

    # ------------------------------------------------------------------
    # (b) default (no --exp, no env var) → exp=dreamer_v3 (back-compat)
    # ------------------------------------------------------------------
    def test_default_exp_is_dreamer_v3(self, capsys) -> None:
        """No --exp and no env var → cmd emits exp=dreamer_v3 (unchanged)."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "dreamerv3",
                "--dataset", "lerobot/pusht",
                "--dry_run",
            ]
        )
        env_backup = os.environ.pop("LEROBOT_ISAAC_EXP", None)
        try:
            rc = _dispatch(args)
        finally:
            if env_backup is not None:
                os.environ["LEROBOT_ISAAC_EXP"] = env_backup
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "exp=dreamer_v3" in out, (
            f"Expected 'exp=dreamer_v3' in default dry-run output.\nstdout: {out!r}"
        )
        assert "exp=p2e_dv3" not in out, (
            f"Unexpected p2e exp= in default output.\nstdout: {out!r}"
        )

    # ------------------------------------------------------------------
    # (c) LEROBOT_ISAAC_EXP env var (no --exp flag) → resolves to p2e
    # ------------------------------------------------------------------
    def test_env_var_exp_resolves(self, capsys, monkeypatch) -> None:
        """LEROBOT_ISAAC_EXP=p2e_dv3_exploration (no --exp) → cmd has exp=p2e_dv3_exploration."""
        monkeypatch.setenv("LEROBOT_ISAAC_EXP", "p2e_dv3_exploration")
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "dreamerv3",
                "--dataset", "lerobot/pusht",
                "--dry_run",
            ]
        )
        rc = _dispatch(args)
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "exp=p2e_dv3_exploration" in out, (
            f"Expected 'exp=p2e_dv3_exploration' via env var.\nstdout: {out!r}"
        )
        assert "exp=dreamer_v3" not in out

    # ------------------------------------------------------------------
    # (d) leading exp=foo in remainder → adapter suppresses its own exp=
    # ------------------------------------------------------------------
    def test_remainder_exp_suppresses_adapter_exp(self, capsys) -> None:
        """exp=foo in remainder → adapter emits NO extra exp=; only one exp= in cmd."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "dreamerv3",
                "--dataset", "lerobot/pusht",
                "--dry_run",
                "--",
                "exp=p2e_dv3_exploration",
            ]
        )
        rc = _dispatch(args)
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out
        # The remainder exp= must appear exactly once — no duplicate from the adapter
        assert out.count("exp=p2e_dv3_exploration") >= 1, (
            f"Expected remainder exp= to pass through.\nstdout: {out!r}"
        )
        # There must NOT be an additional exp=dreamer_v3 injected by the adapter
        assert "exp=dreamer_v3" not in out, (
            f"Adapter incorrectly emitted exp=dreamer_v3 alongside remainder exp=.\nstdout: {out!r}"
        )
        # Total exp= count in the sheeprl cmd line must be exactly 1
        sheeprl_line = next(
            (line for line in out.splitlines() if "sheeprl" in line), ""
        )
        exp_count = sheeprl_line.count("exp=")
        assert exp_count == 1, (
            f"Expected exactly 1 exp= in sheeprl cmd (double-exp guard failed), "
            f"got {exp_count}.\nLine: {sheeprl_line!r}"
        )

    # ------------------------------------------------------------------
    # (e-i) --exp p2e_dv3_finetuning without ckpt → error (rc != 0)
    # ------------------------------------------------------------------
    def test_finetuning_without_ckpt_errors(self, capsys, monkeypatch) -> None:
        """--exp p2e_dv3_finetuning without a ckpt source → returns non-zero."""
        monkeypatch.delenv("LEROBOT_ISAAC_EXPLORATION_CKPT", raising=False)
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "dreamerv3",
                "--dataset", "lerobot/pusht",
                "--exp", "p2e_dv3_finetuning",
                "--dry_run",
            ]
        )
        rc = _dispatch(args)
        assert rc != 0, "Expected non-zero rc when finetuning ckpt is missing"
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "finetuning" in combined.lower() or "exploration_ckpt" in combined.lower(), (
            f"Expected error message mentioning finetuning/ckpt.\noutput: {combined!r}"
        )

    # ------------------------------------------------------------------
    # (e-ii) --exp p2e_dv3_finetuning + --exploration_ckpt → appends ckpt_path
    # ------------------------------------------------------------------
    def test_finetuning_with_exploration_ckpt_flag(self, capsys, tmp_path) -> None:
        """--exp p2e_dv3_finetuning + --exploration_ckpt /x/y.ckpt → ckpt_path in cmd."""
        ckpt = tmp_path / "explore_ckpt"
        ckpt.mkdir()
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "dreamerv3",
                "--dataset", "lerobot/pusht",
                "--exp", "p2e_dv3_finetuning",
                "--exploration_ckpt", str(ckpt),
                "--dry_run",
            ]
        )
        rc = _dispatch(args)
        assert rc == 0, f"Expected rc=0 with valid ckpt. stderr: {capsys.readouterr().err!r}"
        captured = capsys.readouterr()
        out = captured.out
        assert "checkpoint.exploration_ckpt_path=" in out, (
            f"Expected 'checkpoint.exploration_ckpt_path=' in cmd.\nstdout: {out!r}"
        )
        assert "exp=p2e_dv3_finetuning" in out, (
            f"Expected 'exp=p2e_dv3_finetuning' in cmd.\nstdout: {out!r}"
        )

    # ------------------------------------------------------------------
    # (e-iii) --exp flag defaults to None (not set by default)
    # ------------------------------------------------------------------
    def test_exp_default_is_none(self) -> None:
        """--exp defaults to None when not provided."""
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "dreamerv3"])
        assert args.exp is None

    # ------------------------------------------------------------------
    # (e-iv) --exploration_ckpt flag defaults to None
    # ------------------------------------------------------------------
    def test_exploration_ckpt_default_is_none(self) -> None:
        """--exploration_ckpt defaults to None when not provided."""
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "dreamerv3"])
        assert args.exploration_ckpt is None

    # ------------------------------------------------------------------
    # (e-v) LEROBOT_ISAAC_EXPLORATION_CKPT env var used when no --exploration_ckpt
    # ------------------------------------------------------------------
    def test_finetuning_via_env_var_ckpt(self, capsys, monkeypatch, tmp_path) -> None:
        """LEROBOT_ISAAC_EXPLORATION_CKPT used when --exploration_ckpt not passed."""
        ckpt = tmp_path / "explore_env_ckpt"
        ckpt.mkdir()
        monkeypatch.setenv("LEROBOT_ISAAC_EXPLORATION_CKPT", str(ckpt))
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--target_arch", "dreamerv3",
                "--dataset", "lerobot/pusht",
                "--exp", "p2e_dv3_finetuning",
                "--dry_run",
            ]
        )
        rc = _dispatch(args)
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "checkpoint.exploration_ckpt_path=" in out, (
            f"Expected ckpt path in cmd via env var.\nstdout: {out!r}"
        )
