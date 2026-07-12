"""Tests for the pretrained world-model fine-tune recipe auto-injection.

Covers the GPU-verified 2026-07-11 vla_jepa RTX-3080 recipe now auto-applied by
``policy_lerobot``: freeze_qwen + reinit_modules flag injection and camera-count
config adaptation. Pure helpers are unit-tested here; the dry-run dispatch
behaviour is covered in ``test_train_argparse.py::TestWorldModelPolicies``.
"""

from __future__ import annotations

import json

import pytest

from lerobot_isaac_adapters.targets import policy_lerobot as pl


class TestPatchWmPolicyConfig:
    """`patch_wm_policy_config` — pure camera-set down-adaptation."""

    def _cfg(self, cams: list[str]) -> dict:
        feats = {"observation.state": {"shape": [12]}}
        for c in cams:
            feats[c] = {"shape": [3, 480, 640], "type": "VISUAL"}
        return {"input_features": feats, "type": "vla_jepa"}

    def test_two_to_one_camera_down_adaptation(self) -> None:
        cfg = self._cfg(["observation.images.exterior_1_left", "observation.images.exterior_2_left"])
        out = pl.patch_wm_policy_config(cfg, ["observation.images.overhead"])
        img = [k for k in out["input_features"] if k.startswith("observation.images.")]
        assert img == ["observation.images.overhead"]
        # non-image feature preserved
        assert "observation.state" in out["input_features"]
        # shape template carried over from the pretrain image feature
        assert out["input_features"]["observation.images.overhead"]["shape"] == [3, 480, 640]

    def test_noop_when_counts_match(self) -> None:
        cfg = self._cfg(["observation.images.overhead"])
        out = pl.patch_wm_policy_config(cfg, ["observation.images.overhead"])
        assert sorted(out["input_features"]) == sorted(cfg["input_features"])

    def test_noop_when_target_empty(self) -> None:
        cfg = self._cfg(["observation.images.a", "observation.images.b"])
        out = pl.patch_wm_policy_config(cfg, [])
        img = [k for k in out["input_features"] if k.startswith("observation.images.")]
        assert len(img) == 2  # unchanged

    def test_noop_when_target_would_add_cameras(self) -> None:
        cfg = self._cfg(["observation.images.a"])
        out = pl.patch_wm_policy_config(cfg, ["observation.images.a", "observation.images.b"])
        img = [k for k in out["input_features"] if k.startswith("observation.images.")]
        assert len(img) == 1  # never invents camera weights

    def test_input_not_mutated(self) -> None:
        cfg = self._cfg(["observation.images.a", "observation.images.b"])
        before = json.dumps(cfg, sort_keys=True)
        pl.patch_wm_policy_config(cfg, ["observation.images.overhead"])
        assert json.dumps(cfg, sort_keys=True) == before


class TestDatasetCameraKeys:
    def test_reads_info_json(self, tmp_path) -> None:
        meta = tmp_path / "meta"
        meta.mkdir()
        (meta / "info.json").write_text(
            json.dumps(
                {
                    "features": {
                        "observation.state": {"shape": [12]},
                        "observation.images.overhead": {"shape": [3, 480, 640]},
                        "action": {"shape": [6]},
                    }
                }
            )
        )
        assert pl._dataset_camera_keys(str(tmp_path)) == ["observation.images.overhead"]

    def test_missing_root_returns_empty(self, tmp_path) -> None:
        assert pl._dataset_camera_keys(str(tmp_path / "nope")) == []

    def test_none_returns_empty(self) -> None:
        assert pl._dataset_camera_keys(None) == []


class TestMaterializePatchedCheckpoint:
    def test_writes_patched_config_and_symlinks(self, tmp_path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text(
            json.dumps(
                {
                    "input_features": {
                        "observation.state": {"shape": [12]},
                        "observation.images.exterior_1_left": {"shape": [3, 480, 640]},
                        "observation.images.exterior_2_left": {"shape": [3, 480, 640]},
                    }
                }
            )
        )
        (src / "model.safetensors").write_bytes(b"weights")
        out = tmp_path / "out"
        result = pl._materialize_patched_wm_checkpoint(
            str(src), ["observation.images.overhead"], str(out)
        )
        assert result == str(out)
        patched = json.loads((out / "config.json").read_text())
        img = [k for k in patched["input_features"] if k.startswith("observation.images.")]
        assert img == ["observation.images.overhead"]
        # safetensors symlinked, not copied
        assert (out / "model.safetensors").is_symlink()
        assert (out / "model.safetensors").read_bytes() == b"weights"


class TestAugmentPretrainedWm:
    """`_augment_pretrained_wm` — flag injection + opt-out (no filesystem)."""

    def test_injects_freeze_and_reinit_for_vla_jepa_with_policy_path(self) -> None:
        out = pl._augment_pretrained_wm(
            "vla_jepa",
            ["--policy.path=lerobot/VLA-JEPA-Pretrain"],
            dataset_root=None,
            output_dir="/tmp/o",
            dry_run=True,
        )
        assert "--policy.freeze_qwen=true" in out
        assert any(a.startswith("--policy.reinit_modules=[") for a in out)

    def test_no_injection_without_policy_path(self) -> None:
        out = pl._augment_pretrained_wm(
            "vla_jepa", [], dataset_root=None, output_dir="/tmp/o", dry_run=True
        )
        assert out == []

    def test_no_injection_for_other_wm_archs(self) -> None:
        # freeze_qwen is a vla_jepa attribute; must not leak to fastwam/lingbot_va.
        for arch in ("fastwam", "lingbot_va", "act", "smolvla"):
            out = pl._augment_pretrained_wm(
                arch,
                ["--policy.path=some/ckpt"],
                dataset_root=None,
                output_dir="/tmp/o",
                dry_run=True,
            )
            assert out == ["--policy.path=some/ckpt"]

    def test_respects_user_freeze_qwen(self) -> None:
        out = pl._augment_pretrained_wm(
            "vla_jepa",
            ["--policy.path=X", "--policy.freeze_qwen=false"],
            dataset_root=None,
            output_dir="/tmp/o",
            dry_run=True,
        )
        assert out.count("--policy.freeze_qwen=false") == 1
        assert "--policy.freeze_qwen=true" not in out

    def test_respects_user_reinit_modules(self) -> None:
        out = pl._augment_pretrained_wm(
            "vla_jepa",
            ["--policy.path=X", '--policy.reinit_modules=["custom.head"]'],
            dataset_root=None,
            output_dir="/tmp/o",
            dry_run=True,
        )
        assert sum(a.startswith("--policy.reinit_modules=") for a in out) == 1
        assert '--policy.reinit_modules=["custom.head"]' in out

    def test_env_optout(self, monkeypatch) -> None:
        monkeypatch.setenv("LEROBOT_ISAAC_WM_AUTORECIPE", "0")
        out = pl._augment_pretrained_wm(
            "vla_jepa",
            ["--policy.path=X"],
            dataset_root=None,
            output_dir="/tmp/o",
            dry_run=True,
        )
        assert out == ["--policy.path=X"]

    def test_camera_patch_rewrites_local_policy_path(self, tmp_path) -> None:
        # local pretrained ckpt with 2 cams + a dataset with 1 cam -> path rewritten.
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(
            json.dumps(
                {
                    "input_features": {
                        "observation.images.exterior_1_left": {"shape": [3, 480, 640]},
                        "observation.images.exterior_2_left": {"shape": [3, 480, 640]},
                    }
                }
            )
        )
        (ckpt / "model.safetensors").write_bytes(b"w")
        ds = tmp_path / "ds"
        (ds / "meta").mkdir(parents=True)
        (ds / "meta" / "info.json").write_text(
            json.dumps({"features": {"observation.images.overhead": {"shape": [3, 480, 640]}}})
        )
        out_dir = tmp_path / "out"
        result = pl._augment_pretrained_wm(
            "vla_jepa",
            [f"--policy.path={ckpt}"],
            dataset_root=str(ds),
            output_dir=str(out_dir),
            dry_run=False,
        )
        patched_path = out_dir / "_wm_policy_patched"
        assert f"--policy.path={patched_path}" in result
        patched_cfg = json.loads((patched_path / "config.json").read_text())
        img = [k for k in patched_cfg["input_features"] if k.startswith("observation.images.")]
        assert img == ["observation.images.overhead"]

    def test_camera_patch_dry_run_no_write(self, tmp_path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(
            json.dumps(
                {
                    "input_features": {
                        "observation.images.a": {"shape": [3, 480, 640]},
                        "observation.images.b": {"shape": [3, 480, 640]},
                    }
                }
            )
        )
        ds = tmp_path / "ds"
        (ds / "meta").mkdir(parents=True)
        (ds / "meta" / "info.json").write_text(
            json.dumps({"features": {"observation.images.overhead": {"shape": [3, 480, 640]}}})
        )
        out_dir = tmp_path / "out"
        result = pl._augment_pretrained_wm(
            "vla_jepa",
            [f"--policy.path={ckpt}"],
            dataset_root=str(ds),
            output_dir=str(out_dir),
            dry_run=True,
        )
        # path rewritten in the command, but nothing written to disk during dry-run.
        assert f"--policy.path={out_dir / '_wm_policy_patched'}" in result
        assert not (out_dir / "_wm_policy_patched").exists()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
