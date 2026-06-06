"""
test_multi_dataset.py
=====================
Tests for the multi-dataset ``--datasets`` flag (Track B.2).

All tests run without lerobot installed — they exercise the argparse layer
and the dry-run path of the policy backend only.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout, redirect_stderr


from lerobot_isaac_adapters.train import _build_parser, _resolve_dataset_list, _dispatch


# ---------------------------------------------------------------------------
# argparse + normalisation
# ---------------------------------------------------------------------------


def test_datasets_flag_parses():
    parser = _build_parser()
    args = parser.parse_args(["--target_arch", "smolvla", "--datasets", "a,b"])
    assert args.datasets == ["a,b"]


def test_resolve_comma_separated():
    parser = _build_parser()
    args = parser.parse_args(["--target_arch", "smolvla", "--datasets", "a,b,c"])
    assert _resolve_dataset_list(args) == ["a", "b", "c"]


def test_resolve_repeatable():
    parser = _build_parser()
    args = parser.parse_args(
        ["--target_arch", "smolvla", "--datasets", "a", "--datasets", "b,c"]
    )
    assert _resolve_dataset_list(args) == ["a", "b", "c"]


def test_datasets_takes_precedence_over_dataset():
    parser = _build_parser()
    args = parser.parse_args(
        ["--target_arch", "smolvla", "--dataset", "x", "--datasets", "a,b"]
    )
    assert _resolve_dataset_list(args) == ["a", "b"]


def test_single_dataset_fallback():
    parser = _build_parser()
    args = parser.parse_args(["--target_arch", "smolvla", "--dataset", "x"])
    assert _resolve_dataset_list(args) == ["x"]


def test_no_dataset_empty_list():
    parser = _build_parser()
    args = parser.parse_args(["--target_arch", "smolvla"])
    assert _resolve_dataset_list(args) == []


# ---------------------------------------------------------------------------
# dry-run dispatch
# ---------------------------------------------------------------------------


def _dispatch_capture(argv: list[str]) -> tuple[int, str, str]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = _dispatch(args)
    return rc, out.getvalue(), err.getvalue()


def test_dry_run_multi_hf_repo_ids():
    rc, out, _ = _dispatch_capture(
        [
            "--target_arch",
            "smolvla",
            "--datasets",
            "lerobot/pusht,lerobot/aloha",
            "--dry_run",
        ]
    )
    assert rc == 0
    assert "multi-dataset (2)" in out
    assert "lerobot/pusht" in out and "lerobot/aloha" in out
    # comma-joined repo_id forwarded to lerobot-train
    assert "--dataset.repo_id=lerobot/pusht,lerobot/aloha" in out


def test_world_model_rejects_multi():
    rc, _, err = _dispatch_capture(
        [
            "--target_arch",
            "dreamerv3",
            "--datasets",
            "a,b",
            "--dry_run",
        ]
    )
    assert rc == 2
    assert "world-model" in err


def test_single_dataset_unchanged():
    """One dataset → no multi-dataset banner, plain repo_id."""
    rc, out, _ = _dispatch_capture(
        ["--target_arch", "smolvla", "--dataset", "lerobot/pusht", "--dry_run"]
    )
    assert rc == 0
    assert "multi-dataset" not in out
    assert "--dataset.repo_id=lerobot/pusht" in out
