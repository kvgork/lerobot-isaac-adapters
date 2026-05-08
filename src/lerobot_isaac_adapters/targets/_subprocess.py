"""
_subprocess
===========

Internal helper shared by the three training-target modules
(``policy_lerobot``, ``wm_dreamerv3``, ``wm_leworldmodel``).

Each target dispatches a long-running training subprocess and re-emits
``<metric>=<value>`` lines so that ``autoresearch-ml-executor-worker`` can
parse them.  The Popen/stream/regex/emit/wait/error pattern was identical
across the three targets; this module owns the single canonical version.

Usage
-----
>>> import re
>>> _METRIC_RE = re.compile(r"pc_success[=:\\s]+([0-9.eE+\\-]+)")
>>> rc = stream_training_subprocess(
...     cmd=["lerobot-train", "--policy.type=smolvla"],
...     metric_re=_METRIC_RE,
...     metric_name="pc_success",
...     label="policy_lerobot",
...     install_hint="Install LeRobot: pip install lerobot",
... )

Notes
-----
- Stdout is mirrored line-by-line to ``sys.stdout`` (no buffering).
- Stderr is merged into stdout (``stderr=STDOUT``) so the regex sees both
  log streams.
- ``FileNotFoundError`` is mapped to exit code ``127`` per the POSIX
  convention for "command not found".
- The helper does NOT enforce a timeout; per-target wrappers and the
  autoresearch executor enforce their own budgets.
"""

from __future__ import annotations

import re
import subprocess
import sys


def stream_training_subprocess(
    cmd: list[str],
    *,
    metric_re: re.Pattern[str],
    metric_name: str,
    label: str,
    install_hint: str,
) -> int:
    """Run ``cmd``, stream stdout, parse and re-emit metric values.

    Parameters
    ----------
    cmd:
        Subprocess argv vector.  First element must be discoverable on PATH.
    metric_re:
        Compiled regex with one capture group containing the metric value.
    metric_name:
        Canonical metric name forwarded to ``metric_extractor.emit``.
        Examples: ``pc_success``, ``recon_loss``, ``pred_loss``.
    label:
        Short module label (e.g. ``policy_lerobot``) used in error messages.
    install_hint:
        One-line installation hint shown when the subprocess executable is
        missing.  Example: ``"Install LeRobot: pip install lerobot"``.

    Returns
    -------
    int
        Subprocess exit code, ``127`` if the executable was not found.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print(
            f"[{label}] ERROR: {cmd[0]!r} not found in PATH. {install_hint}",
            file=sys.stderr,
        )
        return 127

    # Imported lazily so test stubs that monkeypatch sys.stdout still work.
    from lerobot_isaac_adapters.metric_extractor import emit

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        m = metric_re.search(line)
        if m:
            try:
                emit(metric_name, float(m.group(1)))
            except ValueError:
                # Captured group was not a finite float — leave the line
                # in place so the caller's stdout still shows the value
                # but do not corrupt the metric stream.
                pass

    proc.wait()
    if proc.returncode != 0:
        print(
            f"\033[31m[{label}] Training failed (exit={proc.returncode}) "
            f"— see stdout above\033[0m",
            file=sys.stderr,
        )
    return proc.returncode
