"""Pure phase-transition logic for the residual-RL scripted-grasp controller.

Extracted from ``sheeprl_plugin.isaac_env`` so it can be unit-tested WITHOUT the
Isaac / gymnasium / torch stack (importing ``sheeprl_plugin`` pulls gymnasium via
``hdf5_env``). No third-party imports — stdlib only.

Why this exists (residual-RL launch fix, 2026-07-12)
----------------------------------------------------
The scripted-grasp controller drives the residual-RL base. Its transitions used to
be purely REACTIVE state gates (e.g. ``aligned = xy_to_tgt < 0.015``). Once the
scripted action is blended with the policy action and clamped to ``[-1, 1]``
(``scripts/_wm_isaac_entry.py``), the arm rate-limits and the gate NEVER clears, so
the machine sticks in APPROACH forever — 0 grasps, the diagnosed cup0 / S3 stall.

The demo controller (``scripts/_gen_sim_demos.py``), which succeeds ~67-80 %,
is OPEN-LOOP: it runs a fixed per-phase step schedule. This module mirrors that —
each phase advances on its state gate (fast path) OR on a hard per-phase step cap
(``PHASE_STEP_CAP``), guaranteeing monotonic forward progress so the machine can
never stall. Backward re-grasp resets are bounded by ``MAX_REGRASPS``.

Demo-schedule port (2026-07-20)
--------------------------------
The counts below (``PHASE_STEP_CAP`` + ``STABILIZE_STEPS`` / ``CLOSE_DWELL`` /
``HOLD_STEPS``) now mirror ``scripts/_gen_sim_demos.py`` ``rollout()`` (lines
224-235) EXACTLY: 50 approach-above + 90 descend + 30 settle + 80 close-ramp +
25 firm hold + 60 lift + 60 carry + 40 lower (= 435 steps) + a 50-step RELEASE
ramp (not capped here — it self-terminates on its own ramp counter in the
consumer). The prior values (STABILIZE 20 / CLOSE 60 with no HOLD / DESCEND 60 /
LIFT 90 / CARRY 150 / LOWER 60) were an earlier approximation; this port closes
the gap against the ~80 %-success reference implementation. Do not edit
``scripts/_gen_sim_demos.py`` — it is the frozen reference.
"""

from __future__ import annotations

# Fixed pick-place phase order. HOLD sits between CLOSE and LIFT — a firm
# closed-grip dwell at grasp depth before lifting (mirrors the demo's dedicated
# "hold grip" segment, which the earlier CLOSE-only cradle+hold conflation lacked).
PHASE_ORDER = [
    "APPROACH",
    "DESCEND",
    "STABILIZE",
    "CLOSE",
    "HOLD",
    "LIFT",
    "CARRY",
    "LOWER",
    "RELEASE",
]

# Fixed internal counts for the phases that self-terminate on a dwell counter.
STABILIZE_STEPS = 30  # settle (gripper open) at grasp depth before closing
CLOSE_DWELL = 80  # PURE close ramp (OPEN -> CLOSE) before the firm HOLD phase
HOLD_STEPS = 25  # firm closed grip at depth before lifting (demo "hold" segment)

# Hard per-phase step caps: the phase force-advances at the cap even if its state
# gate never clears (the stall fix). ``None`` ⇒ the phase self-terminates on its own
# fixed count (STABILIZE_STEPS / CLOSE_DWELL / HOLD_STEPS, and RELEASE's own ramp).
# Values mirror scripts/_gen_sim_demos.py's per-phase durations exactly (sum of the
# capped + dwell phases = 435 steps pre-RELEASE, + a 50-step release ramp ≈ 485
# total ⇒ use episodes ≥ ~600 steps). LIFT/CARRY/LOWER caps double as the open-loop
# schedule those phases run under (no early-exit gate expected to fire sooner in the
# demo-parity case).
PHASE_STEP_CAP: dict[str, int | None] = {
    "APPROACH": 50,
    "DESCEND": 90,
    "STABILIZE": None,
    "CLOSE": None,
    "HOLD": None,
    "LIFT": 60,
    "CARRY": 60,
    "LOWER": 40,
    "RELEASE": None,
}

# Cap on backward LIFT/CARRY→APPROACH re-grasp resets, so a bad grip cannot
# ping-pong forever under the blend (which would re-create the stall).
MAX_REGRASPS = 2


def phase_after(phase: str) -> str:
    """Next phase in the fixed order (clamped at the terminal RELEASE)."""
    i = PHASE_ORDER.index(phase)
    return PHASE_ORDER[min(i + 1, len(PHASE_ORDER) - 1)]


def next_phase(
    phase: str,
    phase_steps: int,
    close_count: int,
    regrasps: int,
    *,
    aligned: bool,
    ee_high: bool,
    at_depth: bool,
    obj_lifted: bool,
    ee_at_carry_height: bool,
    not_holding: bool,
    over_bin: bool,
    at_release_depth: bool,
) -> tuple[str, bool]:
    """Decide the next scripted-grasp phase.

    Returns ``(next_phase, do_regrasp_reset)``. ``do_regrasp_reset`` requests a
    bounded fall-back to APPROACH (die dropped mid-lift/carry). GUARANTEES forward
    progress: every phase with a non-``None`` cap force-advances once ``phase_steps``
    reaches the cap, so the machine can never stall.
    """
    cap = PHASE_STEP_CAP.get(phase)
    forced = cap is not None and phase_steps >= cap
    if phase == "APPROACH":
        if (aligned and ee_high) or forced:
            return "DESCEND", False
    elif phase == "DESCEND":
        if at_depth or forced:
            return "STABILIZE", False
    elif phase == "STABILIZE":
        if close_count >= STABILIZE_STEPS:
            return "CLOSE", False
    elif phase == "CLOSE":
        if close_count >= CLOSE_DWELL:
            return "HOLD", False
    elif phase == "HOLD":
        # Purely count-based, like STABILIZE/CLOSE — ignores all state gates so a
        # noisy obj_lifted/ee_at_carry_height/not_holding reading can't cut the
        # firm-hold dwell short (or trigger an early regrasp) before LIFT begins.
        if close_count >= HOLD_STEPS:
            return "LIFT", False
    elif phase == "LIFT":
        if not_holding and regrasps < MAX_REGRASPS:
            return "APPROACH", True
        if (obj_lifted and ee_at_carry_height) or forced:
            return "CARRY", False
    elif phase == "CARRY":
        if not_holding and regrasps < MAX_REGRASPS:
            return "APPROACH", True
        if over_bin or forced:
            return "LOWER", False
    elif phase == "LOWER":
        if at_release_depth or forced:
            return "RELEASE", False
    # RELEASE (or unknown): terminal — hold.
    return phase, False
