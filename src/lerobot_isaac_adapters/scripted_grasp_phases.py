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

Demo-schedule port (2026-07-20; SETTLE added 2026-07-24)
--------------------------------------------------------
The counts below (``PHASE_STEP_CAP`` + ``SETTLE_STEPS`` / ``STABILIZE_STEPS`` /
``CLOSE_DWELL`` / ``HOLD_STEPS``) now mirror ``scripts/_gen_sim_demos.py``
``rollout()`` (lines 224-235) EXACTLY, plus the episode-start settle both the
probe and demo-gen run before approaching: 30 episode-start settle (SETTLE) +
50 approach-above + 90 descend + 30 settle-at-depth + 80 close-ramp + 25 firm
hold + 60 lift + 60 carry + 40 lower (30+50+90+30+80+25+60+60+40 = 465 steps
pre-RELEASE) + a 50-step RELEASE ramp (not capped here — it self-terminates on
its own ramp counter in the consumer), ≈ 515 total; episodes ≥ 600 still fine.
The prior values (STABILIZE 20 / CLOSE 60 with no HOLD / DESCEND 60 /
LIFT 90 / CARRY 150 / LOWER 60) were an earlier approximation; this port closes
the gap against the ~80 %-success reference implementation. Do not edit
``scripts/_gen_sim_demos.py`` — it is the frozen reference.
"""

from __future__ import annotations

# Fixed pick-place phase order. SETTLE is the episode-start open-grip dwell
# (probe/demo-gen parity — see SETTLE_STEPS). HOLD sits between CLOSE and LIFT —
# a firm closed-grip dwell at grasp depth before lifting (mirrors the demo's
# dedicated "hold grip" segment, which the earlier CLOSE-only cradle+hold
# conflation lacked).
PHASE_ORDER = [
    "SETTLE",
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
# SETTLE: episode-start open-grip settle (zero arm action) — probe/demo-gen
# parity; DR reset leaves joints/die in a transient the June-validated
# kinematics never faced.
SETTLE_STEPS = 30
STABILIZE_STEPS = 30  # settle (gripper open) at grasp depth before closing
CLOSE_DWELL = 80  # PURE close ramp (OPEN -> CLOSE) before the firm HOLD phase
HOLD_STEPS = 25  # firm closed grip at depth before lifting (demo "hold" segment)

# Hard per-phase step caps: the phase force-advances at the cap even if its state
# gate never clears (the stall fix). ``None`` ⇒ the phase self-terminates on its own
# fixed count (SETTLE_STEPS / STABILIZE_STEPS / CLOSE_DWELL / HOLD_STEPS, and
# RELEASE's own ramp). Values mirror scripts/_gen_sim_demos.py's per-phase durations
# exactly (sum of the capped + dwell phases = 465 steps pre-RELEASE, + a 50-step
# release ramp ≈ 515 total ⇒ use episodes ≥ ~600 steps). LIFT/CARRY/LOWER caps
# double as the open-loop schedule those phases run under (no early-exit gate
# expected to fire sooner in the demo-parity case).
PHASE_STEP_CAP: dict[str, int | None] = {
    "SETTLE": None,
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
    if phase == "SETTLE":
        # Purely count-based, like STABILIZE — ignores all state gates; the
        # episode-start transient must dwell out regardless of noisy readings.
        if close_count >= SETTLE_STEPS:
            return "APPROACH", False
    elif phase == "APPROACH":
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


def format_phase_transition(
    nxt: str,
    prev: str,
    *,
    obj_lifted: bool,
    oz: float,
    ez: float,
    t: int,
    regrasp: bool = False,
) -> str:
    """Gate-parseable phase-transition line.

    CONTRACT: scripts/_residual_smoke_gate.sh (workspace) parses this with
    ``\\[script-dbg\\] phase=(\\w+) obj_lifted=(\\w+) oz=([\\-0-9.]+) ez=([\\-0-9.]+)``
    — the prefix through ``ez=`` must not change shape. Extra fields go after.
    ``t`` is the EPISODE-RELATIVE step (de-aliases the old fixed-150 cadence,
    which sampled the same episode offset every time on 301-step episodes).
    """
    tail = " REGRASP" if regrasp else ""
    return (
        f"[script-dbg] phase={nxt} obj_lifted={obj_lifted} "
        f"oz={oz:.3f} ez={ez:.3f} t={t} prev={prev}{tail}"
    )
# Phases where the actor may share authority. Mirrors demo-gen's DAgger noise
# gating (scripts/_gen_sim_demos.py): noise only on approach/lift/carry/lower;
# grasp-critical segments (descend/settle/close/hold/release) run script-pure.
# residual-rl-v2 post-mortem (2026-07-22): UNIFORM blending broke the grasp at
# any meaningful actor share — 0 carries in 10k steps.
# LIFT moved to grasp-critical 2026-07-23 — the freshly-closed grip is the most
# slip-fragile moment (v2 slip class); demo-gen's lift "noise OK" was a SMALL
# additive perturbation, not authority replacement. APPROACH stays safe because
# a wandered approach recovers via the force-advance into the script-pure
# DESCEND (target re-latched, full authority).
# APPROACH moved to grasp-critical 2026-07-24 — residual-rl-v3 trace: blended
# approach (frac 0.8-0.93, newborn actor) left the arm at ez~0.31 instead of
# staging at z_high 0.19, so DESCEND crossed the at_depth boundary mid-flight
# and STABILIZE's fresh IK segment stalled at the 0.121 DLS equilibrium (the R4
# freeze, reintroduced via bad staging). Actor authority remains on CARRY/LOWER
# — the residual's actual learning targets.
# SETTLE (2026-07-24) is also critical — the zero-action settle must not be
# perturbed (its whole point is a script-pure, actionless transient dwell).
BLEND_SAFE_PHASES = frozenset({"CARRY", "LOWER"})


def blend_fraction(phase: str, script_frac: float) -> float:
    """Effective script weight for this step.

    Decayed ``script_frac`` applies only on blend-safe phases; grasp-critical
    phases keep FULL script authority (1.0) for the whole run. The actor still
    learns grasping in imagination (Dreamer trains the policy on world-model
    rollouts, not on executed authority) — this gating protects the QUALITY of
    collected experience, which uniform blending destroyed.
    """
    return script_frac if phase in BLEND_SAFE_PHASES else 1.0
