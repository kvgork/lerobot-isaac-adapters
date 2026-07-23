"""Unit tests for the residual-RL scripted-grasp phase machine.

These exercise the pure transition logic (no Isaac / gymnasium / torch), which is
why it was extracted to ``lerobot_isaac_adapters.scripted_grasp_phases``. The
headline invariant is the launch fix: **the machine can never stall** — every phase
with a step cap force-advances at the cap even when its state gate never clears
(the bug that pinned residual-RL in APPROACH under the blended, clamped action).

``TestDemoParity`` covers the 2026-07-20 port of the proven demo-gen grasp
schedule (``scripts/_gen_sim_demos.py`` ``rollout()``, lines 224-235) into this
module: the new HOLD phase and the demo-matched per-phase durations.
"""

from __future__ import annotations

import pytest

from lerobot_isaac_adapters import scripted_grasp_phases as p

# All gates False — the "stalled under the blend" condition.
_NO_GATES = dict(
    aligned=False,
    ee_high=False,
    at_depth=False,
    obj_lifted=False,
    ee_at_carry_height=False,
    not_holding=False,
    over_bin=False,
    at_release_depth=False,
)


class TestNoStall:
    """The launch fix: capped phases advance on the cap regardless of gates."""

    @pytest.mark.parametrize(
        "phase", [ph for ph, cap in p.PHASE_STEP_CAP.items() if cap is not None]
    )
    def test_capped_phase_force_advances_with_no_gates(self, phase: str) -> None:
        cap = p.PHASE_STEP_CAP[phase]
        # one step before the cap: still in the same phase (no gate met)
        nxt, regrasp = p.next_phase(
            phase, cap - 1, close_count=0, regrasps=0, **_NO_GATES
        )
        assert nxt == phase and not regrasp
        # at the cap: forced forward to the successor
        nxt, regrasp = p.next_phase(phase, cap, close_count=0, regrasps=0, **_NO_GATES)
        assert nxt == p.phase_after(phase) and not regrasp

    def test_full_sequence_terminates_within_step_budget(self) -> None:
        """Driving only the caps (no gates) walks APPROACH → RELEASE and stops."""
        phase, phase_steps, close_count, regrasps = "APPROACH", 0, 0, 0
        seen = [phase]
        for _ in range(2000):  # generous bound; must converge well inside it
            # advance the internal dwell counters for the fixed-count phases
            if phase in ("STABILIZE", "CLOSE", "HOLD"):
                close_count += 1
            nxt, regrasp = p.next_phase(
                phase, phase_steps, close_count, regrasps, **_NO_GATES
            )
            if regrasp or nxt != phase:
                phase, phase_steps, close_count = nxt, 0, 0
                seen.append(phase)
                if phase == "RELEASE":
                    break
            else:
                phase_steps += 1
        assert phase == "RELEASE", f"did not reach RELEASE; path={seen}"
        assert seen == p.PHASE_ORDER, f"unexpected path {seen}"


class TestGates:
    """State gates are the fast early-exit path (before the cap)."""

    def test_approach_advances_early_when_aligned_and_high(self) -> None:
        g = {**_NO_GATES, "aligned": True, "ee_high": True}
        assert p.next_phase("APPROACH", 1, 0, 0, **g) == ("DESCEND", False)

    def test_descend_advances_early_at_depth(self) -> None:
        g = {**_NO_GATES, "at_depth": True}
        assert p.next_phase("DESCEND", 1, 0, 0, **g) == ("STABILIZE", False)

    def test_stabilize_waits_for_dwell(self) -> None:
        assert p.next_phase("STABILIZE", 0, p.STABILIZE_STEPS - 1, 0, **_NO_GATES) == (
            "STABILIZE",
            False,
        )
        assert p.next_phase("STABILIZE", 0, p.STABILIZE_STEPS, 0, **_NO_GATES) == (
            "CLOSE",
            False,
        )

    def test_close_waits_for_dwell(self) -> None:
        assert p.next_phase("CLOSE", 0, p.CLOSE_DWELL - 1, 0, **_NO_GATES) == (
            "CLOSE",
            False,
        )
        assert p.next_phase("CLOSE", 0, p.CLOSE_DWELL, 0, **_NO_GATES) == (
            "HOLD",
            False,
        )

    def test_lift_advances_when_lifted_and_high(self) -> None:
        g = {**_NO_GATES, "obj_lifted": True, "ee_at_carry_height": True}
        assert p.next_phase("LIFT", 1, 0, 0, **g) == ("CARRY", False)

    def test_lift_does_not_advance_if_lifted_but_not_carry_height(self) -> None:
        g = {**_NO_GATES, "obj_lifted": True, "ee_at_carry_height": False}
        assert p.next_phase("LIFT", 1, 0, 0, **g) == ("LIFT", False)

    def test_carry_advances_over_bin(self) -> None:
        g = {**_NO_GATES, "over_bin": True}
        assert p.next_phase("CARRY", 1, 0, 0, **g) == ("LOWER", False)

    def test_lower_advances_at_release_depth(self) -> None:
        g = {**_NO_GATES, "at_release_depth": True}
        assert p.next_phase("LOWER", 1, 0, 0, **g) == ("RELEASE", False)

    def test_release_is_terminal(self) -> None:
        # even at a huge step count with no gates, RELEASE holds.
        assert p.next_phase("RELEASE", 9999, 0, 0, **_NO_GATES) == ("RELEASE", False)


class TestRegrasp:
    """Backward re-grasp is requested on a drop but bounded by MAX_REGRASPS."""

    def test_lift_requests_regrasp_when_not_holding(self) -> None:
        g = {**_NO_GATES, "not_holding": True}
        assert p.next_phase("LIFT", 1, 0, regrasps=0, **g) == ("APPROACH", True)

    def test_carry_requests_regrasp_when_not_holding(self) -> None:
        g = {**_NO_GATES, "not_holding": True}
        assert p.next_phase("CARRY", 1, 0, regrasps=0, **g) == ("APPROACH", True)

    def test_regrasp_bounded_by_max(self) -> None:
        # once MAX_REGRASPS reached, a drop no longer resets; the cap carries it forward.
        g = {**_NO_GATES, "not_holding": True}
        nxt, regrasp = p.next_phase(
            "LIFT", p.PHASE_STEP_CAP["LIFT"], 0, p.MAX_REGRASPS, **g
        )
        assert not regrasp
        assert nxt == "CARRY"  # forced forward, not stuck re-grasping


class TestPhaseAfter:
    def test_order_and_clamp(self) -> None:
        assert p.phase_after("APPROACH") == "DESCEND"
        assert p.phase_after("LOWER") == "RELEASE"
        assert p.phase_after("RELEASE") == "RELEASE"  # terminal clamp
        assert p.phase_after("CLOSE") == "HOLD"
        assert p.phase_after("HOLD") == "LIFT"


class TestDemoParity:
    """2026-07-20 port of scripts/_gen_sim_demos.py rollout()'s proven (~80 %
    success) open-loop schedule (lines 224-235) into this phase machine."""

    def test_phase_order_has_hold(self) -> None:
        assert p.PHASE_ORDER == [
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

    def test_demo_schedule_constants(self) -> None:
        assert p.PHASE_STEP_CAP["APPROACH"] == 50
        assert p.PHASE_STEP_CAP["DESCEND"] == 90
        assert p.PHASE_STEP_CAP["LIFT"] == 60
        assert p.PHASE_STEP_CAP["CARRY"] == 60
        assert p.PHASE_STEP_CAP["LOWER"] == 40
        assert p.STABILIZE_STEPS == 30
        assert p.CLOSE_DWELL == 80
        assert p.HOLD_STEPS == 25

    def test_hold_waits_full_dwell(self) -> None:
        assert p.next_phase("HOLD", 0, p.HOLD_STEPS - 1, 0, **_NO_GATES) == (
            "HOLD",
            False,
        )
        assert p.next_phase("HOLD", 0, p.HOLD_STEPS, 0, **_NO_GATES) == (
            "LIFT",
            False,
        )

    def test_hold_ignores_gates(self) -> None:
        # Even with every other-phase gate wide open, HOLD is purely count-based:
        # no early exit to LIFT, and no regrasp reset, before HOLD_STEPS elapses.
        g = {
            **_NO_GATES,
            "obj_lifted": True,
            "ee_at_carry_height": True,
            "not_holding": True,
        }
        assert p.next_phase("HOLD", 0, p.HOLD_STEPS - 1, 0, **g) == ("HOLD", False)

    def test_close_advances_to_hold(self) -> None:
        assert p.next_phase("CLOSE", 0, p.CLOSE_DWELL, 0, **_NO_GATES) == (
            "HOLD",
            False,
        )

    def test_open_loop_walk_matches_demo_durations(self) -> None:
        """Mirrors compute_scripted_action's real per-tick order (phase_steps
        incremented BEFORE the gate/cap check) so the tick that trips a cap is the
        LAST tick dispatched under the old phase — i.e. counts equal per-phase
        wall-clock durations, not off-by-one. Must equal the demo's exact
        per-phase step counts, summing to 435 pre-RELEASE steps."""
        phase, phase_steps, close_count, regrasps = "APPROACH", 0, 0, 0
        counts: dict[str, int] = {}
        for _ in range(2000):
            phase_steps += 1
            if phase in ("STABILIZE", "CLOSE", "HOLD"):
                close_count += 1
            counts[phase] = counts.get(phase, 0) + 1
            nxt, regrasp = p.next_phase(
                phase, phase_steps, close_count, regrasps, **_NO_GATES
            )
            if regrasp or nxt != phase:
                phase, phase_steps, close_count = nxt, 0, 0
                if phase == "RELEASE":
                    break
        assert counts == {
            "APPROACH": 50,
            "DESCEND": 90,
            "STABILIZE": 30,
            "CLOSE": 80,
            "HOLD": 25,
            "LIFT": 60,
            "CARRY": 60,
            "LOWER": 40,
        }
        assert sum(counts.values()) == 435


class TestTransitionFormat:
    """Gate-parseability contract for ``format_phase_transition`` (2026-07-21 fix).

    ``scripts/_residual_smoke_gate.sh`` (workspace, frozen reference) parses
    ``[script-dbg]`` lines with
    ``\\[script-dbg\\] phase=(\\w+) obj_lifted=(\\w+) oz=([\\-0-9.]+) ez=([\\-0-9.]+)``.
    These tests pin that shape so the transition-only trace (replacing the old
    fixed-150-cadence print, which aliased with 301-step episodes) stays
    parseable.
    """

    def test_gate_regex_contract(self) -> None:
        import re

        line = p.format_phase_transition(
            "LIFT", "HOLD", obj_lifted=True, oz=0.0812, ez=0.1734, t=207
        )
        m = re.findall(
            r"\[script-dbg\] phase=(\w+) obj_lifted=(\w+) oz=([\-0-9.]+) ez=([\-0-9.]+)",
            line,
        )
        assert m == [("LIFT", "True", "0.081", "0.173")]

    def test_regrasp_suffix(self) -> None:
        line = p.format_phase_transition(
            "APPROACH", "LIFT", obj_lifted=False, oz=0.05, ez=0.2, t=310, regrasp=True
        )
        assert line.endswith(" REGRASP")
        import re

        m = re.findall(
            r"\[script-dbg\] phase=(\w+) obj_lifted=(\w+) oz=([\-0-9.]+) ez=([\-0-9.]+)",
            line,
        )
        assert m[0][0] == "APPROACH"

    def test_t_and_prev_present(self) -> None:
        line = p.format_phase_transition(
            "LIFT", "HOLD", obj_lifted=True, oz=0.0812, ez=0.1734, t=207
        )
        assert "t=207" in line
        assert "prev=HOLD" in line


class TestBlendGating:
    """Phase-aware residual blend authority (residual-rl-v2 post-mortem).

    Uniform blending broke the grasp at any meaningful actor share; the actor
    may only share authority on the phases demo-gen's DAgger noise gating
    perturbs (approach/lift/carry/lower). Grasp-critical phases keep full
    script authority via ``blend_fraction``.
    """

    def test_safe_set_matches_demo_noise_flags(self) -> None:
        assert p.BLEND_SAFE_PHASES == {"APPROACH", "CARRY", "LOWER"}

    def test_every_phase_classified(self) -> None:
        assert set(p.PHASE_ORDER) == p.BLEND_SAFE_PHASES | {
            "DESCEND",
            "STABILIZE",
            "CLOSE",
            "HOLD",
            "LIFT",
            "RELEASE",
        }

    def test_critical_phase_full_script(self) -> None:
        assert p.blend_fraction("CLOSE", 0.0) == 1.0
        assert p.blend_fraction("HOLD", 0.37) == 1.0
        assert p.blend_fraction("RELEASE", 0.0) == 1.0
        assert p.blend_fraction("LIFT", 0.0) == 1.0

    def test_safe_phase_passthrough(self) -> None:
        assert p.blend_fraction("CARRY", 0.37) == 0.37
        assert p.blend_fraction("APPROACH", 0.0) == 0.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
