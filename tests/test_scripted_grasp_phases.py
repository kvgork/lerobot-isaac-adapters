"""Unit tests for the residual-RL scripted-grasp phase machine.

These exercise the pure transition logic (no Isaac / gymnasium / torch), which is
why it was extracted to ``lerobot_isaac_adapters.scripted_grasp_phases``. The
headline invariant is the launch fix: **the machine can never stall** — every phase
with a step cap force-advances at the cap even when its state gate never clears
(the bug that pinned residual-RL in APPROACH under the blended, clamped action).
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
            if phase in ("STABILIZE", "CLOSE"):
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
            "LIFT",
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
