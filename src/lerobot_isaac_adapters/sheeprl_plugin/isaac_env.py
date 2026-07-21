"""Isaac Lab SO-101 pick-place env wrapped as a sheeprl-compatible gym.Env.

Phase C of `plans/2026-05-23-wm-isaac-env-plan.md`. Replaces the
`HDF5ReplayEnv` in the DreamerV3 training stack:

  * `HDF5ReplayEnv.step()` ignores actions + returns reward=0.0 — the
    actor head learns nothing useful. Result: WM-actor cannot drive a
    real robot.
  * `IsaacSO101Env.step()` runs a real Isaac Lab physics tick + emits
    the shaped pick-place reward from the env's own RewardManager
    (`lerobot_isaac_env.rewards`), so the DreamerV3 actor receives
    causal feedback + a task signal.

Wraps the existing `lerobot_isaac_env.make_env(...)` factory — no need
to re-author the SO-101 scene/articulation/observation/reward managers
(they live in the lerobot-isaac-env sibling). This module is the THIN
gym.Env adapter that:

  1. Boots ManagerBasedRLEnv via `make_env("pick_and_place", num_envs=1)`.
  2. Translates batched (num_envs, ...) tensors → single-env (...,) numpy
     arrays sheeprl expects.
  3. Exposes the canonical sheeprl obs key shape: `{"rgb": (3, H, W),
     "state": (6,)}` by default; expands `state` to (13,) when
     `LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1`.

Soft-imports throughout — module remains importable in any env
(sheeprl-only, dashboard-only). Isaac Lab is only loaded inside
`IsaacSO101Env._boot()`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import gymnasium as gym
import numpy as np

from lerobot_isaac_adapters import scripted_grasp_phases as _phases

logger = logging.getLogger(__name__)

# Mandatory warm-up tick count after sim.reset() before camera obs are
# valid. Inherited from isaac-auto-scene's pitfall list.
WARM_UP_FRAMES = 30

# Module-level singleton: the underlying ManagerBasedRLEnv. Isaac Lab's
# SimulationContext is a process-wide singleton, so multiple
# IsaacSO101Env instances (e.g. sheeprl's train + test envs) MUST share
# one backing env. Without this, the second instance's _boot() tries to
# create a second SimulationContext and gets
# `RuntimeError: Simulation context already exists`.
_GLOBAL_BACKING_ISAAC_ENV: Any = None

# Module global pointing at the most-recently-booted IsaacSO101Env WRAPPER instance
# (distinct from `_GLOBAL_BACKING_ISAAC_ENV`, which is the backing ManagerBasedRLEnv).
# The residual-RL patch in `scripts/_wm_isaac_entry.py` reads this to call
# `compute_scripted_action()` from inside the patched PlayerDV3.get_actions seam — the
# only place where the scripted base action can be both recorded to the buffer AND
# executed (see memory `sheeprl-action-override-buffer-seam`). Training boots its
# wrapper first; the eval wrapper (if any) overwrites this, but residual is skipped on
# greedy/eval actions, so the train wrapper is always the one used during training.
_LAST_WRAPPER: Any = None

# Default obs key set the wrapper exposes to sheeprl. The Isaac Lab env's
# `policy` ObservationGroup must include a `joint_pos`-style term (mapped
# to `state`) AND a camera term (mapped to `rgb`). Camera wiring lives in
# lerobot-isaac-env's `wrist_camera_rgb` / `overhead_camera_rgb` — currently
# scaffolded with NotImplementedError; the wrapper detects that and falls
# back to zero RGB until those land. See CLAUDE.md §"Camera observation
# wiring" in the training workspace.
DEFAULT_STATE_KEY = "joint_pos"
# DR100 Phase 1 (2026-05-26) replaced `wrist_camera_rgb`/`overhead_camera_rgb`
# with the single wrist-mounted `d435_rgb` term (3, 480, 640), matching the real
# SO-101 dataset column `observation.images.d435_rgb`. The wrapper resizes it to
# `image_size`² before handing it to the DreamerV3 CNN encoder.
DEFAULT_CAMERA_KEY = "d435_rgb"

# Opt-in object_pose actor obs — diagnostic for the 2026-05-24 sweep where
# Grads/actor → 0 because the actor had no object-location signal.
# When enabled, state_dim expands from 6 to 13 (joint_pos[6] + object_pose[7]).
_INCLUDE_OBJECT_POSE = os.environ.get("LEROBOT_ISAAC_INCLUDE_OBJECT_POSE", "0") not in (
    "0",
    "",
    "false",
    "False",
)
_STATE_DIM_BASE = 6  # joint_pos (6-DOF)
_STATE_DIM_OBJECT_POSE = 7  # pos[3] + quat[4]

# --- Reactive scripted-grasp controller thresholds (residual RL; see
#     compute_scripted_action). All in metres, in the world frame. Tuned to the
#     pick_and_place scene geometry (die rest z≈0.05, grasp_z≈0.106, z_high≈0.19 —
#     same waypoints as scripts/_gen_sim_demos.py). GPU-validation pending: these gate
#     phase selection, so a wrong value mis-sequences the controller.
_DIE_REST_Z = 0.05  # die resting height above table (object spawn z)
_LIFT_MARGIN = 0.04  # die counts as "lifted" above rest+this
_HOLD_TOL = 0.06  # ee↔die 3-D dist below which the die is deemed IN the gripper
_REACH_MAX = 0.30  # reach-envelope clamp on the grasp target (max planar reach ~0.346)
_ALIGN_TOL = 0.015  # ee within this planar dist of the latched target ⇒ aligned
_HIGH_MARGIN = 0.04  # ee above grasp_z+this ⇒ "high" (align here before descending)
_GRASP_DEPTH_MARGIN = 0.015  # ee below grasp_z+this ⇒ at grasp depth (start closing)
_CLOSE_RAMP = 80  # steps over which the grip interpolates OPEN→CLOSE (slow cradle);
# demo-parity (2026-07-20): the ramp now reaches full CLOSE exactly at CLOSE_DWELL's
# end (scripts/_gen_sim_demos.py: 80-step close ramp, then a separate 25-step HOLD
# at full close before lifting — see the new HOLD phase in compute_scripted_action).
# NOTE: the old rate-limited LIFT-target constant was REMOVED in the same port —
# LIFT now commands `z_high` DIRECTLY (see the LIFT branch below): the old
# incremental target produced weak q_des deltas under the residual blend, which
# campaign evidence traced to slipped lifts (die reached oz≈0.008 then dropped).
# NOTE: the phase SCHEDULE (order, per-phase step caps, STABILIZE/CLOSE/HOLD dwell
# counts, re-grasp cap) + the pure transition function live in
# `lerobot_isaac_adapters.scripted_grasp_phases` (imported as `_phases`) so they are
# unit-testable without the Isaac/gymnasium stack.
# --- carry+place phases (extend the residual base from grasp+lift to the FULL pick-place,
#     mirroring scripts/_gen_sim_demos.py: CARRY→LOWER→RELEASE. Without these the scripted
#     base only grasps+lifts and the residual RL must discover carry+place from scratch — the
#     exact wall it never breaks (S3 run 2026-06-27: reward climbed but ep_len_avg stayed 300,
#     0 places). With them the base places ~the scripted rate and RL only refines it.)
_PLACE_Z = 0.06  # ee z over the bin at release (die lands in the cup); matches demo-gen
_CARRY_TOL = 0.03  # ee planar dist to bin centre ⇒ over the bin, start lowering
_RELEASE_RAMP = (
    50  # steps to GRADUALLY open at the bin (avoid ejecting the die on release)
)


class IsaacSO101Env(gym.Env):
    """SO-101 pick-place env wrapped for sheeprl + DreamerV3.

    Observation:
        dict with keys
            "rgb":   uint8 (3, H, W) — wrist camera, falls back to zeros
                                       until lerobot-isaac-env camera term
                                       wiring lands.
            "state": float32 (6,)    — joint positions (default).
                     float32 (13,)   — joint_pos[6] + object_pose[7] when
                                       LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1.

    Action: float32 (6,) — joint position targets in [-1, 1] (env's
            JointPositionActionCfg scales these internally).

    Reward: passthrough from `ManagerBasedRLEnv.step()[1]`, which
            aggregates the terms wired in
            `lerobot_isaac_env.rewards` (`success_reward`,
            `action_l2_penalty`, `joint_vel_penalty`).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: str = "pickplace",
        num_envs: int = 1,
        image_size: int = 64,
        rate_hz: float = 30.0,
        max_episode_steps: int = 600,
        headless: bool = True,
        device: str = "cuda",
        seed: int | None = None,
        dr_config: str | None = None,
        state_key: str = DEFAULT_STATE_KEY,
        camera_key: str = DEFAULT_CAMERA_KEY,
        enable_cameras: bool = True,
    ) -> None:
        super().__init__()
        self.task = task
        self.num_envs = num_envs
        self.image_size = image_size
        self.rate_hz = rate_hz
        self.max_episode_steps = max_episode_steps
        self.headless = headless
        self.device = device
        self.dr_config = dr_config
        self.state_key = state_key
        self.camera_key = camera_key
        self.enable_cameras = enable_cameras

        # Compute state dimension based on env-var flag.
        state_dim = _STATE_DIM_BASE + (
            _STATE_DIM_OBJECT_POSE if _INCLUDE_OBJECT_POSE else 0
        )
        self._state_dim = state_dim

        # Spaces declared up-front so sheeprl's make_env() space-inspection
        # codepath succeeds without booting Isaac Lab.
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, image_size, image_size),
                    dtype=np.uint8,
                ),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.reward_range = (-np.inf, np.inf)

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._isaac_env: Any = None  # populated by _boot()
        self._app: Any = None  # SimulationApp handle
        self._booted = False
        self._has_camera_term = False  # set by _boot() probe

    # ------------------------------------------------------------------ #
    # gym.Env API
    # ------------------------------------------------------------------ #

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if not self._booted:
            self._boot()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        if getattr(self, "_script_ready", False):
            self._script_reset_phase()  # new episode → restart the grasp phase machine
        # ManagerBasedRLEnv.reset returns (obs_dict, info_dict). obs_dict
        # is keyed by ObservationGroup name; we use "policy".
        # inference_mode(False) guard: sheeprl wraps its env-collection loop in
        # torch.inference_mode() (dreamer_v3.py:553), and any in-step DR auto-reset runs
        # inside it. Isaac Lab's DR reset event terms (reset_root_state_uniform /
        # reset_joints_by_scale) write randomized state into the sim physics buffers
        # IN-PLACE, which raises "RuntimeError: Inference tensors cannot be ..." when done
        # under inference_mode. Disabling it here lets the DR buffer writes succeed and is
        # a no-op when grad mode is already normal.
        # NOTE: not yet GPU-verified — confirm against the actual traceback before relying
        # on it. (sheeprl test() uses @torch.no_grad(), which does NOT create inference
        # tensors, so the trigger is the training-collection inference_mode, not eval.)
        import torch

        with torch.inference_mode(False):
            raw_obs, raw_info = self._isaac_env.reset(seed=seed)
        return self._translate_obs(raw_obs), self._scalar_info(raw_info)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self._booted:
            raise RuntimeError("call reset() before step()")
        self._t += 1
        # ManagerBasedRLEnv expects action shape (num_envs, action_dim).
        # We're single-env → add batch dim; cast to torch on device.
        action_t = self._to_torch(action).view(self.num_envs, -1)
        # inference_mode(False) guard: see reset(). Isaac auto-resets a terminated/
        # truncated env INSIDE step() (below), firing the DR reset event terms whose
        # in-place sim-buffer writes raise "Inference tensors cannot be ..." under the
        # inference_mode that wraps sheeprl's collection loop (dreamer_v3.py:553). No-op
        # when grad mode is already on. (Not yet GPU-verified — confirm via traceback.)
        import torch

        with torch.inference_mode(False):
            raw_obs, raw_reward, raw_term, raw_trunc, raw_info = self._isaac_env.step(
                action_t
            )
        obs = self._translate_obs(raw_obs)
        reward = float(self._scalar(raw_reward))
        terminated = bool(self._scalar(raw_term))
        # Isaac Lab tracks its own truncation; combine with the wrapper's
        # max_episode_steps cap so sheeprl's done-handling is correct.
        truncated = bool(self._scalar(raw_trunc)) or (self._t >= self.max_episode_steps)
        # Isaac auto-resets a terminated/truncated env INSIDE step(), without calling the
        # wrapper's reset() — so restart the grasp phase machine here for the next episode.
        if (terminated or truncated) and getattr(self, "_script_ready", False):
            self._script_reset_phase()
        return obs, reward, terminated, truncated, self._scalar_info(raw_info)

    # ------------------------------------------------------------------ #
    # residual RL: scripted-grasp base action (sim-only)
    # ------------------------------------------------------------------ #

    def _init_script_controller(self) -> bool:
        """Lazy-init the DifferentialIK scripted-grasp controller. Idempotent.

        The IK MATH (entity names, joint indices, jacobian slicing, IK cfg, the
        `(q_des-q_default)/0.5` normalization) is transcribed faithfully from
        `scripts/_gen_sim_demos.py`'s physics-verified grasp. NOTE: the SEQUENCING is
        NOT identical — _gen_sim_demos runs a fixed open-loop phase schedule, whereas
        compute_scripted_action is a REACTIVE state-machine (phase inferred from live
        state) so it can produce a base action at any RL step. Returns True if ready,
        False if the Isaac scene is unavailable (e.g. hardware) → caller falls back to
        pure policy. Retries a few times before latching OFF, so a transient
        scene-not-ready does not permanently disable the residual.
        """
        if getattr(self, "_script_ready", False):
            return True
        if self._isaac_env is None or not hasattr(self._isaac_env, "scene"):
            return False
        try:
            import torch  # noqa: F401
            from isaaclab.controllers import (  # type: ignore[import]
                DifferentialIKController,
                DifferentialIKControllerCfg,
            )

            scene = self._isaac_env.scene
            robot = scene["robot"]
            self._script_robot = robot
            self._script_obj = scene["source_object"]
            self._script_dev = self._isaac_env.device
            self._script_ee_idx = int(robot.find_bodies("gripper_link")[0][0])
            self._script_arm_ids = list(
                robot.find_joints(
                    [
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                    ]
                )[0]
            )
            self._script_grip_idx = int(robot.find_joints("gripper")[0][0])
            _fixed = bool(getattr(robot, "is_fixed_base", True))
            self._script_ee_jac = (
                (self._script_ee_idx - 1) if _fixed else self._script_ee_idx
            )
            self._script_jac_off = 0 if _fixed else 6
            self._script_qdef = robot.data.default_joint_pos.clone()
            self._script_adim = int(self._isaac_env.action_space.shape[-1])
            self._script_ik = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                num_envs=1,
                device=self._script_dev,
            )
            # Waypoint constants — same as _gen_sim_demos.py defaults.
            self._script_grasp_z = float(
                os.environ.get("LEROBOT_ISAAC_GRASP_Z", "0.106")
            )
            # Demo-parity (2026-07-20): 0.19 (LEROBOT_ISAAC_CARRY_Z default) clears the
            # 7 cm cup rim — the die hangs ~0.096 below gripper_link, so z_high must
            # clear grasp_z + the rim + that hang distance. The old 0.17 default maxed
            # the die out at oz≈0.072 and undershot the demo's actual carry height.
            self._script_z_high = float(
                os.environ.get(
                    "LEROBOT_ISAAC_SCRIPT_Z_HIGH",
                    os.environ.get("LEROBOT_ISAAC_CARRY_Z", "0.19"),
                )
            )
            self._script_tgt_x = float(os.environ.get("LEROBOT_ISAAC_TARGET_X", "0.22"))
            self._script_tgt_y = float(
                os.environ.get("LEROBOT_ISAAC_TARGET_Y", "-0.13")
            )
            self._script_quat = [1.0, 0.0, 0.0, 0.0]  # straight-down grasp
            # RELEASE ramps to a PARTIAL open, not full GRIP_OPEN: a full open spreads
            # the fingers into the cup wall -> servo jam (finger-jam demo-gen, 2026-06-24).
            self._script_part_open = float(
                os.environ.get("LEROBOT_ISAAC_PLACE_PART_OPEN", "1.0")
            )
            # Hybrid phase-machine state (per episode): demo-ordered, state-gated.
            self._script_reset_phase()
            self._script_ready = True
            logger.info(
                "scripted-grasp controller initialised (residual RL base action)"
            )
            # print (not just logger): the module logger is swallowed in the
            # sheeprl/Isaac run, so a smoke couldn't tell whether the residual base
            # initialised. This one-time line makes it visible.
            print(
                "[residual-rl] scripted-grasp controller INITIALISED "
                f"(ee_idx={self._script_ee_idx}, arm_ids={self._script_arm_ids}, "
                f"adim={self._script_adim}, fixed_base={_fixed}).",
                flush=True,
            )
            return True
        except Exception as exc:  # noqa: BLE001 — never let init break the run
            self._script_ready = False
            self._script_init_attempts = getattr(self, "_script_init_attempts", 0) + 1
            if self._script_init_attempts >= 3:
                # Latch OFF only after repeated failure (not a transient scene-not-ready).
                logger.error(
                    "scripted-grasp controller init failed %d× — residual DISABLED for "
                    "the rest of this run: %s",
                    self._script_init_attempts,
                    exc,
                )
                self._script_init_failed = True
            else:
                logger.warning(
                    "scripted-grasp controller init failed (attempt %d/3, will retry): %s",
                    self._script_init_attempts,
                    exc,
                )
            # print (not just logger) so the smoke sees the REAL failure reason
            # regardless of logging config; include the traceback on the first hit.
            import traceback

            print(
                f"[residual-rl] scripted-grasp controller INIT FAILED "
                f"(attempt {self._script_init_attempts}/3): {type(exc).__name__}: {exc}",
                flush=True,
            )
            if self._script_init_attempts == 1:
                traceback.print_exc()
            return False

    def _script_reset_phase(self) -> None:
        """Full reset of the scripted-grasp phase machine for a NEW episode."""
        self._script_phase = "APPROACH"
        self._script_gx = None  # target xy, latched at APPROACH (reach-clamped)
        self._script_gy = None
        self._script_close_count = 0
        self._script_phase_steps = (
            0  # steps spent in the current phase (drives the cap)
        )
        self._script_regrasps = (
            0  # bounded backward re-grasp count (see _phases.MAX_REGRASPS)
        )

    def _advance_phase(self, nxt: str) -> None:
        """Enter phase ``nxt``: reset the per-phase step + close counters.

        ``close_count`` is the internal counter for STABILIZE / CLOSE / HOLD / RELEASE,
        so it must start fresh on each phase entry; the other phases ignore it.
        """
        self._script_phase = nxt
        self._script_phase_steps = 0
        self._script_close_count = 0

    def _script_regrasp(self) -> None:
        """Bounded mid-episode fall-back to APPROACH (die dropped) — re-latch the
        target but KEEP the regrasp counter so it cannot loop forever."""
        self._script_regrasps += 1
        self._script_phase = "APPROACH"
        self._script_phase_steps = 0
        self._script_close_count = 0
        self._script_gx = None  # re-latch (the die may have moved)
        self._script_gy = None

    def compute_scripted_action(self) -> np.ndarray | None:
        """Return a (action_dim,) scripted-grasp action for the CURRENT pre-step state.

        HYBRID phase machine — demo-ORDERED (APPROACH→DESCEND→CLOSE→HOLD→LIFT, the
        proven sequence from _gen_sim_demos) but STATE-GATED transitions (robust to the
        residual clamp rate-limiting motion). Critically: it ALIGNS the ee over the
        object while HIGH before descending vertically, and only closes after a dwell —
        so it does NOT knock the (16 mm) die sideways the way a naive "close when
        xy<3cm" reactive controller does (diagnosed by the GPU probe: that pushed the
        die out of reach).
        Same normalized action space as the policy (`(q_des-q_default)/0.5` for arm, grip
        in [-1,1]) → directly blendable. Phase state resets per episode (step()/reset()).

        Returns None when the Isaac scene is unavailable (hardware) or on any error
        → caller uses the pure policy action (residual weight effectively 0).
        """
        if getattr(self, "_script_init_failed", False):
            return None
        if not self._init_script_controller():
            return None
        try:
            import torch
            from isaaclab.utils.math import subtract_frame_transforms  # type: ignore[import]

            robot = self._script_robot
            obj = self._script_obj
            dev = self._script_dev
            ee_idx = self._script_ee_idx
            arm_ids = self._script_arm_ids
            grip_idx = self._script_grip_idx
            qdef = self._script_qdef
            GRIP_OPEN, GRIP_CLOSE = 1.0, -1.0
            grasp_z, z_high = self._script_grasp_z, self._script_z_high

            # ---- live state (world frame; ee pose is a function of joint_pos, which is
            #      in the obs, and obj pose is in the obs when INCLUDE_OBJECT_POSE=1 —
            #      so the scripted action is reproducible by the actor) ----
            obj_pos = obj.data.root_pos_w[0]  # (3,) world
            ee_pos_w = robot.data.body_pos_w[0, ee_idx, :]  # (3,) world
            ox, oy, oz = float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])
            ex, ey, ez = float(ee_pos_w[0]), float(ee_pos_w[1]), float(ee_pos_w[2])

            # Latch the grasp target xy at episode start, REACH-CLAMPED so the arm never
            # chases a die that has been pushed out of the envelope (probe failure mode).
            if self._script_gx is None:
                r = (ox * ox + oy * oy) ** 0.5
                if r > _REACH_MAX and r > 1e-6:
                    s = _REACH_MAX / r
                    self._script_gx, self._script_gy = ox * s, oy * s
                else:
                    self._script_gx, self._script_gy = ox, oy
            gx, gy = self._script_gx, self._script_gy

            xy_to_tgt = ((ex - gx) ** 2 + (ey - gy) ** 2) ** 0.5
            ee_to_obj_3d = ((ex - ox) ** 2 + (ey - oy) ** 2 + (ez - oz) ** 2) ** 0.5
            obj_lifted = oz > (_DIE_REST_Z + _LIFT_MARGIN)
            aligned = xy_to_tgt < _ALIGN_TOL  # tight: < die half-width, so close
            ee_high = ez > (grasp_z + _HIGH_MARGIN)  #        doesn't knock the die
            at_depth = ez < (grasp_z + _GRASP_DEPTH_MARGIN)

            # ---- phase machine: OPEN-LOOP schedule + state-gate early-exit ----
            # Set target+grip for the CURRENT phase; the transition to the NEXT phase
            # is a pure function (_phases.next_phase) with HARD per-phase step caps,
            # so the machine cannot stall on a gate that the blended/clamped action
            # never satisfies (the diagnosed APPROACH stall). The gates remain as fast
            # early-exits when motion is clean.
            self._script_phase_steps += 1
            ph = self._script_phase
            tx, ty = self._script_tgt_x, self._script_tgt_y
            if ph == "APPROACH":  # align over the die while HIGH (open)
                target, grip = [gx, gy, z_high], GRIP_OPEN
            elif ph == "DESCEND":  # straight down to grasp depth (open)
                target, grip = [gx, gy, grasp_z], GRIP_OPEN
            elif ph == "STABILIZE":  # settle open at depth (die sits between fingers)
                target, grip = [gx, gy, grasp_z], GRIP_OPEN
                self._script_close_count += 1
            elif ph == "CLOSE":  # gradual cradle-close (an instant close ejects it)
                self._script_close_count += 1
                frac = min(1.0, self._script_close_count / _CLOSE_RAMP)
                grip = GRIP_OPEN + (GRIP_CLOSE - GRIP_OPEN) * frac  # 1.0 → -1.0
                target = [gx, gy, grasp_z]
            elif ph == "HOLD":  # firm closed grip at depth before lifting (demo-parity)
                self._script_close_count += 1
                target, grip = [gx, gy, grasp_z], GRIP_CLOSE
            elif ph == "LIFT":  # raise DIRECTLY to z_high (demo-parity: no rate limit —
                # the old incremental ez-plus-rate target under-drove the residual blend)
                target, grip = [gx, gy, z_high], GRIP_CLOSE
            elif ph == "CARRY":  # move held+high to over the bin
                target, grip = [tx, ty, z_high], GRIP_CLOSE
            elif ph == "LOWER":  # descend over the bin to release depth (closed)
                target, grip = [tx, ty, _PLACE_Z], GRIP_CLOSE
            else:  # RELEASE — gradual open to drop the die in (PARTIAL open: a full
                # open spreads the fingers into the cup wall -> servo jam)
                self._script_close_count += 1
                frac = min(1.0, self._script_close_count / _RELEASE_RAMP)
                part_open = self._script_part_open
                grip = GRIP_CLOSE + (part_open - GRIP_CLOSE) * frac  # -1.0 → part_open
                target = [tx, ty, _PLACE_Z]

            # transition (pure; hard caps guarantee forward progress → never stalls)
            xy_to_bin = ((ex - tx) ** 2 + (ey - ty) ** 2) ** 0.5
            nxt, regrasp = _phases.next_phase(
                ph,
                self._script_phase_steps,
                self._script_close_count,
                self._script_regrasps,
                aligned=aligned,
                ee_high=ee_high,
                at_depth=at_depth,
                obj_lifted=obj_lifted,
                ee_at_carry_height=(ez > z_high - 0.01),
                not_holding=(
                    not obj_lifted and ee_to_obj_3d > _HOLD_TOL and ez > grasp_z + 0.03
                ),
                over_bin=(xy_to_bin < _CARRY_TOL),
                at_release_depth=(ez < _PLACE_Z + _GRASP_DEPTH_MARGIN),
            )
            if regrasp or nxt != ph:
                # de-aliased trace: one line per phase TRANSITION with the
                # episode-relative step (contract: _residual_smoke_gate.sh regex).
                print(
                    _phases.format_phase_transition(
                        nxt, ph, obj_lifted=obj_lifted, oz=oz, ez=ez,
                        t=self._t, regrasp=regrasp,
                    ),
                    flush=True,
                )
            if regrasp:
                self._script_regrasp()
            elif nxt != ph:
                self._advance_phase(nxt)

            # ---- IK (transcribed from _gen_sim_demos.step_to) ----
            self._script_ik.reset()
            cmd = torch.tensor(
                [target + self._script_quat], device=dev, dtype=torch.float32
            )
            rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
            pos_b, quat_b = subtract_frame_transforms(
                rp,
                rq,
                robot.data.body_pos_w[:, ee_idx, :],
                robot.data.body_quat_w[:, ee_idx, :],
            )
            self._script_ik.set_command(cmd, ee_pos=pos_b, ee_quat=quat_b)
            jac = robot.root_physx_view.get_jacobians()[
                :, self._script_ee_jac, :6, [self._script_jac_off + j for j in arm_ids]
            ]
            q_des = self._script_ik.compute(
                pos_b, quat_b, jac, robot.data.joint_pos[:, arm_ids]
            )
            # Per-joint action scale (C1 ee-descent fix, 2026-07-15): divide by the SAME
            # per-joint scale the env applies (lerobot_isaac_env.load_action_scale_dict),
            # so q_cmd == q_des and the residual clamp becomes a no-op. Cached once at
            # first call. Defaults to 0.5 for every joint when LEROBOT_ISAAC_ACTION_SCALE_JSON
            # is unset → byte-identical to the historical `/ 0.5` behaviour.
            if not hasattr(self, "_script_arm_scales"):
                try:
                    from lerobot_isaac_env.so101_env_cfg import load_action_scale_dict

                    _sd = load_action_scale_dict()
                    _jn = robot.data.joint_names
                    self._script_arm_scales = [float(_sd.get(_jn[jid], 0.5)) for jid in arm_ids]
                except Exception:  # noqa: BLE001
                    self._script_arm_scales = [0.5 for _ in arm_ids]
            action = torch.zeros((1, self._script_adim), device=dev)
            for k, jid in enumerate(arm_ids):
                action[0, jid] = (q_des[0, k] - qdef[0, jid]) / self._script_arm_scales[k]
            action[0, grip_idx] = grip
            return action[0].detach().cpu().numpy().astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            n = getattr(self, "_script_warn_count", 0) + 1
            self._script_warn_count = n
            if n <= 3:
                logger.warning(
                    "compute_scripted_action failed (residual skipped this step; "
                    "warning %d, further suppressed): %s",
                    n,
                    exc,
                )
            return None

    def render(self) -> np.ndarray:
        # Return HWC for sheeprl's RecordVideoV0 wrapper.
        return (
            self._last_rgb_hwc.copy()
            if hasattr(self, "_last_rgb_hwc")
            else (np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        )

    def close(self) -> None:
        # NO-OP on the shared backing env — deliberately do NOT close it.
        #
        # sheeprl's dreamer_v3.main() calls `envs.close()` immediately BEFORE
        # the eval phase (`dreamer_v3.py:765` then `test(player, ...)` at :767).
        # test() builds a FRESH env via `make_env(...)()` and resets it — that
        # fresh IsaacSO101Env reuses `_GLOBAL_BACKING_ISAAC_ENV`. If close()
        # actually called `self._isaac_env.close()`, ManagerBasedRLEnv deletes
        # its `.scene`, so the test reset crashes with
        # `'ManagerBasedRLEnv' object has no attribute 'scene'` — which then
        # hangs forever in Isaac's atexit SimulationApp.close()→render() and
        # masquerades as the WM-Isaac "training stall" (metric=-9999).
        #
        # The backing env is a process-wide singleton (Isaac's SimulationContext
        # is process-global); it must outlive any single wrapper. Real teardown
        # happens at process exit, which `_wm_isaac_entry.py` forces via
        # os._exit() to bypass the hanging atexit close.
        logger.info(
            "IsaacSO101Env.close(): no-op — shared backing env kept alive for "
            "the eval/test phase (see sheeprl dreamer_v3.py:765-767)."
        )

    # ------------------------------------------------------------------ #
    # boot
    # ------------------------------------------------------------------ #

    def _boot(self) -> None:
        """Spin up Isaac Lab + the SO-101 pick-place env. Idempotent.

        AppLauncher MUST run BEFORE any `isaaclab.*` import — the
        managers import `omni.kit.app` at module-load time, which only
        exists once SimulationApp is alive. Failing to do this gives
        `ModuleNotFoundError: omni.kit.app`. Same recipe as Isaac Lab's
        own example scripts.
        """
        if self._booted:
            return

        # 1. Boot SimulationApp via AppLauncher FIRST — unless the caller
        #    already booted it (e.g. scripts/_wm_isaac_entry.py does this
        #    to claim libgobject before sheeprl imports). AppLauncher is
        #    NOT a singleton — calling it twice hangs waiting for kit
        #    extension reload. Probe `omni.kit.app` for an existing app.
        existing_app = None
        try:
            import omni.kit.app as _kit_app  # type: ignore[import]

            existing_app = _kit_app.get_app()
        except Exception:  # noqa: BLE001
            existing_app = None
        if (
            existing_app is not None
            and getattr(existing_app, "is_running", lambda: False)()
        ):
            logger.info("SimulationApp already alive — skipping AppLauncher")
            self._app = existing_app
        else:
            try:
                from isaaclab.app import AppLauncher  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "Isaac Lab (isaaclab.app.AppLauncher) is required. "
                    "Run `pixi install -e sim && pixi run install-isaac-lab` "
                    f"in the training workspace. ({exc})"
                ) from exc
            launcher = AppLauncher(headless=self.headless, enable_cameras=True)
            self._app = launcher.app
            for _ in range(2):
                self._app.update()

        # 2. NOW it's safe to import lerobot_isaac_env (which transitively
        #    imports isaaclab.envs / managers).
        try:
            from lerobot_isaac_env import make_env  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "lerobot_isaac_env required for IsaacSO101Env. "
                "Install via the training workspace's editable-siblings "
                f"feature (pixi install -e sim). ({exc})"
            ) from exc

        # Translate this wrapper's `task` to lerobot_isaac_env's task name.
        # Sibling accepts: 'pick' | 'pick_and_place' | full gym IDs.
        task_alias = {
            "pickplace": "pick_and_place",
            "pick_and_place": "pick_and_place",
            "pick": "pick",
        }.get(self.task, self.task)

        global _GLOBAL_BACKING_ISAAC_ENV
        if _GLOBAL_BACKING_ISAAC_ENV is None:
            logger.info(
                "booting Isaac Lab env task=%s num_envs=%d headless=%s cameras=%s",
                task_alias,
                self.num_envs,
                self.headless,
                self.enable_cameras,
            )
            _GLOBAL_BACKING_ISAAC_ENV = make_env(
                task=task_alias,
                num_envs=self.num_envs,
                headless=self.headless,
                enable_cameras=self.enable_cameras,
            )
        else:
            logger.info(
                "reusing existing Isaac Lab backing env (task=%s) — "
                "SimulationContext singleton enforced",
                task_alias,
            )
        self._isaac_env = _GLOBAL_BACKING_ISAAC_ENV
        # Backing-cap fix (2026-07-21): the backing ManagerBasedRLEnv has its own
        # episode_length_s time_out (default 10 s * 30 Hz = 300 steps) which fires
        # BEFORE this wrapper's max_episode_steps cap (step() line ~270 only ADDS a
        # truncation). The scripted pick->place needs ~485 steps, so a 300-step
        # backing cap truncates mid-grasp — demo-gen disables it the same way
        # (_gen_sim_demos.py). max_episode_length is a read-only derived property;
        # bumping cfg.episode_length_s recomputes it. Wrapper truncation remains the
        # single episode-length authority.
        try:
            backing_max = int(getattr(self._isaac_env, "max_episode_length", 0) or 0)
            if backing_max and self.max_episode_steps > backing_max:
                self._isaac_env.cfg.episode_length_s = 1.0e6
                logger.info(
                    "backing episode cap raised: max_episode_length %d -> %s "
                    "(wrapper max_episode_steps=%d governs)",
                    backing_max,
                    getattr(self._isaac_env, "max_episode_length", "?"),
                    self.max_episode_steps,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not raise backing episode cap: %s", exc)
        # Eager flag: warm-up below is best-effort; if it throws we must
        # NOT re-enter _boot() and re-create the SimulationContext.
        self._booted = True
        # Publish this wrapper for the residual-RL patch (reads the backing scene via
        # compute_scripted_action). ONLY when residual is enabled — otherwise this is a
        # true no-op for the 99% of runs that don't use it (no module-scope reference
        # held). Last-booted wins; the train wrapper boots first, the eval wrapper (if
        # any) boots later but the patch skips eval via the _in_eval guard.
        if (
            float(os.environ.get("LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT", "0.0") or "0.0")
            > 0.0
        ):
            global _LAST_WRAPPER
            _LAST_WRAPPER = self

        # 30-frame warm-up so camera buffers are populated. Use the env's
        # sim handle; fall back to no-op if not exposed.
        sim = getattr(self._isaac_env, "sim", None)
        if sim is not None:
            for _ in range(WARM_UP_FRAMES):
                try:
                    sim.step(render=True)
                except Exception:  # noqa: BLE001
                    break

        # Probe whether the camera obs term is wired. If lerobot-isaac-env
        # still has NotImplementedError stubs for cameras, we'll find out
        # at the first translate and fall back to zeros without crashing.

    # ------------------------------------------------------------------ #
    # obs / action translation
    # ------------------------------------------------------------------ #

    def _translate_obs(self, raw_obs: Any) -> dict[str, np.ndarray]:
        """Convert ManagerBasedRLEnv obs (dict[group]→dict[term]→tensor)
        into the flat {rgb, state} dict sheeprl expects.

        Defensive: if camera term raises (the lerobot-isaac-env scaffold
        still has NotImplementedError for `wrist_camera_rgb`), return a
        zero RGB. Logs once.

        When LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1, concatenates the
        object_pose term (7 dims) to the 6-dim joint_pos vector, yielding
        a 13-dim state vector that gives the actor direct access to object
        location — the key diagnostic for the 2026-05-24 sweep collapse.
        """
        # raw_obs shapes seen in the wild:
        #   * dict[group(str)] -> dict[term(str)] -> Tensor    (older API)
        #   * dict[group(str)] -> Tensor (concat of all terms) (newer API,
        #     ObservationGroup with concatenate_terms=True default)
        if isinstance(raw_obs, dict):
            group = raw_obs.get("policy", raw_obs)
        else:
            group = raw_obs

        # ---- state (joint positions + optional object_pose) ----
        if isinstance(group, dict):
            # Term-wise obs (older API): concat joint_pos + object_pose if enabled.
            jp = group.get(self.state_key)
            parts = [
                self._tensor_to_np(
                    jp, default_shape=(_STATE_DIM_BASE,), default_dtype=np.float32
                ).reshape(-1)[:_STATE_DIM_BASE]
            ]
            if _INCLUDE_OBJECT_POSE:
                op = group.get("object_pose")
                op_np = self._tensor_to_np(
                    op,
                    default_shape=(_STATE_DIM_OBJECT_POSE,),
                    default_dtype=np.float32,
                ).reshape(-1)[:_STATE_DIM_OBJECT_POSE]
                parts.append(op_np)
            state_np = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        elif hasattr(group, "shape"):
            # Flat concat tensor (newer API): field order from PolicyObsGroupCfg:
            #   joint_pos[0:6] + joint_vel[6:12] + last_action[12:18]
            #   + object_pose[18:25] (when INCLUDE_OBJECT_POSE=1).
            flat = self._tensor_to_np(
                group, default_shape=(_STATE_DIM_BASE,), default_dtype=np.float32
            ).reshape(-1)
            parts = [flat[:_STATE_DIM_BASE]]
            if _INCLUDE_OBJECT_POSE:
                # object_pose lives at dims 18..25 per PolicyObsGroupCfg field order.
                if flat.size >= 25:
                    parts.append(flat[18:25])
                else:
                    parts.append(np.zeros(_STATE_DIM_OBJECT_POSE, dtype=np.float32))
            state_np = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        else:
            state_np = np.zeros(self._state_dim, dtype=np.float32)

        # Post-process: squeeze batch dim when single-env; coerce to declared dim.
        if state_np.ndim == 2 and state_np.shape[0] == self.num_envs:
            state_np = state_np[0]
        if state_np.size >= self._state_dim:
            state_np = state_np.reshape(-1)[: self._state_dim]
        else:
            state_np = np.zeros(self._state_dim, dtype=np.float32)

        # ---- rgb (camera) ----
        # With enable_cameras=True the policy group is a dict carrying the
        # `d435_rgb` term at the camera's NATIVE resolution (3, 480, 640) — far
        # larger than the DreamerV3 CNN's 64² input — so we RESIZE it down here
        # (the encoder cnn_keys point at this `rgb` key). If cameras are off or
        # the term is a stub, fall back to a zero frame of the declared shape.
        rgb_val = group.get(self.camera_key) if isinstance(group, dict) else None
        try:
            rgb_np = self._tensor_to_np(
                rgb_val,
                default_shape=(self.image_size, self.image_size, 3),
                default_dtype=np.uint8,
            )
        except NotImplementedError:
            # lerobot-isaac-env camera term is a stub; fall back to zeros
            # and remember so we don't retry every step.
            rgb_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            self._has_camera_term = False
        else:
            self._has_camera_term = rgb_val is not None

        # Normalise shape — Isaac Lab cameras emit (num_envs, H, W, 3) uint8.
        if rgb_np.ndim == 4 and rgb_np.shape[0] == self.num_envs:
            rgb_np = rgb_np[0]
        if rgb_np.ndim == 3 and rgb_np.shape[-1] == 3:
            rgb_np = rgb_np.transpose(2, 0, 1)  # HWC → (3, H, W)
        # Now rgb_np should be (3, H, W). Resize to (3, image_size, image_size)
        # if it carries a real frame; only zero-fill as a last resort.
        if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
            if rgb_np.shape[1:] != (self.image_size, self.image_size):
                rgb_np = self._resize_chw(rgb_np, self.image_size)
            self._last_rgb_hwc = rgb_np.transpose(1, 2, 0)  # for render()
        else:
            rgb_np = np.zeros((3, self.image_size, self.image_size), dtype=np.uint8)
            self._last_rgb_hwc = np.zeros(
                (self.image_size, self.image_size, 3), dtype=np.uint8
            )

        return {
            "rgb": rgb_np.astype(np.uint8, copy=False),
            "state": state_np.astype(np.float32, copy=False),
        }

    def _scalar_info(self, raw_info: Any) -> dict[str, Any]:
        """Flatten Isaac Lab's batched info dict to a single-env dict."""
        if not isinstance(raw_info, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in raw_info.items():
            if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1:
                try:
                    out[k] = float(self._scalar(v))
                except Exception:  # noqa: BLE001
                    out[k] = v
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------ #
    # tiny helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tensor_to_np(
        val: Any,
        *,
        default_shape: tuple[int, ...],
        default_dtype: type,
    ) -> np.ndarray:
        if val is None:
            return np.zeros(default_shape, dtype=default_dtype)
        if hasattr(val, "detach"):
            return val.detach().cpu().numpy()
        if hasattr(val, "cpu"):
            return val.cpu().numpy()
        return np.asarray(val)

    def _to_torch(self, arr: np.ndarray) -> Any:
        """Bring an action array onto the env's torch device."""
        import torch  # local import — keep module light

        if hasattr(arr, "to"):
            return arr.to(self.device)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    @staticmethod
    def _resize_chw(chw_np: np.ndarray, size: int) -> np.ndarray:
        """Resize a (3, H, W) uint8 array to (3, size, size) uint8.

        Uses torch bilinear interpolation (cv2-free — cv2 is not a dependency of
        this env, matching the bridge's PIL/torch-only stance). Falls back to a
        crude stride subsample if torch is somehow unavailable.
        """
        try:
            import torch
            import torch.nn.functional as F

            t = torch.from_numpy(np.ascontiguousarray(chw_np)).unsqueeze(0).float()
            t = F.interpolate(
                t, size=(size, size), mode="bilinear", align_corners=False
            )
            return t.squeeze(0).clamp_(0, 255).to(torch.uint8).numpy()
        except Exception:  # noqa: BLE001 — never let a resize break the rollout
            h, w = chw_np.shape[1], chw_np.shape[2]
            ys = (np.linspace(0, h - 1, size)).astype(np.int64)
            xs = (np.linspace(0, w - 1, size)).astype(np.int64)
            return chw_np[:, ys][:, :, xs].astype(np.uint8, copy=False)

    @staticmethod
    def _scalar(t: Any) -> Any:
        """Squeeze a (1,)-shape tensor or array to a python scalar."""
        if t is None:
            return 0.0
        if hasattr(t, "detach"):
            return t.detach().cpu().reshape(-1)[0].item()
        if hasattr(t, "item"):
            try:
                return t.item()
            except Exception:  # noqa: BLE001
                pass
        arr = np.asarray(t).reshape(-1)
        return arr[0] if arr.size else 0.0


# --------------------------------------------------------------------------- #
# Hydra factory — sheeprl loads this via `env._target_`
# --------------------------------------------------------------------------- #


def get_isaac_env(
    task: str = "pickplace",
    image_size: int = 64,
    num_envs: int = 1,
    rate_hz: float = 30.0,
    max_episode_steps: int = 600,
    headless: bool = True,
    device: str = "cuda",
    seed: int | None = None,
    dr_config: str | None = None,
    enable_cameras: bool = True,
) -> IsaacSO101Env:
    """Hydra-friendly factory wrapping :class:`IsaacSO101Env`.

    Drop-in replacement for ``hdf5_env.get_hdf5_env``. Activate via
    ``env=isaac_so101`` (resolved against
    ``configs/env/isaac_so101.yaml``).
    """
    return IsaacSO101Env(
        task=task,
        num_envs=num_envs,
        image_size=image_size,
        rate_hz=rate_hz,
        max_episode_steps=max_episode_steps,
        headless=headless,
        device=device,
        seed=seed,
        dr_config=dr_config,
        enable_cameras=enable_cameras,
    )
