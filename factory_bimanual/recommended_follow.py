"""MuJoCo adapter for the PDF-aligned PiperX recommended-v3.1 protocol."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import mujoco
import numpy as np

from .mujoco_candidate_generator import (
    CandidateGeneratorConfig,
    MuJoCoCandidateGenerator,
)
from .mujoco_collision_adapter import MuJoCoPairedCollisionChecker
from .piperx_recommended import PiperXRecommendedConfig
from .rescue_v31 import (
    CandidateFrame,
    FollowSchedule,
    RescueEvent,
    RescueScheduleConfig,
    StrictGate,
    schedule_rescue_v31,
    wrist_branch_signature,
)
from .robot_contracts import ROBOT_CONTRACTS
from scripts.strict_mujoco_ik import joint_periodic_mask


@dataclass(frozen=True)
class RecommendedFollowResult:
    execution_qpos: np.ndarray
    source_qpos: np.ndarray
    source_index: np.ndarray
    execution_time_s: np.ndarray
    source_state: np.ndarray
    source_accepted: np.ndarray
    actual_tcp: Mapping[str, np.ndarray]
    position_error_m: Mapping[str, np.ndarray]
    orientation_error_rad: Mapping[str, np.ndarray]
    collision: np.ndarray
    events: tuple[RescueEvent, ...]
    candidate_pair_count: np.ndarray
    warm_start_attempted: np.ndarray
    anchor_restart_count: int
    dropped_source_frames: int
    cycle_delay_s: float
    schedule: FollowSchedule


class RecommendedFollowRunner:
    """Generate strict IK branches, then schedule controlled bimanual rescue."""

    def __init__(
        self,
        model: mujoco.MjModel,
        task,
        mapped_quaternions: Mapping[str, np.ndarray],
        config: PiperXRecommendedConfig,
        *,
        collision_clearance_margin_m: float | None = None,
        maximum_candidates_per_side: int = 6,
    ):
        self.model = model
        self.task = task
        self.mapped_quaternions = {
            side: np.asarray(mapped_quaternions[side], dtype=float)
            for side in ("left", "right")
        }
        self.config = config
        self.contract = ROBOT_CONTRACTS["piperx"]
        self.maximum_candidates_per_side = int(maximum_candidates_per_side)
        if self.maximum_candidates_per_side < 1:
            raise ValueError("maximum candidates per side must be positive")
        count = len(np.asarray(task.time_s))
        if count < 1:
            raise ValueError("recommended follow requires at least one source frame")
        for side in ("left", "right"):
            position = np.asarray(getattr(task, f"{side}_position_m"), dtype=float)
            quaternion = self.mapped_quaternions[side]
            if position.shape != (count, 3) or quaternion.shape != (count, 4):
                raise ValueError(f"{side} targets must align with source time")
        self.names = {
            side: {
                "joints": self.contract.prefixed_joint_names(side),
                "site": f"{side}_tcp",
            }
            for side in ("left", "right")
        }
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        candidate_config = CandidateGeneratorConfig(
            position_tolerance_m=config.accept.position_tolerance_m,
            orientation_tolerance_rad=config.accept.orientation_tolerance_rad,
            damping=config.dls.damping,
            step_scale=config.dls.step_scale,
            maximum_step_rad=config.dls.maximum_step_rad,
            position_error_clip_m=config.dls.position_error_clip_m,
            orientation_error_clip_rad=config.dls.orientation_error_clip_rad,
            max_iterations=config.dls.max_iterations,
            global_seed_count=config.anchor_restarts,
            maximum_candidates=self.maximum_candidates_per_side,
            constrained_fallback_enabled=True,
            stratified_seed_enabled=True,
            wrist_risk_enabled=True,
            dedup_rad=np.deg2rad(0.25),
            rolling_early_stop_candidates=2,
        )
        self.generator = MuJoCoCandidateGenerator(
            model, data, self.contract,
            name_map=self.names, config=candidate_config,
        )
        self.checker = MuJoCoPairedCollisionChecker(
            model, data, self.names, transition_steps=7,
            clearance_margin_m=collision_clearance_margin_m,
        )
        self.qids = {}
        self.periodic = {}
        for side in ("left", "right"):
            joint_ids = np.asarray([
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.names[side]["joints"]
            ], dtype=int)
            if np.any(joint_ids < 0):
                raise ValueError(f"PiperX scene is missing {side} joints")
            self.qids[side] = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
            self.periodic[side] = joint_periodic_mask(model, joint_ids)

    def _target(self, side: str, row: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(getattr(self.task, f"{side}_position_m")[row], float),
            self.mapped_quaternions[side][row],
        )

    @staticmethod
    def _deduplicate(candidates) -> list:
        result = []
        for candidate in candidates:
            if not any(np.linalg.norm(candidate.q - old.q) <= 1e-6
                       for old in result):
                result.append(candidate)
        return result

    def _pair_layer(self, left, right) -> tuple[CandidateFrame, ...]:
        pairs = []
        for left_candidate, right_candidate in product(left, right):
            if not self.checker.state(
                    left_candidate.q, right_candidate.q).valid:
                continue
            pairs.append(CandidateFrame(
                q=np.r_[left_candidate.q, right_candidate.q],
                branch_index=(left_candidate.branch_index + 1) * 100
                + (right_candidate.branch_index + 1),
                position_error_m=max(
                    left_candidate.position_error_m,
                    right_candidate.position_error_m,
                ),
                orientation_error_rad=max(
                    left_candidate.orientation_error_rad,
                    right_candidate.orientation_error_rad,
                ),
                wrist_risk=(left_candidate.wrist_risk
                            + right_candidate.wrist_risk),
                joint_limit_margin_rad=min(
                    left_candidate.joint_limit_margin_rad,
                    right_candidate.joint_limit_margin_rad,
                ),
                wrist_signature_value=(
                    *wrist_branch_signature(left_candidate.q),
                    *wrist_branch_signature(right_candidate.q),
                ),
            ))
        pairs.sort(key=lambda item: (
            item.wrist_risk,
            item.position_error_m,
            item.orientation_error_rad,
            -item.joint_limit_margin_rad,
            item.branch_index,
        ))
        return tuple(pairs)

    def _candidate_layers(self):
        count = len(self.task.time_s)
        layers = []
        pair_counts = np.zeros(count, dtype=int)
        warm_attempted = np.zeros(count, dtype=bool)
        warm_q = {"left": None, "right": None}
        rescue_scan_remaining = 0
        self.generator.reset()
        for row in range(count):
            per_side = {}
            warm_failed = row > 0
            for side in ("left", "right"):
                target_p, target_q = self._target(side, row)
                candidates = []
                if row > 0 and warm_q[side] is not None:
                    warm_attempted[row] = True
                    warm = self.generator.generate_warm_start_candidate(
                        side, target_p, target_q,
                        reference_q=warm_q[side],
                    )
                    if warm is not None:
                        candidates.append(warm)
                        warm_failed = False if side == "left" else warm_failed
                per_side[side] = candidates
            if row > 0:
                warm_failed = any(not per_side[side] for side in ("left", "right"))
                if warm_failed:
                    rescue_scan_remaining = max(
                        rescue_scan_remaining,
                        self.config.execution.lookahead_frames,
                    )
            need_global = row == 0 or warm_failed or rescue_scan_remaining > 0
            if need_global:
                for side in ("left", "right"):
                    target_p, target_q = self._target(side, row)
                    per_side[side].extend(self.generator.generate_target(
                        side, target_p, target_q,
                        force_stratified=(row == 0 or warm_failed),
                    ))
                    per_side[side] = self._deduplicate(per_side[side])
                if row > 0 and rescue_scan_remaining > 0:
                    rescue_scan_remaining -= 1
            layer = self._pair_layer(per_side["left"], per_side["right"])
            layers.append(layer)
            pair_counts[row] = len(layer)
            if layer:
                # The first pair is deterministic and supplies only the next
                # warm-start seed.  The scheduler later decides what is issued.
                warm_q["left"] = layer[0].q[:6].copy()
                warm_q["right"] = layer[0].q[6:].copy()
        return tuple(layers), pair_counts, warm_attempted

    @staticmethod
    def _split_pair(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=float)
        if q.shape != (12,):
            raise ValueError("paired PiperX command must contain 12 joints")
        return q[:6], q[6:]

    def _pair_state_valid(self, q: np.ndarray) -> bool:
        left, right = self._split_pair(q)
        return self.checker.state(left, right).valid

    def _pair_transition_valid(
            self, previous: np.ndarray, current: np.ndarray) -> bool:
        return self.checker.transition(
            self._split_pair(previous), self._split_pair(current)).valid

    def _full_qpos(self, pair_q: np.ndarray) -> np.ndarray:
        pair_q = np.asarray(pair_q, dtype=float)
        result = np.repeat(
            self.model.qpos0[None, :], len(pair_q), axis=0)
        result[:, self.qids["left"]] = pair_q[:, :6]
        result[:, self.qids["right"]] = pair_q[:, 6:]
        return result

    def _measure_source(self, source_qpos: np.ndarray):
        count = len(source_qpos)
        actual = {side: np.zeros((count, 7)) for side in ("left", "right")}
        position_error = {side: np.zeros(count) for side in ("left", "right")}
        orientation_error = {side: np.zeros(count) for side in ("left", "right")}
        data = mujoco.MjData(self.model)
        for row, qpos in enumerate(source_qpos):
            data.qpos[:] = qpos
            mujoco.mj_forward(self.model, data)
            for side in ("left", "right"):
                site = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
                quaternion = np.empty(4)
                mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
                actual[side][row] = np.r_[data.site_xpos[site], quaternion]
                position_error[side][row] = np.linalg.norm(
                    getattr(self.task, f"{side}_position_m")[row]
                    - data.site_xpos[site])
                residual = np.empty(3)
                mujoco.mju_subQuat(
                    residual, self.mapped_quaternions[side][row], quaternion)
                orientation_error[side][row] = np.linalg.norm(residual)
        return actual, position_error, orientation_error

    def run(self) -> RecommendedFollowResult:
        layers, pair_counts, warm_attempted = self._candidate_layers()
        schedule_config = RescueScheduleConfig(
            gate=StrictGate(
                self.config.accept.position_tolerance_m,
                self.config.accept.orientation_tolerance_rad,
                self.config.accept.branch_guard_rad,
            ),
            velocity_rad_s=self.config.execution.maximum_velocity_rad_s,
            acceleration_rad_s2=(
                self.config.execution.maximum_acceleration_rad_s2),
            settle_s=self.config.execution.settle_time_s,
            decision_s=self.config.execution.decision_time_s,
            source_rate_hz=self.config.execution.source_rate_hz,
            dwell_frames=self.config.execution.dwell_frames,
            lookahead_frames=self.config.execution.lookahead_frames,
            early_trigger_frames=min(
                12, self.config.execution.lookahead_frames),
        )
        periodic_pair = np.r_[self.periodic["left"], self.periodic["right"]]
        schedule = schedule_rescue_v31(
            layers, np.asarray(self.task.time_s, dtype=float),
            periodic_pair, schedule_config,
            state_valid=self._pair_state_valid,
            transition_valid=self._pair_transition_valid,
        )
        execution_qpos = self._full_qpos(schedule.command_q)
        source_qpos = self._full_qpos(schedule.source_command_q)
        actual, position_error, orientation_error = self._measure_source(source_qpos)
        strict = schedule.source_accepted.copy()
        for side in ("left", "right"):
            strict &= (
                (position_error[side]
                 <= self.config.accept.position_tolerance_m)
                & (orientation_error[side]
                   <= self.config.accept.orientation_tolerance_rad)
            )
        collision = np.zeros(len(source_qpos), dtype=bool)
        for row, pair in enumerate(schedule.source_command_q):
            collision[row] = not self._pair_state_valid(pair)
            if row and not collision[row]:
                collision[row] = not self._pair_transition_valid(
                    schedule.source_command_q[row - 1], pair)
        strict &= ~collision
        return RecommendedFollowResult(
            execution_qpos=execution_qpos,
            source_qpos=source_qpos,
            source_index=schedule.source_index,
            execution_time_s=schedule.execution_time_s,
            source_state=schedule.source_state,
            source_accepted=strict,
            actual_tcp=actual,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            collision=collision,
            events=schedule.events,
            candidate_pair_count=pair_counts,
            warm_start_attempted=warm_attempted,
            anchor_restart_count=self.config.anchor_restarts,
            dropped_source_frames=schedule.dropped_source_frames,
            cycle_delay_s=schedule.cycle_delay_s,
            schedule=schedule,
        )
