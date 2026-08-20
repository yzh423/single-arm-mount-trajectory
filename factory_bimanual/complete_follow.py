"""Lossless source-order retiming for strict dual-arm pose following."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Callable

import mujoco
import numpy as np

from .mujoco_candidate_generator import (
    CandidateGeneratorConfig,
    MuJoCoCandidateGenerator,
)
from .mujoco_collision_adapter import MuJoCoPairedCollisionChecker
from .piperx_recommended import PiperXRecommendedConfig
from .robot_contracts import ROBOT_CONTRACTS
from scripts.strict_mujoco_ik import joint_periodic_mask


@dataclass(frozen=True)
class CompleteRetiming:
    execution_q: np.ndarray
    execution_time_s: np.ndarray
    execution_source_index: np.ndarray
    execution_state: np.ndarray
    source_execution_index: np.ndarray
    source_reached: np.ndarray
    fixed_time_accepted: np.ndarray
    inserted_transition_frames: int
    cycle_delay_s: float
    time_scale: float
    maximum_velocity_rad_s: float
    maximum_acceleration_rad_s2: float


@dataclass(frozen=True)
class CompleteFollowResult:
    execution_qpos: np.ndarray
    source_qpos: np.ndarray
    source_pair_q: np.ndarray
    execution_time_s: np.ndarray
    execution_source_index: np.ndarray
    execution_state: np.ndarray
    source_execution_index: np.ndarray
    source_reached: np.ndarray
    fixed_time_accepted: np.ndarray
    actual_tcp: dict[str, np.ndarray]
    position_error_m: dict[str, np.ndarray]
    orientation_error_rad: dict[str, np.ndarray]
    source_collision: np.ndarray
    execution_collision: np.ndarray
    execution_state_collision: np.ndarray
    execution_incoming_transition_collision: np.ndarray
    per_side_candidate_count: dict[str, np.ndarray]
    per_side_global_rescue: dict[str, np.ndarray]
    inserted_transition_frames: int
    cycle_delay_s: float
    time_scale: float
    maximum_velocity_rad_s: float
    maximum_acceleration_rad_s2: float
    source_velocity_rad_s: np.ndarray
    source_acceleration_rad_s2: np.ndarray


def _joint_delta(previous, current, periodic):
    delta = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
    result = delta.copy()
    result[periodic] = (result[periodic] + np.pi) % (2.0*np.pi) - np.pi
    return result


def _path_derivatives(q, time_s):
    values = np.asarray(q, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if len(values) < 2:
        return (
            np.empty((0, values.shape[1]), dtype=float),
            np.empty((0, values.shape[1]), dtype=float),
        )
    dt = np.diff(time)
    velocity = np.diff(values, axis=0) / dt[:, None]
    boundary_start = velocity[0] / (0.5 * dt[0])
    boundary_end = -velocity[-1] / (0.5 * dt[-1])
    if len(velocity) < 2:
        acceleration = np.vstack((boundary_start, boundary_end))
    else:
        acceleration_dt = 0.5 * (dt[:-1] + dt[1:])
        internal = np.diff(velocity, axis=0) / acceleration_dt[:, None]
        acceleration = np.vstack((boundary_start, internal, boundary_end))
    return velocity, acceleration


def _maximum_absolute(values):
    array = np.asarray(values, dtype=float)
    return 0.0 if not array.size else float(np.max(np.abs(array)))


def _locally_retime_dynamic_limits(
    q,
    time_s,
    *,
    maximum_velocity_rad_s: float | None,
    maximum_acceleration_rad_s2: float | None,
):
    """Stretch only path segments that violate the discrete dynamic limits."""
    values = np.asarray(q, dtype=float)
    original_time = np.asarray(time_s, dtype=float)
    if len(values) < 2:
        return original_time.copy(), 1.0
    duration = float(original_time[-1] - original_time[0])
    segment_dt = np.diff(original_time).copy()
    delta = np.diff(values, axis=0)

    if maximum_velocity_rad_s is not None:
        velocity_limit = float(maximum_velocity_rad_s)
        if not np.isfinite(velocity_limit) or velocity_limit <= 0.0:
            raise ValueError(
                "maximum_velocity_rad_s must be positive and finite")
        required = np.max(np.abs(delta), axis=1) / velocity_limit
        segment_dt = np.maximum(segment_dt, required)

    if maximum_acceleration_rad_s2 is not None:
        acceleration_limit = float(maximum_acceleration_rad_s2)
        if not np.isfinite(acceleration_limit) or acceleration_limit <= 0.0:
            raise ValueError(
                "maximum_acceleration_rad_s2 must be positive and finite")
        # A segment stretch changes the velocity on both adjacent knots.  The
        # simultaneous max update avoids order-dependent left-to-right bias.
        for _ in range(10_000):
            velocity = delta / segment_dt[:, None]
            boundary_start = velocity[0] / (0.5 * segment_dt[0])
            boundary_end = -velocity[-1] / (0.5 * segment_dt[-1])
            if len(velocity) < 2:
                acceleration = np.vstack((boundary_start, boundary_end))
            else:
                knot_dt = 0.5 * (segment_dt[:-1] + segment_dt[1:])
                internal = np.diff(velocity, axis=0) / knot_dt[:, None]
                acceleration = np.vstack(
                    (boundary_start, internal, boundary_end))
            ratio = (
                np.max(np.abs(acceleration), axis=1) / acceleration_limit
            )
            if float(np.max(ratio, initial=0.0)) <= 1.0 + 1e-12:
                break
            factors = np.ones_like(segment_dt)
            if ratio[0] > 1.0:
                factors[0] = max(
                    factors[0], np.sqrt(ratio[0]) * (1.0 + 1e-12))
            internal = np.flatnonzero(ratio[1:-1] > 1.0) + 1
            if len(internal):
                local = np.sqrt(ratio[internal]) * (1.0 + 1e-12)
                np.maximum.at(factors, internal - 1, local)
                np.maximum.at(factors, internal, local)
            if ratio[-1] > 1.0:
                factors[-1] = max(
                    factors[-1], np.sqrt(ratio[-1]) * (1.0 + 1e-12))
            segment_dt *= factors
        else:
            raise RuntimeError("local dynamic retiming did not converge")

    retimed = original_time[0] + np.r_[0.0, np.cumsum(segment_dt)]
    scale = 1.0 if duration <= 0.0 else float(
        (retimed[-1] - retimed[0]) / duration)
    return retimed, scale


def retime_complete_source_path(
    source_q,
    source_time_s,
    *,
    periodic,
    branch_guard_rad,
    maximum_velocity_rad_s: float | None = None,
    maximum_acceleration_rad_s2: float | None = None,
    state_valid: Callable[[np.ndarray], bool] | None = None,
    transition_valid: Callable[[np.ndarray, np.ndarray], bool] | None = None,
):
    """Insert bounded joint-space steps without dropping a source target.

    Every source pose remains an explicit final state.  Frames inserted before
    that state are reported as retimed transitions and do not count as strict
    pose hits.  This separates fixed-time coverage from eventual complete
    source-pose coverage.
    """
    source = np.asarray(source_q, dtype=float)
    time_s = np.asarray(source_time_s, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    guard = float(branch_guard_rad)
    if (source.ndim != 2 or len(source) < 1 or
            time_s.shape != (len(source),) or
            periodic.shape != (source.shape[1],)):
        raise ValueError("source path, time and periodic mask shapes disagree")
    if (np.any(~np.isfinite(source)) or np.any(~np.isfinite(time_s)) or
            np.any(np.diff(time_s) <= 0.0) or
            not np.isfinite(guard) or guard <= 0.0):
        raise ValueError("source path and retiming parameters must be valid")
    state_valid = state_valid or (lambda _q: True)
    transition_valid = transition_valid or (
        lambda _previous, _current: True)
    if not state_valid(source[0]):
        raise ValueError("source frame 0 is not collision-free")

    execution = [source[0].copy()]
    execution_source = [0]
    execution_state = ["FOLLOW"]
    execution_time = [float(time_s[0])]
    source_execution = np.zeros(len(source), dtype=int)
    inserted = 0
    previous = source[0].copy()
    for row in range(1, len(source)):
        delta = _joint_delta(previous, source[row], periodic)
        steps = max(1, int(np.ceil(
            float(np.max(np.abs(delta), initial=0.0)) / guard)))
        target_unwrapped = previous + delta
        if not transition_valid(previous, target_unwrapped):
            raise ValueError(
                f"source edge {row-1}->{row} has no collision-free transition")
        interval = float(time_s[row] - time_s[row - 1])
        for step in range(1, steps + 1):
            fraction = step / steps
            command = previous + fraction * delta
            if not state_valid(command):
                raise ValueError(
                    f"retimed source edge {row-1}->{row} enters collision")
            final = step == steps
            execution.append(command)
            execution_source.append(row)
            if steps == 1:
                execution_state.append("FOLLOW")
            elif final:
                execution_state.append("FOLLOW_RETIMED")
            else:
                execution_state.append("RETIMED_TRANSITION")
            execution_time.append(execution_time[-1] + interval)
        inserted += steps - 1
        source_execution[row] = len(execution) - 1
        previous = target_unwrapped
    execution_array = np.asarray(execution, dtype=float)
    execution_time_array = np.asarray(execution_time, dtype=float)
    execution_time_array, time_scale = _locally_retime_dynamic_limits(
        execution_array,
        execution_time_array,
        maximum_velocity_rad_s=maximum_velocity_rad_s,
        maximum_acceleration_rad_s2=maximum_acceleration_rad_s2,
    )
    velocity, acceleration = _path_derivatives(
        execution_array, execution_time_array)
    source_reached = np.ones(len(source), dtype=bool)
    fixed = np.isclose(
        execution_time_array[source_execution], time_s,
        rtol=0.0, atol=1e-12,
    )
    delay = float(execution_time_array[-1] - time_s[-1])
    return CompleteRetiming(
        execution_q=execution_array,
        execution_time_s=execution_time_array,
        execution_source_index=np.asarray(execution_source, dtype=int),
        execution_state=np.asarray(execution_state, dtype="U32"),
        source_execution_index=source_execution,
        source_reached=source_reached,
        fixed_time_accepted=fixed,
        inserted_transition_frames=inserted,
        cycle_delay_s=delay,
        time_scale=float(time_scale),
        maximum_velocity_rad_s=_maximum_absolute(velocity),
        maximum_acceleration_rad_s2=_maximum_absolute(acceleration),
    )


class CompleteFollowInfeasibleError(RuntimeError):
    """Raised only after strict warm-start and global rescue both fail."""

    def __init__(self, row, message):
        super().__init__(f"source frame {row}: {message}")
        self.row = int(row)


class CompleteFollowRunner:
    """Strict local/global IK with lossless branch-switch retiming."""

    def __init__(
        self,
        model: mujoco.MjModel,
        task,
        mapped_quaternions,
        config: PiperXRecommendedConfig,
        *,
        collision_clearance_margin_m: float | None = None,
        maximum_candidates_per_side: int = 8,
    ):
        self.model = model
        self.task = task
        self.mapped_quaternions = {
            side: np.asarray(mapped_quaternions[side], dtype=float)
            for side in ("left", "right")
        }
        self.config = config
        self.contract = ROBOT_CONTRACTS["piperx"]
        count = len(np.asarray(task.time_s))
        if count < 1:
            raise ValueError("complete follow requires at least one source frame")
        for side in ("left", "right"):
            position = np.asarray(
                getattr(task, f"{side}_position_m"), dtype=float)
            if (position.shape != (count, 3) or
                    self.mapped_quaternions[side].shape != (count, 4)):
                raise ValueError("target pose arrays must align with source time")
        self.names = {
            side: {
                "joints": self.contract.prefixed_joint_names(side),
                "site": f"{side}_tcp",
            }
            for side in ("left", "right")
        }
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.generator = MuJoCoCandidateGenerator(
            model,
            data,
            self.contract,
            name_map=self.names,
            config=CandidateGeneratorConfig(
                position_tolerance_m=config.accept.position_tolerance_m,
                orientation_tolerance_rad=(
                    config.accept.orientation_tolerance_rad),
                damping=config.dls.damping,
                step_scale=config.dls.step_scale,
                maximum_step_rad=config.dls.maximum_step_rad,
                position_error_clip_m=config.dls.position_error_clip_m,
                orientation_error_clip_rad=(
                    config.dls.orientation_error_clip_rad),
                max_iterations=config.dls.max_iterations,
                global_seed_count=config.anchor_restarts,
                maximum_candidates=int(maximum_candidates_per_side),
                constrained_fallback_enabled=True,
                constrained_fallback_seed_count=20,
                constrained_fallback_max_iterations=300,
                stratified_seed_enabled=True,
                wrist_risk_enabled=True,
                dedup_rad=np.deg2rad(0.25),
                rolling_early_stop_candidates=4,
            ),
        )
        self.checker = MuJoCoPairedCollisionChecker(
            model,
            data,
            self.names,
            transition_steps=11,
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
            self.qids[side] = np.asarray(
                model.jnt_qposadr[joint_ids], dtype=int)
            self.periodic[side] = joint_periodic_mask(model, joint_ids)

    def _target(self, side, row):
        return (
            np.asarray(
                getattr(self.task, f"{side}_position_m")[row], dtype=float),
            self.mapped_quaternions[side][row],
        )

    @staticmethod
    def _deduplicate(candidates):
        output = []
        for candidate in candidates:
            if not any(np.linalg.norm(candidate.q-old.q) <= 1e-6
                       for old in output):
                output.append(candidate)
        return output

    def _global(self, side, row):
        position, quaternion = self._target(side, row)
        return self.generator.generate_target(
            side, position, quaternion, force_stratified=True)

    def _warm(self, side, row, reference):
        position, quaternion = self._target(side, row)
        return self.generator.generate_warm_start_candidate(
            side, position, quaternion, reference_q=reference)

    def _pair_candidates(self, per_side):
        return [
            (left, right)
            for left in per_side["left"]
            for right in per_side["right"]
        ]

    def _pair_delta(self, previous, pair):
        current = np.r_[pair[0].q, pair[1].q]
        periodic = np.r_[self.periodic["left"], self.periodic["right"]]
        return _joint_delta(previous, current, periodic)

    def _choose_pair(self, pairs, previous):
        viable = []
        for pair in pairs:
            current = np.r_[pair[0].q, pair[1].q]
            state_collision = not self.checker.state(
                pair[0].q, pair[1].q).valid
            if previous is not None:
                delta = self._pair_delta(previous, pair)
                unwrapped = previous + delta
                left_previous, right_previous = previous[:6], previous[6:]
                left_current, right_current = unwrapped[:6], unwrapped[6:]
                transition_collision = not self.checker.transition(
                        (left_previous, right_previous),
                        (left_current, right_current)).valid
                maximum_delta = float(np.max(np.abs(delta), initial=0.0))
                steps = max(1, int(np.ceil(
                    maximum_delta/self.config.accept.branch_guard_rad)))
            else:
                maximum_delta = 0.0
                steps = 1
                transition_collision = state_collision
            score = (
                int(state_collision),
                int(transition_collision),
                steps,
                maximum_delta,
                pair[0].wrist_risk + pair[1].wrist_risk,
                max(pair[0].pose_cost, pair[1].pose_cost),
                -min(pair[0].joint_limit_margin_rad,
                     pair[1].joint_limit_margin_rad),
                *current.tolist(),
            )
            viable.append((score, pair))
        return None if not viable else min(viable, key=lambda item: item[0])[1]

    def _pair_has_collision(self, pair, previous):
        if pair is None:
            return False
        if not self.checker.state(pair[0].q, pair[1].q).valid:
            return True
        if previous is None:
            return False
        delta = self._pair_delta(previous, pair)
        current = previous + delta
        return not self.checker.transition(
            (previous[:6], previous[6:]),
            (current[:6], current[6:]),
        ).valid

    def _source_path(self):
        count = len(self.task.time_s)
        selected = np.zeros((count, 12), dtype=float)
        candidate_count = {
            side: np.zeros(count, dtype=int) for side in ("left", "right")}
        global_rescue = {
            side: np.zeros(count, dtype=bool) for side in ("left", "right")}
        previous_side = {"left": None, "right": None}
        previous_pair = None
        self.generator.reset()
        for row in range(count):
            per_side = {}
            for side in ("left", "right"):
                candidates = []
                if row > 0:
                    warm = self._warm(side, row, previous_side[side])
                    if warm is not None:
                        candidates.append(warm)
                if not candidates:
                    candidates.extend(self._global(side, row))
                    global_rescue[side][row] = True
                per_side[side] = self._deduplicate(candidates)
                candidate_count[side][row] = len(per_side[side])
                if not per_side[side]:
                    raise CompleteFollowInfeasibleError(
                        row, f"{side} has no strict local or global IK solution")
            pairs = self._pair_candidates(per_side)
            chosen = self._choose_pair(pairs, previous_pair)
            if (row > 0 and
                    (chosen is None or
                     self._pair_has_collision(chosen, previous_pair))):
                # Collision or a disconnected branch can require alternatives
                # even when both local warm starts individually succeeded.
                for side in ("left", "right"):
                    if not global_rescue[side][row]:
                        per_side[side] = self._deduplicate(
                            [*per_side[side], *self._global(side, row)])
                        global_rescue[side][row] = True
                        candidate_count[side][row] = len(per_side[side])
                pairs = self._pair_candidates(per_side)
                chosen = self._choose_pair(pairs, previous_pair)
            if chosen is None:
                collision_classes = Counter()
                for left in per_side["left"]:
                    for right in per_side["right"]:
                        report = self.checker.state(left.q, right.q)
                        collision_classes.update(
                            item.value for item in report.classes)
                raise CompleteFollowInfeasibleError(
                    row,
                    "no collision-free connected dual-arm IK pair "
                    f"(safe target pairs={len(pairs)}, "
                    f"left candidates={len(per_side['left'])}, "
                    f"right candidates={len(per_side['right'])}, "
                    f"collision classes={dict(collision_classes)})",
                )
            pair_q = np.r_[chosen[0].q, chosen[1].q]
            if previous_pair is not None:
                pair_q = previous_pair + self._pair_delta(
                    previous_pair, chosen)
            selected[row] = pair_q
            previous_pair = pair_q
            previous_side["left"] = pair_q[:6].copy()
            previous_side["right"] = pair_q[6:].copy()
        return selected, candidate_count, global_rescue

    def _full_qpos(self, pair_q):
        pair_q = np.asarray(pair_q, dtype=float)
        output = np.repeat(self.model.qpos0[None, :], len(pair_q), axis=0)
        output[:, self.qids["left"]] = pair_q[:, :6]
        output[:, self.qids["right"]] = pair_q[:, 6:]
        return output

    def _measure(self, source_qpos):
        count = len(source_qpos)
        actual = {side: np.zeros((count, 7)) for side in ("left", "right")}
        position_error = {
            side: np.zeros(count) for side in ("left", "right")}
        orientation_error = {
            side: np.zeros(count) for side in ("left", "right")}
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

    def run(self):
        source_pair, candidate_count, global_rescue = self._source_path()
        periodic = np.r_[self.periodic["left"], self.periodic["right"]]
        retiming = retime_complete_source_path(
            source_pair,
            np.asarray(self.task.time_s, dtype=float),
            periodic=periodic,
            branch_guard_rad=self.config.accept.branch_guard_rad,
            maximum_velocity_rad_s=(
                self.config.execution.maximum_velocity_rad_s),
            maximum_acceleration_rad_s2=(
                self.config.execution.maximum_acceleration_rad_s2),
        )
        source_velocity, source_acceleration = _path_derivatives(
            source_pair, np.asarray(self.task.time_s, dtype=float))
        source_qpos = self._full_qpos(source_pair)
        execution_qpos = self._full_qpos(retiming.execution_q)
        actual, position_error, orientation_error = self._measure(source_qpos)
        reached = retiming.source_reached.copy()
        for side in ("left", "right"):
            reached &= (
                (position_error[side]
                 <= self.config.accept.position_tolerance_m)
                & (orientation_error[side]
                   <= self.config.accept.orientation_tolerance_rad)
            )
        collision = np.zeros(len(source_pair), dtype=bool)
        for row, pair in enumerate(source_pair):
            collision[row] = not self.checker.state(
                pair[:6], pair[6:]).valid
            if row and not collision[row]:
                previous = source_pair[row-1]
                collision[row] = not self.checker.transition(
                    (previous[:6], previous[6:]),
                    (pair[:6], pair[6:])).valid
        execution_state_collision = np.zeros(
            len(retiming.execution_q), dtype=bool)
        execution_incoming_transition_collision = np.zeros(
            len(retiming.execution_q), dtype=bool)
        for row, pair in enumerate(retiming.execution_q):
            execution_state_collision[row] = not self.checker.state(
                pair[:6], pair[6:]).valid
            if row:
                previous = retiming.execution_q[row-1]
                execution_incoming_transition_collision[row] = not self.checker.transition(
                    (previous[:6], previous[6:]),
                    (pair[:6], pair[6:])).valid
        execution_collision = (
            execution_state_collision
            | execution_incoming_transition_collision
        )
        return CompleteFollowResult(
            execution_qpos=execution_qpos,
            source_qpos=source_qpos,
            source_pair_q=source_pair,
            execution_time_s=retiming.execution_time_s,
            execution_source_index=retiming.execution_source_index,
            execution_state=retiming.execution_state,
            source_execution_index=retiming.source_execution_index,
            source_reached=reached,
            fixed_time_accepted=retiming.fixed_time_accepted & reached,
            actual_tcp=actual,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            source_collision=collision,
            execution_collision=execution_collision,
            execution_state_collision=execution_state_collision,
            execution_incoming_transition_collision=(
                execution_incoming_transition_collision),
            per_side_candidate_count=candidate_count,
            per_side_global_rescue=global_rescue,
            inserted_transition_frames=retiming.inserted_transition_frames,
            cycle_delay_s=retiming.cycle_delay_s,
            time_scale=retiming.time_scale,
            maximum_velocity_rad_s=retiming.maximum_velocity_rad_s,
            maximum_acceleration_rad_s2=(
                retiming.maximum_acceleration_rad_s2),
            source_velocity_rad_s=source_velocity,
            source_acceleration_rad_s2=source_acceleration,
        )


__all__ = [
    "CompleteFollowInfeasibleError",
    "CompleteFollowResult",
    "CompleteFollowRunner",
    "CompleteRetiming",
    "retime_complete_source_path",
]
