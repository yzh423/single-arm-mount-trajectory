"""Continuous position-path IK evaluated by the same MuJoCo model as rendering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import mujoco
import numpy as np

from scripts.rolling_multibranch_ik import (
    BranchCandidate,
    select_receding_horizon_path,
    select_rolling_branch,
)


def joint_periodic_mask(model: mujoco.MjModel, joint_ids: Sequence[int]) -> np.ndarray:
    """Return the physical topology: only unlimited hinge joints wrap."""
    ids = np.asarray(joint_ids, dtype=int)
    return ((model.jnt_type[ids] == mujoco.mjtJoint.mjJNT_HINGE) &
            ~model.jnt_limited[ids].astype(bool))


def wrapped_joint_delta(candidate: np.ndarray, previous: np.ndarray,
                        periodic: np.ndarray) -> np.ndarray:
    """Return shortest angular deltas only for joints spanning a full revolution."""
    delta = np.asarray(candidate, dtype=float) - np.asarray(previous, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    if delta.shape != periodic.shape:
        raise ValueError("joint values and periodic mask must have matching shapes")
    delta[periodic] = (delta[periodic] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def continuity_jump(has_valid_pose: bool, maximum_joint_delta_rad: float,
                    threshold_rad: float) -> bool:
    """Continuity is meaningful only after a full 6D pose branch is established."""
    return bool(has_valid_pose and maximum_joint_delta_rad > threshold_rad)


def continuity_recovery_step(candidate: np.ndarray, previous: np.ndarray,
                             periodic: np.ndarray,
                             maximum_step_rad: float | np.ndarray) -> np.ndarray:
    """Move toward an IK branch without freezing after one oversized transition."""
    candidate = np.asarray(candidate, dtype=float)
    previous = np.asarray(previous, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    maximum_step = np.broadcast_to(np.asarray(maximum_step_rad, dtype=float), candidate.shape)
    if np.any(~np.isfinite(maximum_step)) or np.any(maximum_step <= 0.0):
        raise ValueError("maximum_step_rad must be finite and positive")
    delta = wrapped_joint_delta(candidate, previous, periodic)
    limited = np.clip(delta, -maximum_step, maximum_step)
    recovered = previous + limited
    # Preserve an in-range equivalent supplied by the solver when a periodic
    # joint can reach it within this frame.  Consumers compare it with wrapped
    # deltas, so crossing the numeric range boundary remains physically short.
    use_candidate = periodic & (np.abs(delta) <= maximum_step)
    recovered[use_candidate] = candidate[use_candidate]
    return recovered


def realized_velocity_violation(realized_delta, *, dt_s, velocity_limit_rad_s,
                                initializing=False) -> bool:
    """Report only actual output-speed violations, not rejected IK jumps."""
    if initializing:
        return False
    delta = np.asarray(realized_delta, dtype=float)
    limits = np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float), delta.shape)
    if not np.isfinite(dt_s) or dt_s <= 0 or np.any(limits <= 0):
        raise ValueError("dt_s and velocity limits must be positive")
    return bool(np.any(np.abs(delta) / float(dt_s) > limits + 1e-12))


def joint_discontinuity_from_recovery_modes(recovery_mode) -> np.ndarray:
    """Classify only blocked valid-pose branch transitions as discontinuity."""
    modes = np.asarray(recovery_mode).astype(str)
    return np.isin(modes, ("limited_step", "hold_wrist_branch_switch"))


@dataclass(frozen=True)
class PositionPathResult:
    q: np.ndarray
    reached_xyz_m: np.ndarray
    position_error_m: np.ndarray
    success: np.ndarray
    joint_limits_respected: bool


@dataclass(frozen=True)
class PosePathResult:
    q: np.ndarray
    reached_xyz_m: np.ndarray
    reached_quaternion_wxyz: np.ndarray
    position_error_m: np.ndarray
    orientation_error_rad: np.ndarray
    success: np.ndarray
    joint_discontinuity: np.ndarray
    joint_limits_respected: bool
    velocity_violation: np.ndarray
    acceleration_warning: np.ndarray
    branch_count: np.ndarray
    chosen_branch_index: np.ndarray
    recovery_mode: np.ndarray
    singularity_margin: np.ndarray
    joint_limit_margin: np.ndarray


def generate_pose_candidate_layers(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: Sequence[str],
    targets_xyz_m: np.ndarray,
    targets_quaternion_wxyz: np.ndarray,
    *,
    candidates_per_frame: int = 8,
    global_seed_count: int = 16,
    iterations: int = 120,
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = np.deg2rad(1.5),
    candidate_collision_free: Callable[[np.ndarray], bool] | None = None,
    rng_seed: int = 20260814,
) -> tuple[tuple[BranchCandidate, ...], ...]:
    """Generate several real MuJoCo IK branches for every target pose."""
    targets = np.asarray(targets_xyz_m, dtype=float)
    quaternions = np.asarray(targets_quaternion_wxyz, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets_xyz_m must have shape (frames, 3)")
    if quaternions.shape != (len(targets), 4):
        raise ValueError("targets_quaternion_wxyz must have shape (frames, 4)")
    if candidates_per_frame < 2 or global_seed_count < candidates_per_frame:
        raise ValueError("candidate and global seed counts are inconsistent")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in joint_names]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError("all layered IK joints must exist in the model")
    limits = np.asarray([
        model.jnt_range[joint_id] if model.jnt_limited[joint_id]
        else (-np.pi, np.pi)
        for joint_id in joint_ids], dtype=float)
    lower, upper = limits[:, 0], limits[:, 1]
    midpoint = 0.5 * (lower + upper)
    periodic = joint_periodic_mask(model, joint_ids)
    rng = np.random.default_rng(rng_seed)
    static_seeds = [midpoint]
    static_seeds.extend(rng.uniform(lower, upper) for _ in range(global_seed_count - 1))
    prior_solutions: list[np.ndarray] = []
    layers: list[tuple[BranchCandidate, ...]] = []
    for frame_index, (target, quaternion) in enumerate(zip(targets, quaternions)):
        seeds = prior_solutions + static_seeds
        candidates: list[BranchCandidate] = []
        for seed_index, seed in enumerate(seeds):
            proposal = solve_pose_path(
                model, site_name, joint_names, target[None, :], quaternion[None, :],
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                iterations=iterations, restarts=1, initial_q=np.asarray(seed),
                maximum_frame_jump_rad=np.inf, hold_invalid=False,
                rng_seed=rng_seed + frame_index * max(1, len(seeds)) + seed_index,
            )
            q = np.asarray(proposal.q[0], dtype=float)
            if any(np.max(np.abs(wrapped_joint_delta(q, item.q, periodic)))
                   <= np.deg2rad(1.0) for item in candidates):
                continue
            pose_valid = bool(
                proposal.position_error_m[0] <= position_tolerance_m
                and proposal.orientation_error_rad[0] <= orientation_tolerance_rad)
            collision_free = (True if candidate_collision_free is None
                              else bool(candidate_collision_free(q)))
            candidates.append(BranchCandidate(
                q=q.copy(), pose_valid=pose_valid,
                collision_free=collision_free,
                position_error_m=float(proposal.position_error_m[0]),
                orientation_error_rad=float(proposal.orientation_error_rad[0]),
                joint_limit_margin=float(proposal.joint_limit_margin[0]),
                singularity_margin=float(proposal.singularity_margin[0]),
                index=len(candidates),
            ))
        candidates.sort(key=lambda item: (
            not item.pose_valid,
            not item.collision_free,
            item.position_error_m + 0.1 * item.orientation_error_rad,
            -item.joint_limit_margin,
            item.index,
        ))
        candidates = candidates[:candidates_per_frame]
        # Indices identify candidates inside this layer and must remain dense
        # after ranking/truncation for artifact traceability.
        layer = tuple(BranchCandidate(
            q=item.q, pose_valid=item.pose_valid,
            collision_free=item.collision_free,
            position_error_m=item.position_error_m,
            orientation_error_rad=item.orientation_error_rad,
            joint_limit_margin=item.joint_limit_margin,
            singularity_margin=item.singularity_margin,
            index=index,
        ) for index, item in enumerate(candidates))
        layers.append(layer)
        prior_solutions = [item.q.copy() for item in layer
                           if item.pose_valid and item.collision_free]
    return tuple(layers)


def solve_pose_path_layered(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: Sequence[str],
    targets_xyz_m: np.ndarray,
    targets_quaternion_wxyz: np.ndarray,
    *,
    time_s: np.ndarray,
    candidates_per_frame: int = 8,
    global_seed_count: int = 16,
    horizon: int = 12,
    beam_width: int = 8,
    velocity_limit_rad_s: float | np.ndarray = np.deg2rad(720.0),
    maximum_frame_jump_rad: float | np.ndarray = np.deg2rad(25.0),
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = np.deg2rad(1.5),
    iterations: int = 120,
    candidate_collision_free: Callable[[np.ndarray], bool] | None = None,
    transition_collision_free: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    rng_seed: int = 20260814,
) -> PosePathResult:
    """Solve a pose sequence through explicit per-frame IK candidate layers."""
    targets = np.asarray(targets_xyz_m, dtype=float)
    quaternions = np.asarray(targets_quaternion_wxyz, dtype=float)
    times = np.asarray(time_s, dtype=float)
    if times.shape != (len(targets),) or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_s must be strictly increasing and match targets")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in joint_names]
    limits = np.asarray([
        model.jnt_range[joint_id] if model.jnt_limited[joint_id]
        else (-np.pi, np.pi)
        for joint_id in joint_ids], dtype=float)
    lower, upper = limits[:, 0], limits[:, 1]
    midpoint = 0.5 * (lower + upper)
    periodic = joint_periodic_mask(model, joint_ids)
    layers = generate_pose_candidate_layers(
        model, site_name, joint_names, targets, quaternions,
        candidates_per_frame=candidates_per_frame,
        global_seed_count=global_seed_count, iterations=iterations,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
        candidate_collision_free=candidate_collision_free,
        rng_seed=rng_seed,
    )
    intervals = np.r_[times[1] - times[0] if len(times) > 1 else 1.0,
                      np.diff(times)]
    velocity_limits = np.broadcast_to(
        np.asarray(velocity_limit_rad_s, dtype=float), midpoint.shape)
    jump_limits = np.broadcast_to(
        np.asarray(maximum_frame_jump_rad, dtype=float), midpoint.shape)
    selected = select_receding_horizon_path(
        layers=layers, initial_q=midpoint, periodic=periodic, dt_s=intervals,
        velocity_limit_rad_s=velocity_limits, cap_rad=jump_limits,
        horizon=horizon, beam_width=beam_width,
        transition_valid=transition_collision_free,
    )

    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    addresses = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
    reached, reached_quaternion, position_error, orientation_error = [], [], [], []
    for q, target, target_quaternion in zip(selected.q, targets, quaternions):
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
        residual = np.empty(3)
        mujoco.mju_subQuat(residual, target_quaternion, quaternion)
        reached.append(data.site_xpos[site_id].copy())
        reached_quaternion.append(quaternion)
        position_error.append(float(np.linalg.norm(target - data.site_xpos[site_id])))
        orientation_error.append(float(np.linalg.norm(residual)))
    position_error_array = np.asarray(position_error)
    orientation_error_array = np.asarray(orientation_error)
    success = (selected.pose_valid
               & (position_error_array <= position_tolerance_m)
               & (orientation_error_array <= orientation_tolerance_rad))
    velocity_violation = np.zeros(len(selected.q), dtype=bool)
    acceleration_warning = np.zeros(len(selected.q), dtype=bool)
    previous_velocity = np.zeros_like(midpoint)
    for frame_index in range(1, len(selected.q)):
        delta = wrapped_joint_delta(selected.q[frame_index], selected.q[frame_index - 1], periodic)
        velocity = delta / intervals[frame_index]
        velocity_violation[frame_index] = bool(
            np.any(np.abs(velocity) > velocity_limits + 1e-12))
        acceleration = (velocity - previous_velocity) / intervals[frame_index]
        acceleration_warning[frame_index] = bool(
            np.any(np.abs(acceleration) > velocity_limits / intervals[frame_index]))
        previous_velocity = velocity
    singularity = np.full(len(layers), np.nan)
    joint_margin = np.full(len(layers), np.nan)
    for frame_index, candidate_index in enumerate(selected.selected_indices):
        match = next((item for item in layers[frame_index]
                      if item.index == candidate_index), None)
        if match is not None:
            singularity[frame_index] = match.singularity_margin
            joint_margin[frame_index] = match.joint_limit_margin
    joint_discontinuity = joint_discontinuity_from_recovery_modes(
        selected.recovery_mode)
    return PosePathResult(
        q=selected.q,
        reached_xyz_m=np.asarray(reached),
        reached_quaternion_wxyz=np.asarray(reached_quaternion),
        position_error_m=position_error_array,
        orientation_error_rad=orientation_error_array,
        success=success,
        joint_discontinuity=joint_discontinuity,
        joint_limits_respected=bool(
            np.all(selected.q >= lower - 1e-10)
            and np.all(selected.q <= upper + 1e-10)),
        velocity_violation=velocity_violation,
        acceleration_warning=acceleration_warning,
        branch_count=np.asarray([len(layer) for layer in layers], dtype=int),
        chosen_branch_index=selected.selected_indices,
        recovery_mode=selected.recovery_mode,
        singularity_margin=singularity,
        joint_limit_margin=joint_margin,
    )


def solve_pose_path(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: Sequence[str],
    targets_xyz_m: np.ndarray,
    targets_quaternion_wxyz: np.ndarray,
    *,
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = np.deg2rad(1.5),
    iterations: int = 240,
    damping: float = 5e-4,
    restarts: int = 12,
    max_step_rad: float = 0.18,
    maximum_frame_jump_rad: float = np.deg2rad(25.0),
    initial_q: np.ndarray | None = None,
    time_s: np.ndarray | None = None,
    velocity_limit_rad_s: float | np.ndarray = np.deg2rad(720.0),
    rng_seed: int = 20260807,
    hold_invalid: bool = True,
) -> PosePathResult:
    """Track a continuous TCP pose path with MuJoCo's site Jacobian."""
    targets = np.asarray(targets_xyz_m, dtype=float)
    quaternions = np.asarray(targets_quaternion_wxyz, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets_xyz_m must have shape (frames, 3)")
    if quaternions.shape != (len(targets), 4):
        raise ValueError("targets_quaternion_wxyz must have shape (frames, 4)")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("target quaternion must be nonzero")
    quaternions = quaternions / norms[:, None]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if site_id < 0 or any(j < 0 for j in joint_ids):
        raise ValueError("site or joint not found")
    q_addresses = np.asarray([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
    dof_addresses = np.asarray([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
    limits = np.asarray([model.jnt_range[j] if model.jnt_limited[j] else (-np.pi, np.pi) for j in joint_ids])
    lower, upper = limits[:, 0], limits[:, 1]
    midpoint = 0.5 * (lower + upper)
    periodic = joint_periodic_mask(model, joint_ids)
    if time_s is None:
        frame_dt = np.ones(len(targets), dtype=float)
    else:
        times = np.asarray(time_s, dtype=float)
        if times.shape != (len(targets),) or np.any(~np.isfinite(times)):
            raise ValueError("time_s must contain one finite timestamp per target")
        differences = np.diff(times)
        if np.any(differences <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        frame_dt = np.r_[differences[0] if len(differences) else 1.0, differences]
    velocity_limits = np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float), midpoint.shape)
    if np.any(velocity_limits <= 0.0):
        raise ValueError("velocity_limit_rad_s must be positive")
    data = mujoco.MjData(model); rng = np.random.default_rng(rng_seed)
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    previous = midpoint.copy() if initial_q is None else np.clip(np.asarray(initial_q, dtype=float), lower, upper)
    if previous.shape != midpoint.shape:
        raise ValueError(f"initial_q must have shape {midpoint.shape}")
    solved=[]; reached=[]; reached_q=[]; pe_all=[]; oe_all=[]; success=[]; discontinuity=[]
    velocity_violations=[]; acceleration_warnings=[]; branch_counts=[]; chosen_indices=[]
    recovery_modes=[]; singularity_margins=[]; joint_limit_margins=[]
    previous_velocity = np.zeros_like(previous)
    has_valid_pose = False
    for frame_index, (target, target_quat) in enumerate(zip(targets, quaternions)):
        held_q = previous.copy()
        seeds = [previous, midpoint] + [rng.uniform(lower, upper) for _ in range(max(0, restarts - 2))]
        best = (np.inf, np.inf, np.inf, previous.copy(), np.full(3, np.nan), np.full(4, np.nan))
        valid_candidates = 0
        selected_index = -1
        selected_singularity = 0.0
        selected_limit_margin = 0.0
        for seed_index, seed in enumerate(seeds):
            q = np.clip(np.asarray(seed).copy(), lower, upper)
            for _ in range(iterations):
                data.qpos[q_addresses] = q; mujoco.mj_forward(model, data)
                current_quat = np.zeros(4); mujoco.mju_mat2Quat(current_quat, data.site_xmat[site_id])
                rotation_residual = np.zeros(3); mujoco.mju_subQuat(rotation_residual, target_quat, current_quat)
                # mju_subQuat expresses angular displacement in the current
                # site's local frame; mj_jacSite's rotational rows are world-aligned.
                rotation_residual = data.site_xmat[site_id].reshape(3, 3) @ rotation_residual
                position_residual = target - data.site_xpos[site_id]
                if np.linalg.norm(position_residual) <= position_tolerance_m and np.linalg.norm(rotation_residual) <= orientation_tolerance_rad:
                    break
                mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                jacobian = np.vstack((jacp[:, dof_addresses], jacr[:, dof_addresses]))
                residual = np.r_[position_residual, rotation_residual]
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), residual)
                delta_norm = np.linalg.norm(delta)
                if delta_norm > max_step_rad:
                    delta *= max_step_rad / delta_norm
                q = np.clip(q + delta, lower, upper)
            data.qpos[q_addresses] = q; mujoco.mj_forward(model, data)
            current_quat = np.zeros(4); mujoco.mju_mat2Quat(current_quat, data.site_xmat[site_id])
            rotation_residual = np.zeros(3); mujoco.mju_subQuat(rotation_residual, target_quat, current_quat)
            pe = float(np.linalg.norm(target - data.site_xpos[site_id])); oe = float(np.linalg.norm(rotation_residual))
            joint_delta = wrapped_joint_delta(q, previous, periodic)
            joint_distance = float(np.linalg.norm(joint_delta))
            pose_valid_candidate = pe <= position_tolerance_m and oe <= orientation_tolerance_rad
            valid_candidates += int(pose_valid_candidate)
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            candidate_jacobian = np.vstack((jacp[:, dof_addresses], jacr[:, dof_addresses]))
            singular_values = np.linalg.svd(candidate_jacobian, compute_uv=False)
            singularity = float(singular_values[-1]) if len(singular_values) else 0.0
            span = np.maximum(upper - lower, 1e-12)
            limit_margin = float(np.min(np.minimum(q - lower, upper - q) / span))
            candidate_key = (
                0 if pe <= position_tolerance_m and oe <= orientation_tolerance_rad else 1,
                joint_distance if pe <= position_tolerance_m and oe <= orientation_tolerance_rad else pe + 0.1 * oe,
                pe + 0.1 * oe,
            )
            best_key = (
                0 if best[0] <= position_tolerance_m and best[1] <= orientation_tolerance_rad else 1,
                best[2] if best[0] <= position_tolerance_m and best[1] <= orientation_tolerance_rad else best[0] + 0.1 * best[1],
                best[0] + 0.1 * best[1],
            )
            if candidate_key < best_key:
                best = (pe, oe, joint_distance, q.copy(), data.site_xpos[site_id].copy(), current_quat.copy())
                selected_index = seed_index
                selected_singularity = singularity
                selected_limit_margin = limit_margin
        pe, oe, joint_distance, candidate_q, xyz, quat = best
        # The first pose initializes the episode and therefore has no temporal
        # predecessor. Continuity is enforced from the second measured frame.
        candidate_delta = wrapped_joint_delta(candidate_q, previous, periodic)
        allowed_delta = np.minimum(velocity_limits * frame_dt[frame_index],
                                   maximum_frame_jump_rad)
        jump = bool(has_valid_pose and np.any(np.abs(candidate_delta) > allowed_delta))
        frame_success = pe <= position_tolerance_m and oe <= orientation_tolerance_rad and not jump
        if frame_success:
            previous = candidate_q
            has_valid_pose = True
        else:
            # An oversized but pose-valid branch must not latch the remainder
            # of the episode at a stale posture. Advance toward it at the same
            # per-frame continuity limit; other IK failures still hold safely.
            pose_valid = pe <= position_tolerance_m and oe <= orientation_tolerance_rad
            previous = (continuity_recovery_step(candidate_q, held_q, periodic,
                                                 allowed_delta)
                        if jump and pose_valid else
                        (candidate_q.copy() if not hold_invalid else held_q))
            data.qpos[q_addresses] = previous; mujoco.mj_forward(model, data)
            xyz = data.site_xpos[site_id].copy()
            quat = np.zeros(4); mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
            rotation_residual = np.zeros(3); mujoco.mju_subQuat(rotation_residual, target_quat, quat)
            pe = float(np.linalg.norm(target - xyz)); oe = float(np.linalg.norm(rotation_residual))
        solved.append(previous.copy()); reached.append(xyz); reached_q.append(quat); pe_all.append(pe); oe_all.append(oe)
        success.append(frame_success); discontinuity.append(jump)
        realized_delta = wrapped_joint_delta(previous, held_q, periodic)
        velocity = realized_delta / frame_dt[frame_index]
        acceleration = (velocity - previous_velocity) / frame_dt[frame_index]
        velocity_violations.append(realized_velocity_violation(
            realized_delta, dt_s=frame_dt[frame_index],
            velocity_limit_rad_s=velocity_limits,
            initializing=frame_index == 0))
        acceleration_warnings.append(bool(np.any(np.abs(acceleration) > velocity_limits / frame_dt[frame_index])))
        branch_counts.append(valid_candidates); chosen_indices.append(selected_index)
        recovery_modes.append("limited_step" if jump and pose_valid else ("hold" if not frame_success else "none"))
        singularity_margins.append(selected_singularity); joint_limit_margins.append(selected_limit_margin)
        previous_velocity = velocity
    q_array = np.asarray(solved)
    return PosePathResult(q_array, np.asarray(reached), np.asarray(reached_q), np.asarray(pe_all), np.asarray(oe_all),
                          np.asarray(success, dtype=bool), np.asarray(discontinuity, dtype=bool),
                          bool(np.all(q_array >= lower-1e-10) and np.all(q_array <= upper+1e-10)),
                          np.asarray(velocity_violations, dtype=bool), np.asarray(acceleration_warnings, dtype=bool),
                          np.asarray(branch_counts, dtype=int), np.asarray(chosen_indices, dtype=int),
                          np.asarray(recovery_modes), np.asarray(singularity_margins, dtype=float),
                          np.asarray(joint_limit_margins, dtype=float))


def _solve_pose_path_precomputed_multibranch(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: Sequence[str],
    targets_xyz_m: np.ndarray,
    targets_quaternion_wxyz: np.ndarray,
    *,
    time_s: np.ndarray,
    branch_candidates: int = 8,
    horizon: int = 12,
    beam_width: int = 8,
    velocity_limit_rad_s: float | np.ndarray = np.deg2rad(720.0),
    maximum_frame_jump_rad: float = np.deg2rad(25.0),
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = np.deg2rad(1.5),
    iterations: int = 240,
    damping: float = 5e-4,
    initial_q: np.ndarray | None = None,
    candidate_collision_free: Callable[[np.ndarray], bool] | None = None,
    transition_collision_free: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> PosePathResult:
    """Track a pose path using a rolling selection over complete IK branch proposals."""
    if branch_candidates < 2 or horizon < 2:
        raise ValueError("multibranch planning needs at least two branches and two horizon frames")
    targets = np.asarray(targets_xyz_m, dtype=float)
    quaternions = np.asarray(targets_quaternion_wxyz, dtype=float)
    times = np.asarray(time_s, dtype=float)
    if times.shape != (len(targets),) or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_s must be strictly increasing and match targets")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    limits = np.asarray([model.jnt_range[j] if model.jnt_limited[j] else (-np.pi, np.pi)
                         for j in joint_ids], dtype=float)
    lower, upper = limits[:, 0], limits[:, 1]
    midpoint = 0.5 * (lower + upper)
    periodic = joint_periodic_mask(model, joint_ids)
    rng = np.random.default_rng(20260811)
    seeds = [midpoint if initial_q is None else np.clip(np.asarray(initial_q, dtype=float), lower, upper)]
    seeds.extend(rng.uniform(lower, upper) for _ in range(branch_candidates - 1))
    if candidate_collision_free is not None:
        # A failed first-frame solve holds its frontier seed. Therefore every
        # seed admitted to the rolling frontier must itself pass the same hard
        # collision gate as an IK solution; otherwise a colliding mount can be
        # propagated through the whole episode as "safe hold" recovery.
        pool = seeds + [midpoint]
        pool.extend(rng.uniform(lower, upper)
                    for _ in range(max(64, branch_candidates * 32)))
        safe_seeds = []
        for seed in pool:
            if not bool(candidate_collision_free(seed)):
                continue
            if any(np.max(np.abs(wrapped_joint_delta(seed, kept, periodic))) < 1e-6
                   for kept in safe_seeds):
                continue
            safe_seeds.append(np.asarray(seed, dtype=float).copy())
            if len(safe_seeds) >= branch_candidates:
                break
        if not safe_seeds:
            raise RuntimeError("rolling IK found no collision-free initial seed")
        seeds = safe_seeds
    proposals = [
        solve_pose_path(
            model, site_name, joint_names, targets, quaternions,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            iterations=iterations, damping=damping, restarts=(12 if seed_index == 0 else 2),
            maximum_frame_jump_rad=maximum_frame_jump_rad,
            initial_q=seed, time_s=times,
            velocity_limit_rad_s=velocity_limit_rad_s,
        )
        for seed_index, seed in enumerate(seeds)
    ]
    layers: list[tuple[BranchCandidate, ...]] = []
    for frame_index in range(len(targets)):
        unique: list[BranchCandidate] = []
        for proposal_index, proposal in enumerate(proposals):
            q = proposal.q[frame_index]
            if any(np.max(np.abs(wrapped_joint_delta(q, item.q, periodic))) < 1e-5 for item in unique):
                continue
            unique.append(BranchCandidate(
                q=q.copy(),
                pose_valid=bool(proposal.position_error_m[frame_index] <= position_tolerance_m and
                                proposal.orientation_error_rad[frame_index] <= orientation_tolerance_rad),
                collision_free=(True if candidate_collision_free is None
                                else bool(candidate_collision_free(q))),
                position_error_m=float(proposal.position_error_m[frame_index]),
                orientation_error_rad=float(proposal.orientation_error_rad[frame_index]),
                joint_limit_margin=float(proposal.joint_limit_margin[frame_index]),
                singularity_margin=float(proposal.singularity_margin[frame_index]),
                index=proposal_index,
            ))
        layers.append(tuple(unique))
    intervals = np.r_[times[1] - times[0] if len(times) > 1 else 1.0, np.diff(times)]
    velocity_limits = np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float), midpoint.shape)
    previous = seeds[0].copy()
    q_path=[]; branch_count=[]; chosen=[]; recovery=[]; discontinuity=[]; velocity_violation=[]
    singularity=[]; limit_margin=[]; previous_velocity=np.zeros_like(previous); acceleration_warning=[]
    for frame_index in range(len(targets)):
        stop = min(len(targets), frame_index + horizon)
        selection = select_rolling_branch(
            layers=layers[frame_index:stop], initial_q=previous, periodic=periodic,
            dt_s=intervals[frame_index:stop], velocity_limit_rad_s=velocity_limits,
            cap_rad=np.full_like(previous, maximum_frame_jump_rad), beam_width=beam_width,
            transition_valid=transition_collision_free)
        candidates = layers[frame_index]
        # A blocked future layer must not discard a safe first transition from
        # the best partial horizon; rolling replanning can recover next frame.
        selected = selection.path[0] if selection.path else None
        mode = "none"; jump = False
        if selected is None:
            valid = [item for item in candidates if item.pose_valid and item.collision_free]
            if valid:
                selected = min(valid, key=lambda item: float(np.linalg.norm(
                    wrapped_joint_delta(item.q, previous, periodic))))
                allowed = float(np.min(np.minimum(velocity_limits * intervals[frame_index],
                                                  maximum_frame_jump_rad)))
                delta = wrapped_joint_delta(selected.q, previous, periodic)
                jump = bool(np.max(np.abs(delta)) > allowed)
                next_q = continuity_recovery_step(selected.q, previous, periodic, allowed) if jump else selected.q.copy()
                mode = "limited_step" if jump else "none"
            else:
                next_q = previous.copy(); mode = "hold"
        else:
            next_q = selected.q.copy()
        transition_safe = (transition_collision_free is None or
                           bool(transition_collision_free(previous, next_q)))
        state_safe = (candidate_collision_free is None or
                      bool(candidate_collision_free(next_q)))
        if not transition_safe or not state_safe:
            next_q = previous.copy()
            mode = "hold"
            jump = False
        delta = wrapped_joint_delta(next_q, previous, periodic)
        velocity = delta / intervals[frame_index]
        acceleration = (velocity - previous_velocity) / intervals[frame_index]
        q_path.append(next_q); branch_count.append(len(candidates))
        chosen.append(-1 if selected is None else selected.index); recovery.append(mode)
        discontinuity.append(jump)
        velocity_violation.append(realized_velocity_violation(
            delta, dt_s=intervals[frame_index],
            velocity_limit_rad_s=velocity_limits,
            initializing=frame_index == 0))
        acceleration_warning.append(bool(np.any(np.abs(acceleration) > velocity_limits / intervals[frame_index])))
        singularity.append(0.0 if selected is None else selected.singularity_margin)
        limit_margin.append(0.0 if selected is None else selected.joint_limit_margin)
        previous, previous_velocity = next_q, velocity
    q_array = np.asarray(q_path)
    data = mujoco.MjData(model); site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    addresses = [model.jnt_qposadr[j] for j in joint_ids]
    reached=[]; reached_q=[]; pe=[]; oe=[]
    for q, target, target_quat in zip(q_array, targets, quaternions):
        data.qpos[addresses] = q; mujoco.mj_forward(model, data)
        quat=np.zeros(4); mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        residual=np.zeros(3); mujoco.mju_subQuat(residual, target_quat, quat)
        reached.append(data.site_xpos[site_id].copy()); reached_q.append(quat)
        pe.append(float(np.linalg.norm(target-data.site_xpos[site_id]))); oe.append(float(np.linalg.norm(residual)))
    pe_array=np.asarray(pe); oe_array=np.asarray(oe)
    success=(pe_array <= position_tolerance_m) & (oe_array <= orientation_tolerance_rad) & ~np.asarray(discontinuity)
    return PosePathResult(
        q_array, np.asarray(reached), np.asarray(reached_q), pe_array, oe_array, success,
        np.asarray(discontinuity, bool),
        bool(np.all(q_array >= lower-1e-10) and np.all(q_array <= upper+1e-10)),
        np.asarray(velocity_violation, bool), np.asarray(acceleration_warning, bool),
        np.asarray(branch_count, int), np.asarray(chosen, int), np.asarray(recovery),
        np.asarray(singularity, float), np.asarray(limit_margin, float))


def solve_pose_path_multibranch(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: Sequence[str],
    targets_xyz_m: np.ndarray,
    targets_quaternion_wxyz: np.ndarray,
    *,
    time_s: np.ndarray,
    branch_candidates: int = 8,
    horizon: int = 12,
    beam_width: int = 8,
    velocity_limit_rad_s: float | np.ndarray = np.deg2rad(720.0),
    maximum_frame_jump_rad: float = np.deg2rad(25.0),
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = np.deg2rad(1.5),
    iterations: int = 240,
    damping: float = 5e-4,
    initial_q: np.ndarray | None = None,
    candidate_collision_free: Callable[[np.ndarray], bool] | None = None,
    transition_collision_free: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> PosePathResult:
    """Track poses with a bounded frontier of collision-safe IK branches."""
    del horizon  # The bounded frontier carries branch history across rolling updates.
    targets = np.asarray(targets_xyz_m, dtype=float)
    quaternions = np.asarray(targets_quaternion_wxyz, dtype=float)
    times = np.asarray(time_s, dtype=float)
    if branch_candidates < 2 or beam_width < 2:
        raise ValueError("multibranch planning needs at least two candidates and beam states")
    if targets.ndim != 2 or targets.shape[1] != 3 or quaternions.shape != (len(targets), 4):
        raise ValueError("target pose arrays have inconsistent shapes")
    if times.shape != (len(targets),) or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_s must be strictly increasing and match targets")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    limits = np.asarray([model.jnt_range[j] if model.jnt_limited[j] else (-np.pi, np.pi)
                         for j in joint_ids], dtype=float)
    lower, upper = limits[:, 0], limits[:, 1]
    midpoint = 0.5 * (lower + upper)
    periodic = joint_periodic_mask(model, joint_ids)
    velocity_limits = np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float), midpoint.shape)
    intervals = np.r_[times[1] - times[0] if len(times) > 1 else 1.0, np.diff(times)]
    rng = np.random.default_rng(20260811)
    seeds = [midpoint if initial_q is None else np.clip(np.asarray(initial_q, dtype=float), lower, upper)]
    seeds.extend(rng.uniform(lower, upper) for _ in range(branch_candidates - 1))
    if candidate_collision_free is not None:
        pool = seeds + [midpoint]
        pool.extend(rng.uniform(lower, upper)
                    for _ in range(max(64, branch_candidates * 32)))
        safe_seeds = []
        for seed in pool:
            if not bool(candidate_collision_free(seed)):
                continue
            if any(np.max(np.abs(wrapped_joint_delta(seed, kept, periodic))) < 1e-6
                   for kept in safe_seeds):
                continue
            safe_seeds.append(np.asarray(seed, dtype=float).copy())
            if len(safe_seeds) >= branch_candidates:
                break
        if not safe_seeds:
            raise RuntimeError("rolling IK found no collision-free initial seed")
        seeds = safe_seeds
    anchor = solve_pose_path(
        model, site_name, joint_names, targets, quaternions,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
        iterations=iterations, damping=damping, restarts=12,
        maximum_frame_jump_rad=maximum_frame_jump_rad,
        initial_q=seeds[0], time_s=times,
        velocity_limit_rad_s=velocity_limits)
    # Each state owns its realized history.  Keeping multiple prior states is
    # the essential difference from a single greedy previous-q seed.
    frontier = [{
        "q": seed.copy(), "velocity": np.zeros_like(midpoint),
        "cost": (0, 0.0, 0.0, 0.0, 0.0), "path": [], "modes": [],
        "jumps": [], "chosen": [], "singularity": [], "limit_margin": [],
        "acceleration": [],
    } for seed in seeds]
    per_frame_branch_count: list[int] = []

    for frame_index, (target, target_quat) in enumerate(zip(targets, quaternions)):
        expanded = []
        interval = float(intervals[frame_index])
        allowed = np.minimum(velocity_limits * interval, maximum_frame_jump_rad)
        for state_index, state in enumerate(frontier):
            proposal = solve_pose_path(
                model, site_name, joint_names, target[None, :], target_quat[None, :],
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                iterations=iterations, damping=damping,
                restarts=(12 if state_index == 0 else 4),
                maximum_frame_jump_rad=maximum_frame_jump_rad,
                initial_q=state["q"], velocity_limit_rad_s=velocity_limits,
                rng_seed=20260811 + frame_index * beam_width + state_index,
                hold_invalid=False)
            candidate_q = proposal.q[0]
            proposal_pe = float(proposal.position_error_m[0])
            proposal_oe = float(proposal.orientation_error_rad[0])
            proposal_limit_margin = float(proposal.joint_limit_margin[0])
            proposal_singularity = float(proposal.singularity_margin[0])
            pose_valid = bool(proposal_pe <= position_tolerance_m and
                              proposal_oe <= orientation_tolerance_rad)
            delta = wrapped_joint_delta(candidate_q, state["q"], periodic)
            speed_valid = frame_index == 0 or bool(np.all(np.abs(delta) <= allowed + 1e-12))
            state_safe = candidate_collision_free is None or bool(candidate_collision_free(candidate_q))
            edge_safe = (frame_index == 0 or transition_collision_free is None or
                         bool(transition_collision_free(state["q"], candidate_q)))
            mode = "none"
            jump = bool(pose_valid and not speed_valid)
            next_q = candidate_q.copy()
            frame_success = pose_valid and speed_valid and state_safe and edge_safe
            if not frame_success:
                if pose_valid and jump:
                    next_q = continuity_recovery_step(candidate_q, state["q"], periodic,
                                                      allowed)
                    recovery_safe = (candidate_collision_free is None or
                                     bool(candidate_collision_free(next_q)))
                    recovery_edge_safe = (transition_collision_free is None or
                                          bool(transition_collision_free(state["q"], next_q)))
                    if recovery_safe and recovery_edge_safe:
                        mode = "limited_step"
                    else:
                        next_q = state["q"].copy(); mode = "hold"; jump = False
                else:
                    # An invalid pose proposal is not a valid recovery target.
                    # Walking toward it can move the frontier out of a nearby
                    # feasible IK basin and create a long cascade of failures.
                    next_q = state["q"].copy(); mode = "hold"; jump = False
            realized_delta = wrapped_joint_delta(next_q, state["q"], periodic)
            velocity = realized_delta / interval
            acceleration = (velocity - state["velocity"]) / interval
            failure_increment = 0 if frame_success else 1
            cost = (
                state["cost"][0] + failure_increment,
                state["cost"][1] + float(np.linalg.norm(acceleration)),
                state["cost"][2] - proposal_limit_margin,
                state["cost"][3] - proposal_singularity,
                state["cost"][4] + proposal_pe + 0.1 * proposal_oe,
            )
            expanded.append({
                "q": next_q, "velocity": velocity, "cost": cost,
                "path": state["path"] + [next_q.copy()],
                "modes": state["modes"] + [mode],
                "jumps": state["jumps"] + [jump],
                "chosen": state["chosen"] + [state_index],
                "singularity": state["singularity"] + [proposal_singularity],
                "limit_margin": state["limit_margin"] + [proposal_limit_margin],
                "acceleration": state["acceleration"] + [bool(np.any(
                    np.abs(acceleration) > velocity_limits / interval))],
            })
        # Deduplicate equivalent states while preserving the better history.
        expanded.sort(key=lambda item: item["cost"])
        unique = []
        for state in expanded:
            if any(np.max(np.abs(wrapped_joint_delta(state["q"], kept["q"], periodic))) < np.deg2rad(1.0)
                   for kept in unique):
                continue
            unique.append(state)
            if len(unique) >= beam_width:
                break
        if not unique:
            raise RuntimeError(f"rolling IK frontier exhausted at frame {frame_index}")
        frontier = unique
        per_frame_branch_count.append(len(unique))

    best = min(frontier, key=lambda item: item["cost"])
    q_array = np.asarray(best["path"])
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    addresses = [model.jnt_qposadr[j] for j in joint_ids]
    reached=[]; reached_q=[]; pe=[]; oe=[]
    for q, target, target_quat in zip(q_array, targets, quaternions):
        data.qpos[addresses] = q; mujoco.mj_forward(model, data)
        quat=np.zeros(4); mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        residual=np.zeros(3); mujoco.mju_subQuat(residual, target_quat, quat)
        reached.append(data.site_xpos[site_id].copy()); reached_q.append(quat)
        pe.append(float(np.linalg.norm(target-data.site_xpos[site_id]))); oe.append(float(np.linalg.norm(residual)))
    pe_array=np.asarray(pe); oe_array=np.asarray(oe); jumps=np.asarray(best["jumps"], bool)
    success=(pe_array <= position_tolerance_m) & (oe_array <= orientation_tolerance_rad) & ~jumps
    realized_velocity = np.zeros(len(q_array), dtype=bool)
    for frame_index in range(1, len(q_array)):
        realized_velocity[frame_index] = realized_velocity_violation(
            wrapped_joint_delta(q_array[frame_index], q_array[frame_index - 1], periodic),
            dt_s=intervals[frame_index], velocity_limit_rad_s=velocity_limits)
    multibranch_result = PosePathResult(
        q_array, np.asarray(reached), np.asarray(reached_q), pe_array, oe_array, success, jumps,
        bool(np.all(q_array >= lower-1e-10) and np.all(q_array <= upper+1e-10)),
        realized_velocity, np.asarray(best["acceleration"], bool), np.asarray(per_frame_branch_count, int),
        np.asarray(best["chosen"], int), np.asarray(best["modes"]),
        np.asarray(best["singularity"], float), np.asarray(best["limit_margin"], float))

    # The original high-restart continuous solver is retained as a portfolio
    # floor. Apply the exact same state/edge safety policy before comparing it
    # with the frontier result, so the planner cannot regress a robot/task.
    anchor_path=[]; anchor_modes=[]; anchor_speed=[]; previous=seeds[0].copy()
    for frame_index, candidate_q in enumerate(anchor.q):
        delta = wrapped_joint_delta(candidate_q, previous, periodic)
        allowed = np.minimum(velocity_limits * intervals[frame_index], maximum_frame_jump_rad)
        speed_safe = frame_index == 0 or bool(np.all(np.abs(delta) <= allowed + 1e-12))
        state_safe = candidate_collision_free is None or bool(candidate_collision_free(candidate_q))
        edge_safe = (frame_index == 0 or transition_collision_free is None or
                     bool(transition_collision_free(previous, candidate_q)))
        if speed_safe and state_safe and edge_safe:
            next_q=candidate_q.copy(); mode="none"
        else:
            next_q=previous.copy(); mode="hold"
        anchor_path.append(next_q); anchor_modes.append(mode); anchor_speed.append(speed_safe)
        previous=next_q
    anchor_q=np.asarray(anchor_path); anchor_reached=[]; anchor_reached_q=[]; anchor_pe=[]; anchor_oe=[]
    for q, target, target_quat in zip(anchor_q, targets, quaternions):
        data.qpos[addresses]=q; mujoco.mj_forward(model,data)
        quat=np.zeros(4);mujoco.mju_mat2Quat(quat,data.site_xmat[site_id])
        residual=np.zeros(3);mujoco.mju_subQuat(residual,target_quat,quat)
        anchor_reached.append(data.site_xpos[site_id].copy());anchor_reached_q.append(quat)
        anchor_pe.append(float(np.linalg.norm(target-data.site_xpos[site_id])));anchor_oe.append(float(np.linalg.norm(residual)))
    anchor_pe=np.asarray(anchor_pe);anchor_oe=np.asarray(anchor_oe)
    anchor_jumps=np.asarray(anchor.joint_discontinuity,bool) | ~np.asarray(anchor_speed,bool)
    anchor_success=(anchor_pe<=position_tolerance_m)&(anchor_oe<=orientation_tolerance_rad)&~anchor_jumps
    anchor_velocity=np.zeros(len(anchor_q),dtype=bool)
    for frame_index in range(1,len(anchor_q)):
        anchor_velocity[frame_index]=realized_velocity_violation(
            wrapped_joint_delta(anchor_q[frame_index],anchor_q[frame_index-1],periodic),
            dt_s=intervals[frame_index],velocity_limit_rad_s=velocity_limits)
    anchor_result=PosePathResult(
        anchor_q,np.asarray(anchor_reached),np.asarray(anchor_reached_q),anchor_pe,anchor_oe,
        anchor_success,anchor_jumps,
        bool(np.all(anchor_q>=lower-1e-10) and np.all(anchor_q<=upper+1e-10)),
        anchor_velocity,anchor.acceleration_warning,np.ones(len(targets),int),
        np.zeros(len(targets),int),np.asarray(anchor_modes),anchor.singularity_margin,
        anchor.joint_limit_margin)
    frontier_rank=(int(multibranch_result.success.sum()),-int(multibranch_result.joint_discontinuity.sum()))
    anchor_rank=(int(anchor_result.success.sum()),-int(anchor_result.joint_discontinuity.sum()))
    return multibranch_result if frontier_rank>=anchor_rank else anchor_result


def solve_position_path(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: Sequence[str],
    targets_xyz_m: np.ndarray,
    *,
    tolerance_m: float = 1e-4,
    iterations: int = 180,
    damping: float = 2e-4,
    restarts: int = 8,
) -> PositionPathResult:
    targets = np.asarray(targets_xyz_m, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets_xyz_m must have shape (frames, 3)")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"site not found: {site_name}")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if any(joint_id < 0 for joint_id in joint_ids):
        missing = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
        raise ValueError(f"joints not found: {missing}")
    q_addresses = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
    dof_addresses = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=int)
    limits = np.asarray([
        model.jnt_range[joint_id] if model.jnt_limited[joint_id] else (-np.pi, np.pi)
        for joint_id in joint_ids
    ], dtype=float)
    lower, upper = limits[:, 0], limits[:, 1]
    midpoint = 0.5 * (lower + upper)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(20260807)
    solved_q, reached, errors, successes = [], [], [], []
    previous = midpoint.copy()
    jac_position = np.zeros((3, model.nv))
    jac_rotation = np.zeros((3, model.nv))
    for target in targets:
        seeds = [previous, midpoint]
        seeds.extend(rng.uniform(lower, upper) for _ in range(max(0, restarts - 2)))
        best_error = np.inf
        best_q = previous.copy()
        best_reached = np.full(3, np.nan)
        for seed in seeds:
            q = np.clip(np.asarray(seed, dtype=float).copy(), lower, upper)
            for _ in range(iterations):
                data.qpos[q_addresses] = q
                mujoco.mj_forward(model, data)
                current = data.site_xpos[site_id].copy()
                residual = target - current
                error = float(np.linalg.norm(residual))
                if error <= tolerance_m:
                    break
                mujoco.mj_jacSite(model, data, jac_position, jac_rotation, site_id)
                jacobian = jac_position[:, dof_addresses]
                system = jacobian @ jacobian.T + damping * np.eye(3)
                delta = jacobian.T @ np.linalg.solve(system, residual)
                norm = float(np.linalg.norm(delta))
                if norm > 0.18:
                    delta *= 0.18 / norm
                q = np.clip(q + delta, lower, upper)
            data.qpos[q_addresses] = q
            mujoco.mj_forward(model, data)
            current = data.site_xpos[site_id].copy()
            error = float(np.linalg.norm(target - current))
            if error < best_error:
                best_error, best_q, best_reached = error, q.copy(), current
            if best_error <= tolerance_m:
                break
        previous = best_q
        solved_q.append(best_q)
        reached.append(best_reached)
        errors.append(best_error)
        successes.append(best_error <= tolerance_m)
    q_array = np.asarray(solved_q)
    respected = bool(np.all(q_array >= lower - 1e-10) and np.all(q_array <= upper + 1e-10))
    return PositionPathResult(
        q=q_array,
        reached_xyz_m=np.asarray(reached),
        position_error_m=np.asarray(errors),
        success=np.asarray(successes, dtype=bool),
        joint_limits_respected=respected,
    )
