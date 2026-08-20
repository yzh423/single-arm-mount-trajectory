"""Bounded rolling-horizon selection over precomputed IK branches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class BranchCandidate:
    q: np.ndarray
    pose_valid: bool
    collision_free: bool
    position_error_m: float
    orientation_error_rad: float
    joint_limit_margin: float
    singularity_margin: float
    index: int


@dataclass(frozen=True)
class BranchSelection:
    path: tuple[BranchCandidate, ...]
    complete: bool
    cost: tuple[float, ...]


@dataclass(frozen=True)
class RecedingHorizonPath:
    q: np.ndarray
    selected_indices: np.ndarray
    pose_valid: np.ndarray
    recovery_mode: np.ndarray


@dataclass(frozen=True)
class RetimedPath:
    q: np.ndarray
    selected_indices: np.ndarray
    pose_valid: np.ndarray
    required_dt_s: np.ndarray
    recovery_mode: np.ndarray


def build_collision_free_pair_layers(
    *, left_layers: Sequence[Sequence[BranchCandidate]],
    right_layers: Sequence[Sequence[BranchCandidate]],
    state_valid: Callable[[np.ndarray, np.ndarray], bool],
    maximum_pairs_per_layer: int = 24,
) -> tuple[tuple[BranchCandidate, ...], ...]:
    """Build bounded bimanual states without changing either TCP target."""
    if len(left_layers) != len(right_layers):
        raise ValueError("left and right layers must have the same length")
    if maximum_pairs_per_layer < 1:
        raise ValueError("maximum_pairs_per_layer must be positive")
    paired = []
    for left, right in zip(left_layers, right_layers):
        candidates = []
        for left_item in left:
            if not left_item.pose_valid or not left_item.collision_free:
                continue
            for right_item in right:
                if not right_item.pose_valid or not right_item.collision_free:
                    continue
                if not state_valid(left_item.q, right_item.q):
                    continue
                q = np.r_[left_item.q, right_item.q]
                candidates.append(BranchCandidate(
                    q=q, pose_valid=True, collision_free=True,
                    position_error_m=max(left_item.position_error_m,
                                         right_item.position_error_m),
                    orientation_error_rad=max(left_item.orientation_error_rad,
                                              right_item.orientation_error_rad),
                    joint_limit_margin=min(left_item.joint_limit_margin,
                                           right_item.joint_limit_margin),
                    singularity_margin=min(left_item.singularity_margin,
                                           right_item.singularity_margin),
                    index=(int(left_item.index) << 16) | int(right_item.index),
                ))
        candidates.sort(key=lambda item: (
            1.0 / max(item.joint_limit_margin, 1e-9)
            + .1 / max(item.singularity_margin, 1e-9),
            item.position_error_m + .1 * item.orientation_error_rad,
            item.index,
        ))
        paired.append(tuple(candidates[:maximum_pairs_per_layer]))
    return tuple(paired)


def select_minimum_retime_path(
    *, layers: Sequence[Sequence[BranchCandidate]], initial_q: np.ndarray,
    periodic: np.ndarray, dt_s: np.ndarray,
    velocity_limit_rad_s: np.ndarray, safety_fraction: float = .8,
    minimum_joint_limit_margin_rad: float = 0.0,
    minimum_singularity_margin: float = 0.0,
    maximum_recovery_wrist_distance_rad: float = np.inf,
    maximum_joint_step_rad: float = np.inf,
    maximum_wrist_step_norm_rad: float = np.inf,
    transition_valid: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> RetimedPath:
    """Globally select safe IK branches while minimally stretching timestamps.

    Exact-pose layers are connected even when the source timestamp is too short;
    the required edge duration is recorded instead of replacing the target by a
    bounded recovery step. Empty collision-free layers remain explicit holds and
    split the graph into independently optimized exact-pose segments.
    """
    initial = np.asarray(initial_q, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    dt = np.asarray(dt_s, dtype=float)
    velocity = np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float), initial.shape)
    if dt.shape != (len(layers),) or periodic.shape != initial.shape:
        raise ValueError("planner inputs have inconsistent shapes")
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0) or not 0 < safety_fraction <= 1:
        raise ValueError("timestamps and safety_fraction must be positive")
    if maximum_joint_step_rad <= 0 or maximum_wrist_step_norm_rad <= 0:
        raise ValueError("visual branch step limits must be positive")

    def branch_step_valid(left, right):
        delta = np.abs(_wrapped_delta(right, left, periodic))
        wrist = delta[-3:] if len(delta) >= 3 else delta
        return (float(np.max(delta)) <= maximum_joint_step_rad + 1e-12 and
                float(np.linalg.norm(wrist)) <= maximum_wrist_step_norm_rad + 1e-12)
    safe_velocity = velocity * float(safety_fraction)
    collision_safe = [tuple(c for c in layer if c.pose_valid and c.collision_free)
                      for layer in layers]
    valid = [tuple(c for c in layer
                   if c.joint_limit_margin >= minimum_joint_limit_margin_rad
                   and c.singularity_margin >= minimum_singularity_margin)
             for layer in collision_safe]
    q_out = np.repeat(initial[None, :], len(layers), axis=0)
    chosen = np.full(len(layers), -1, dtype=int)
    pose = np.zeros(len(layers), dtype=bool)
    required = dt.copy()
    modes = np.full(len(layers), "hold_no_candidate", dtype=object)
    previous = initial.copy()
    row = 0
    while row < len(valid):
        if not valid[row]:
            q_out[row] = previous
            if collision_safe[row]:
                modes[row] = "hold_low_quality_candidate"
            row += 1
            continue
        stop = row
        while stop < len(valid) and valid[stop]:
            stop += 1
        # state = (cost, candidate, parent, edge_dt, edge_velocity)
        frontier = []
        for j, item in enumerate(valid[row]):
            q = np.asarray(item.q, float)
            if row == 0:
                edge_dt = dt[row]
            else:
                delta = np.abs(_wrapped_delta(q, previous, periodic))
                if not branch_step_valid(previous, q):
                    continue
                edge_dt = max(float(dt[row]), float(np.max(delta / safe_velocity)))
                if transition_valid is not None and not transition_valid(previous, q):
                    continue
            risk = 1.0 / max(float(item.joint_limit_margin), 1e-9)
            risk += .1 / max(float(item.singularity_margin), 1e-9)
            velocity_now = _wrapped_delta(q, previous, periodic) / edge_dt
            frontier.append(((edge_dt-dt[row], 0.0, risk, 0.0), j, None,
                             edge_dt, velocity_now))
        if not frontier and valid[row]:
            # A preceding empty layer may leave the realized robot too far from
            # every exact-pose branch.  Advance toward the nearest safe branch
            # without teleporting; the frame remains explicitly pose-invalid.
            targets = sorted(valid[row], key=lambda item: float(np.linalg.norm(
                _wrapped_delta(item.q, previous, periodic))))
            recovered = None
            for item in targets:
                delta = _wrapped_delta(item.q, previous, periodic)
                step = np.clip(delta, -maximum_joint_step_rad,
                               maximum_joint_step_rad)
                wrist = step[-3:] if len(step) >= 3 else step
                wrist_norm = float(np.linalg.norm(wrist))
                if wrist_norm > maximum_wrist_step_norm_rad:
                    step[-len(wrist):] *= maximum_wrist_step_norm_rad / wrist_norm
                trial = previous + step
                if transition_valid is None or transition_valid(previous, trial):
                    recovered = trial
                    break
            if recovered is not None:
                q_out[row] = recovered
                required[row] = max(float(dt[row]), float(np.max(
                    np.abs(_wrapped_delta(recovered, previous, periodic)) / safe_velocity)))
                modes[row] = "limited_step"
                previous = recovered
                row += 1
                continue
        histories = [frontier]
        for current_row in range(row + 1, stop):
            next_frontier = []
            for j, item in enumerate(valid[current_row]):
                q = np.asarray(item.q, float)
                best = None
                for parent_index, state in enumerate(frontier):
                    old = np.asarray(valid[current_row-1][state[1]].q, float)
                    if not branch_step_valid(old, q):
                        continue
                    if transition_valid is not None and not transition_valid(old, q):
                        continue
                    delta = np.abs(_wrapped_delta(q, old, periodic))
                    edge_dt = max(float(dt[current_row]), float(np.max(delta / safe_velocity)))
                    edge_velocity = _wrapped_delta(q, old, periodic) / edge_dt
                    velocity_variation = float(np.linalg.norm(edge_velocity - state[4]))
                    risk = 1.0 / max(float(item.joint_limit_margin), 1e-9)
                    risk += .1 / max(float(item.singularity_margin), 1e-9)
                    cost = (state[0][0] + edge_dt-dt[current_row],
                            state[0][1] + velocity_variation,
                            state[0][2] + risk,
                            state[0][3] + float(np.linalg.norm(delta)))
                    candidate_state = (cost, j, parent_index, edge_dt,
                                       edge_velocity)
                    if best is None or candidate_state[0] < best[0]:
                        best = candidate_state
                if best is not None:
                    next_frontier.append(best)
            if not next_frontier:
                break
            frontier = next_frontier
            histories.append(frontier)
        if not frontier:
            q_out[row] = previous; row += 1
            continue
        end_offset = len(histories) - 1
        state_index = min(range(len(frontier)), key=lambda k: frontier[k][0])
        trace = []
        for offset in range(end_offset, -1, -1):
            state = histories[offset][state_index]
            trace.append(state)
            state_index = state[2] if state[2] is not None else 0
        trace.reverse()
        for offset, state in enumerate(trace):
            out_row = row + offset
            item = valid[out_row][state[1]]
            previous = np.asarray(item.q, float).copy()
            q_out[out_row] = previous; chosen[out_row] = int(item.index)
            pose[out_row] = True; required[out_row] = float(state[3])
            modes[out_row] = "initialize" if out_row == 0 else (
                "retimed" if required[out_row] > dt[out_row] + 1e-12 else "none")
        row += len(trace)
    return RetimedPath(q_out, chosen, pose, required, modes)


def select_global_feasible_path(
    *, layers: Sequence[Sequence[BranchCandidate]], initial_q: np.ndarray,
    periodic: np.ndarray, dt_s: np.ndarray,
    velocity_limit_rad_s: np.ndarray, cap_rad: np.ndarray,
    transition_valid: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> RecedingHorizonPath:
    """Select complete feasible segments without finite-horizon branch myopia.

    Candidate layers are small, so exact dynamic programming over every valid
    inter-layer edge is cheaper and safer than pruning histories with a beam.
    Empty/disconnected layers split the episode; such gaps are recovered with
    the same bounded realized step used by the online selector.
    """
    initial = np.asarray(initial_q, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    intervals = np.asarray(dt_s, dtype=float)
    velocity = np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float), initial.shape)
    cap = np.broadcast_to(np.asarray(cap_rad, dtype=float), initial.shape)
    if intervals.shape != (len(layers),) or periodic.shape != initial.shape:
        raise ValueError("planner inputs have inconsistent shapes")

    valid_layers = [tuple(item for item in layer
                          if item.pose_valid and item.collision_free)
                    for layer in layers]
    # Backward feasibility marks nodes that can reach the end of their current
    # connected segment.  It prevents committing to a smooth-looking dead end.
    can_reach = [np.zeros(len(layer), dtype=bool) for layer in valid_layers]
    if valid_layers:
        can_reach[-1][:] = True
    for row in range(len(valid_layers) - 2, -1, -1):
        if not valid_layers[row] or not valid_layers[row + 1]:
            can_reach[row][:] = True
            continue
        limits = np.minimum(velocity * intervals[row + 1], cap)
        for index, item in enumerate(valid_layers[row]):
            can_reach[row][index] = any(
                can_reach[row + 1][next_index]
                and np.all(np.abs(_wrapped_delta(next_item.q, item.q, periodic))
                           <= limits + 1e-12)
                and (transition_valid is None or
                     transition_valid(np.asarray(item.q), np.asarray(next_item.q)))
                for next_index, next_item in enumerate(valid_layers[row + 1]))
    previous = initial.copy(); q_path=[]; chosen=[]; pose=[]; modes=[]
    for row, layer in enumerate(valid_layers):
        feasible = [item for index, item in enumerate(layer) if can_reach[row][index]]
        if row == 0 and feasible:
            selected = min(feasible, key=lambda item: (
                float(item.position_error_m) + .1 * float(item.orientation_error_rad),
                int(item.index)))
            previous = np.asarray(selected.q).copy(); q_path.append(previous.copy())
            chosen.append(int(selected.index)); pose.append(True); modes.append("initialize")
            continue
        limits = np.minimum(velocity * intervals[row], cap)
        connected = [item for item in feasible
                     if np.all(np.abs(_wrapped_delta(item.q, previous, periodic))
                               <= limits + 1e-12)
                     and (transition_valid is None or
                          transition_valid(previous, np.asarray(item.q)))]
        if connected:
            selected = min(connected, key=lambda item: (
                float(np.linalg.norm(_wrapped_delta(item.q, previous, periodic))),
                float(item.position_error_m) + .1 * float(item.orientation_error_rad),
                int(item.index)))
            previous = np.asarray(selected.q).copy(); mode="none"; valid=True
        elif layer:
            selected = min(layer, key=lambda item: float(np.linalg.norm(
                _wrapped_delta(item.q, previous, periodic))))
            delta = _wrapped_delta(selected.q, previous, periodic)
            previous = previous + np.clip(delta, -limits, limits)
            mode="limited_step"; valid=False
        else:
            selected=None; mode="hold_no_candidate"; valid=False
        q_path.append(previous.copy()); chosen.append(-1 if selected is None else int(selected.index))
        pose.append(valid); modes.append(mode)
    return RecedingHorizonPath(np.asarray(q_path), np.asarray(chosen, int),
                               np.asarray(pose, bool), np.asarray(modes))


def transition_limit_rad(dt_s: float, velocity_limit_rad_s: float,
                         cap_rad: float) -> float:
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if velocity_limit_rad_s <= 0.0 or cap_rad <= 0.0:
        raise ValueError("velocity limit and cap must be positive")
    return float(min(velocity_limit_rad_s * dt_s, cap_rad))


def _wrapped_delta(candidate: np.ndarray, previous: np.ndarray,
                   periodic: np.ndarray) -> np.ndarray:
    delta = np.asarray(candidate, dtype=float) - np.asarray(previous, dtype=float)
    delta = delta.copy()
    delta[periodic] = (delta[periodic] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def select_rolling_branch(
    *,
    layers: Sequence[Sequence[BranchCandidate]],
    initial_q: np.ndarray,
    periodic: np.ndarray,
    dt_s: np.ndarray,
    velocity_limit_rad_s: np.ndarray,
    cap_rad: np.ndarray,
    beam_width: int = 8,
    transition_valid: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    initial_velocity_rad_s: np.ndarray | None = None,
    initial_acceleration_rad_s2: np.ndarray | None = None,
    acceleration_limit_rad_s2: np.ndarray | float = np.inf,
    jerk_limit_rad_s3: np.ndarray | float = np.inf,
) -> BranchSelection:
    """Return the best complete feasible path through layered IK candidates."""
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    initial = np.asarray(initial_q, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    dt = np.asarray(dt_s, dtype=float)
    velocity_limit = np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float), initial.shape)
    cap = np.broadcast_to(np.asarray(cap_rad, dtype=float), initial.shape)
    initial_velocity = (np.zeros_like(initial) if initial_velocity_rad_s is None
                        else np.asarray(initial_velocity_rad_s, dtype=float))
    initial_acceleration = (
        np.zeros_like(initial) if initial_acceleration_rad_s2 is None
        else np.asarray(initial_acceleration_rad_s2, dtype=float)
    )
    acceleration_limit = np.broadcast_to(
        np.asarray(acceleration_limit_rad_s2, dtype=float), initial.shape)
    jerk_limit = np.broadcast_to(
        np.asarray(jerk_limit_rad_s3, dtype=float), initial.shape)
    if periodic.shape != initial.shape or dt.shape != (len(layers),):
        raise ValueError("planner inputs have inconsistent shapes")
    if initial_velocity.shape != initial.shape or np.any(~np.isfinite(initial_velocity)):
        raise ValueError("initial_velocity_rad_s must match the joint shape and be finite")
    if (initial_acceleration.shape != initial.shape
            or np.any(~np.isfinite(initial_acceleration))):
        raise ValueError("initial_acceleration_rad_s2 must match the joint shape and be finite")
    if np.any(acceleration_limit <= 0.0) or np.any(jerk_limit <= 0.0):
        raise ValueError("acceleration and jerk limits must be positive")
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("all frame intervals must be finite and positive")

    # state: (lexicographic cost, path, last q, last velocity, last acceleration)
    frontier = [((0.0, 0.0, 0.0, 0.0, 0.0), tuple(), initial,
                 initial_velocity, initial_acceleration)]
    deepest = frontier
    for layer_index, layer in enumerate(layers):
        next_frontier = []
        interval = float(dt[layer_index])
        limits = np.minimum(velocity_limit * interval, cap)
        for cost, path, previous_q, previous_velocity, previous_acceleration in frontier:
            for item in layer:
                q = np.asarray(item.q, dtype=float)
                if q.shape != initial.shape or not np.all(np.isfinite(q)):
                    continue
                if not item.pose_valid or not item.collision_free:
                    continue
                delta = _wrapped_delta(q, previous_q, periodic)
                if np.any(np.abs(delta) > limits + 1e-12):
                    continue
                if transition_valid is not None and not transition_valid(previous_q, q):
                    continue
                velocity = delta / interval
                acceleration_vector = (velocity - previous_velocity) / interval
                if np.any(np.abs(acceleration_vector) > acceleration_limit + 1e-12):
                    continue
                jerk_vector = (acceleration_vector - previous_acceleration) / interval
                if np.any(np.abs(jerk_vector) > jerk_limit + 1e-12):
                    continue
                acceleration = float(np.linalg.norm(acceleration_vector))
                addition = (
                    acceleration,
                    -float(item.joint_limit_margin),
                    -float(item.singularity_margin),
                    float(item.position_error_m) + 0.1 * float(item.orientation_error_rad),
                    float(np.linalg.norm(delta)),
                )
                total = tuple(left + right for left, right in zip(cost, addition))
                next_frontier.append((total, path + (item,), q, velocity,
                                      acceleration_vector))
        if not next_frontier:
            return BranchSelection(path=deepest[0][1] if deepest else tuple(), complete=False,
                                   cost=deepest[0][0] if deepest else (float("inf"),) * 5)
        next_frontier.sort(key=lambda state: (state[0], tuple(item.index for item in state[1])))
        frontier = next_frontier[:beam_width]
        deepest = frontier
    best = frontier[0]
    return BranchSelection(path=best[1], complete=True, cost=best[0])


def select_receding_horizon_path(
    *,
    layers: Sequence[Sequence[BranchCandidate]],
    initial_q: np.ndarray,
    periodic: np.ndarray,
    dt_s: np.ndarray,
    velocity_limit_rad_s: np.ndarray,
    cap_rad: np.ndarray,
    horizon: int = 12,
    beam_width: int = 8,
    reseed_after_empty_frames: int | None = None,
    transition_valid: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    minimum_joint_limit_margin_rad: float = 0.0,
    minimum_singularity_margin: float = 0.0,
    maximum_recovery_wrist_distance_rad: float = np.inf,
    initial_soft_joint_limit_margin_rad: float = 0.0,
    acceleration_limit_rad_s2: np.ndarray | float = np.inf,
    jerk_limit_rad_s3: np.ndarray | float = np.inf,
) -> RecedingHorizonPath:
    """Commit one feasible edge per rolling window and restart after gaps."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if reseed_after_empty_frames is not None and reseed_after_empty_frames < 1:
        raise ValueError("reseed_after_empty_frames must be positive")
    previous = np.asarray(initial_q, dtype=float).copy()
    previous_velocity = np.zeros_like(previous)
    previous_acceleration = np.zeros_like(previous)
    acceleration_limit = np.broadcast_to(
        np.asarray(acceleration_limit_rad_s2, dtype=float), previous.shape)
    jerk_limit = np.broadcast_to(
        np.asarray(jerk_limit_rad_s3, dtype=float), previous.shape)
    intervals = np.asarray(dt_s, dtype=float)
    if intervals.shape != (len(layers),):
        raise ValueError("dt_s must contain one interval per candidate layer")
    q_path = []
    selected_indices = []
    pose_valid = []
    recovery_mode = []
    consecutive_empty = 0
    filtered_layers = [tuple(item for item in future_layer
                       if item.pose_valid and item.collision_free
                       and item.joint_limit_margin >= minimum_joint_limit_margin_rad
                       and item.singularity_margin >= minimum_singularity_margin)
                       for future_layer in layers]
    for frame_index, layer in enumerate(layers):
        pose_valid_layer = [item for item in layer if item.pose_valid]
        collision_safe_layer = [item for item in layer
                                if item.pose_valid and item.collision_free]
        valid_layer = [item for item in collision_safe_layer
                       if item.joint_limit_margin >= minimum_joint_limit_margin_rad
                       and item.singularity_margin >= minimum_singularity_margin]
        if frame_index == 0 and valid_layer:
            stop = min(len(layers), horizon)
            ranked = []
            for item in valid_layer:
                source_initial_velocity = np.zeros_like(previous)
                if stop > 1 and filtered_layers[1]:
                    source_initial_velocity = min(
                        (_wrapped_delta(next_item.q, item.q, periodic)
                         / intervals[1] for next_item in filtered_layers[1]),
                        key=lambda value: float(np.linalg.norm(value)),
                    )
                    source_initial_velocity = np.clip(
                        source_initial_velocity,
                        -np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float),
                                         previous.shape),
                        np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float),
                                        previous.shape),
                    )
                future = select_rolling_branch(
                    layers=filtered_layers[1:stop], initial_q=np.asarray(item.q, dtype=float),
                    periodic=periodic, dt_s=intervals[1:stop],
                    velocity_limit_rad_s=velocity_limit_rad_s, cap_rad=cap_rad,
                    beam_width=beam_width, transition_valid=transition_valid,
                    initial_velocity_rad_s=source_initial_velocity,
                    initial_acceleration_rad_s2=np.zeros_like(previous),
                    acceleration_limit_rad_s2=acceleration_limit,
                    jerk_limit_rad_s3=jerk_limit,
                ) if stop > 1 else BranchSelection(tuple(), True, (0.0,) * 5)
                ranked.append((
                    0 if future.complete else 1, -len(future.path),
                    0 if item.joint_limit_margin >= initial_soft_joint_limit_margin_rad else 1,
                    future.cost,
                    -float(item.joint_limit_margin), -float(item.singularity_margin),
                    float(item.position_error_m) + .1 * float(item.orientation_error_rad),
                    int(item.index), item, source_initial_velocity,
                ))
            chosen = min(ranked, key=lambda entry: entry[:-2])
            selected, source_initial_velocity = chosen[-2:]
            previous_velocity = source_initial_velocity.copy()
            previous_acceleration = np.zeros_like(previous)
            previous = np.asarray(selected.q, dtype=float).copy()
            q_path.append(previous.copy())
            selected_indices.append(int(selected.index))
            pose_valid.append(True)
            recovery_mode.append("initialize")
            consecutive_empty = 0
            continue
        if (valid_layer and reseed_after_empty_frames is not None and
                consecutive_empty >= reseed_after_empty_frames):
            selected = min(valid_layer, key=lambda item: (
                float(item.position_error_m) + .1 * float(item.orientation_error_rad),
                int(item.index)))
            previous = np.asarray(selected.q, dtype=float).copy()
            previous_velocity = np.zeros_like(previous)
            previous_acceleration = np.zeros_like(previous)
            q_path.append(previous.copy()); selected_indices.append(int(selected.index))
            pose_valid.append(True); recovery_mode.append("reinitialize_after_gap")
            consecutive_empty = 0
            continue
        stop = min(len(layers), frame_index + horizon)
        selection = select_rolling_branch(
            layers=filtered_layers[frame_index:stop], initial_q=previous,
            periodic=periodic, dt_s=intervals[frame_index:stop],
            velocity_limit_rad_s=velocity_limit_rad_s, cap_rad=cap_rad,
            beam_width=beam_width, transition_valid=transition_valid,
            initial_velocity_rad_s=previous_velocity,
            initial_acceleration_rad_s2=previous_acceleration,
            acceleration_limit_rad_s2=acceleration_limit,
            jerk_limit_rad_s3=jerk_limit,
        )
        selected = selection.path[0] if selection.path else None
        if selected is None:
            if valid_layer:
                selected = min(valid_layer, key=lambda item: float(np.linalg.norm(
                    _wrapped_delta(item.q, previous, periodic))))
                target_delta = _wrapped_delta(selected.q, previous, periodic)
                wrist_distance = float(np.linalg.norm(target_delta[-3:]))
                if wrist_distance > maximum_recovery_wrist_distance_rad:
                    q_path.append(previous.copy())
                    selected_indices.append(int(selected.index))
                    pose_valid.append(False)
                    recovery_mode.append("hold_wrist_branch_switch")
                    previous_velocity = np.zeros_like(previous)
                    previous_acceleration = np.zeros_like(previous)
                    consecutive_empty = 0
                    continue
                limit = np.minimum(
                    np.broadcast_to(np.asarray(velocity_limit_rad_s, dtype=float), previous.shape)
                    * intervals[frame_index],
                    np.broadcast_to(np.asarray(cap_rad, dtype=float), previous.shape),
                )
                desired_velocity = np.clip(
                    target_delta / intervals[frame_index],
                    -limit / intervals[frame_index], limit / intervals[frame_index])
                desired_acceleration = (
                    desired_velocity - previous_velocity) / intervals[frame_index]
                bounded_acceleration = previous_acceleration + np.clip(
                    desired_acceleration - previous_acceleration,
                    -jerk_limit * intervals[frame_index],
                    jerk_limit * intervals[frame_index],
                )
                bounded_acceleration = np.clip(
                    bounded_acceleration, -acceleration_limit, acceleration_limit)
                recovered_velocity = previous_velocity + (
                    bounded_acceleration * intervals[frame_index])
                recovered_velocity = np.clip(
                    recovered_velocity,
                    -limit / intervals[frame_index], limit / intervals[frame_index])
                recovered = previous + recovered_velocity * intervals[frame_index]
                if (transition_valid is not None and
                        not transition_valid(previous, recovered)):
                    q_path.append(previous.copy())
                    selected_indices.append(int(selected.index))
                    pose_valid.append(False)
                    recovery_mode.append("hold_recovery_collision")
                    previous_velocity = np.zeros_like(previous)
                    previous_acceleration = np.zeros_like(previous)
                    consecutive_empty = 0
                    continue
                applied = _wrapped_delta(recovered, previous, periodic)
                previous = recovered
                previous_velocity = applied / intervals[frame_index]
                previous_acceleration = bounded_acceleration
                q_path.append(previous.copy())
                selected_indices.append(int(selected.index))
                pose_valid.append(False)
                recovery_mode.append("limited_step")
                consecutive_empty = 0
                continue
            q_path.append(previous.copy())
            selected_indices.append(-1)
            pose_valid.append(False)
            if collision_safe_layer:
                mode = "hold_low_quality_candidate"
            elif pose_valid_layer:
                mode = "hold_state_collision_blocked"
            elif layer:
                mode = "hold_no_pose_candidate"
            else:
                mode = "hold_no_candidate"
            recovery_mode.append(mode)
            previous_velocity = np.zeros_like(previous)
            previous_acceleration = np.zeros_like(previous)
            consecutive_empty = consecutive_empty + 1 if not layer else 0
            continue
        next_q = np.asarray(selected.q, dtype=float).copy()
        next_velocity = _wrapped_delta(next_q, previous, periodic) / intervals[frame_index]
        previous_acceleration = (
            next_velocity - previous_velocity) / intervals[frame_index]
        previous_velocity = next_velocity
        previous = next_q
        q_path.append(previous.copy())
        selected_indices.append(int(selected.index))
        pose_valid.append(bool(selected.pose_valid))
        recovery_mode.append("none")
        consecutive_empty = 0
    return RecedingHorizonPath(
        q=np.asarray(q_path), selected_indices=np.asarray(selected_indices, dtype=int),
        pose_valid=np.asarray(pose_valid, dtype=bool),
        recovery_mode=np.asarray(recovery_mode),
    )
