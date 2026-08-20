"""Offline strict bimanual rolling multibranch path selection.

Robot-specific IK and MuJoCo evaluation enter through callbacks.  This module
owns synchronized branch pairing, timestamp constraints, beam search, traceback,
and full source-row alignment only.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Sequence

import numpy as np

from .bimanual_collision import CallbackCollisionChecker


@dataclass(frozen=True)
class IKCandidate:
    q: np.ndarray
    branch_index: int
    pose_cost: float = 0.0
    actual_tcp: np.ndarray | None = None
    position_error_m: float = np.nan
    orientation_error_rad: float = np.nan
    joint_limit_margin_rad: float = np.nan
    singularity_margin: float = np.nan
    wrist_risk: float = 0.0
    adaptation_level: str = "original"
    adapted_target_quaternion_wxyz: np.ndarray | None = None
    tool_axis_offset_deg: float = 0.0
    swing_offset_deg: float = 0.0


CandidateGenerator = Callable[[Any, Any, Any, int, str], Sequence[IKCandidate]]


@dataclass(frozen=True)
class BimanualIKConfig:
    candidate_generator: CandidateGenerator
    collision_checker: CallbackCollisionChecker
    beam_width: int = 64
    max_velocity_rad_s: float | np.ndarray = np.inf
    max_jump_rad: float | np.ndarray = np.inf
    transition_cost_weight: float = 1.0
    periodic_joints: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None


@dataclass
class BimanualIKResult:
    left_q: np.ndarray
    right_q: np.ndarray
    left_actual_tcp: np.ndarray
    right_actual_tcp: np.ndarray
    left_position_error_m: np.ndarray
    right_position_error_m: np.ndarray
    left_orientation_error_rad: np.ndarray
    right_orientation_error_rad: np.ndarray
    paired_branch_indices: list[tuple[int, int] | None]
    success: np.ndarray
    failures: list[str | None]
    collision_classes: list[tuple[str, ...]]
    source_row_indices: np.ndarray


@dataclass
class _Node:
    left: IKCandidate
    right: IKCandidate
    cost: float
    parent: "_Node | None"
    row: int


def _periodic_for_side(config: BimanualIKConfig, side_index: int,
                       shape: tuple[int, ...]) -> np.ndarray:
    value = config.periodic_joints
    if value is None:
        return np.zeros(shape, dtype=bool)
    if isinstance(value, tuple):
        value = value[side_index]
    result = np.asarray(value, dtype=bool)
    if result.shape != shape:
        raise ValueError("periodic_joints must match each arm's joint shape")
    return result


def _motion_delta(new: np.ndarray, old: np.ndarray, periodic: np.ndarray) -> np.ndarray:
    delta = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    delta = delta.copy()
    delta[periodic] = (delta[periodic] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def _within_motion(previous: _Node, left: IKCandidate, right: IKCandidate, dt: float, config: BimanualIKConfig) -> bool:
    if dt <= 0:
        return False
    for side_index, (old, new) in enumerate(
            ((previous.left.q, left.q), (previous.right.q, right.q))):
        periodic = _periodic_for_side(config, side_index, np.asarray(old).shape)
        delta = np.abs(_motion_delta(new, old, periodic))
        if np.any(delta > np.asarray(config.max_jump_rad)):
            return False
        if np.any(delta / dt > np.asarray(config.max_velocity_rad_s)):
            return False
    return True


def _transition_cost(previous: _Node, left: IKCandidate, right: IKCandidate, config: BimanualIKConfig) -> float:
    left_periodic = _periodic_for_side(config, 0, previous.left.q.shape)
    right_periodic = _periodic_for_side(config, 1, previous.right.q.shape)
    squared = (np.sum(_motion_delta(left.q, previous.left.q, left_periodic) ** 2) +
               np.sum(_motion_delta(right.q, previous.right.q, right_periodic) ** 2))
    return float(config.transition_cost_weight * squared)


def solve_strict_bimanual_path(model: Any, contract: Any, task: Any, config: BimanualIKConfig) -> BimanualIKResult:
    """Select a globally consistent path within a bounded per-row frontier."""
    times = np.asarray(task.time_s, dtype=float)
    rows = np.asarray(task.source_row_indices)
    count = len(times)
    generated = [
        (
            list(config.candidate_generator(model, contract, task, row, "left")),
            list(config.candidate_generator(model, contract, task, row, "right")),
        )
        for row in range(count)
    ]
    sample_left = next((c for sides in generated for c in sides[0]), None)
    sample_right = next((c for sides in generated for c in sides[1]), None)
    left_dof = len(sample_left.q) if sample_left is not None else int(getattr(contract, "dof", 0))
    right_dof = len(sample_right.q) if sample_right is not None else int(getattr(contract, "dof", 0))
    selected: list[_Node | None] = [None] * count
    failures: list[str | None] = [None] * count
    collisions: list[tuple[str, ...]] = [()] * count

    frontier: list[_Node] = []
    def finish_segment() -> None:
        if not frontier:
            return
        node = min(frontier, key=lambda item: item.cost)
        while node is not None:
            selected[node.row] = node
            node = node.parent

    for row, (left_candidates, right_candidates) in enumerate(generated):
        valid_pairs: list[tuple[IKCandidate, IKCandidate]] = []
        rejected_classes: set[str] = set()
        for left, right in product(left_candidates, right_candidates):
            report = config.collision_checker.state(left.q, right.q)
            if report.valid:
                valid_pairs.append((left, right))
            else:
                rejected_classes.update(item.value for item in report.classes)
        if not valid_pairs:
            finish_segment()
            frontier = []
            failures[row] = "branch_lost" if not left_candidates or not right_candidates else "collision"
            collisions[row] = tuple(sorted(rejected_classes))
            continue

        next_frontier: list[_Node] = []
        if not frontier:
            next_frontier = [_Node(l, r, l.pose_cost + r.pose_cost, None, row) for l, r in valid_pairs]
        else:
            dt = times[row] - times[row - 1]
            motion_rejected = False
            transition_rejected: set[str] = set()
            for left, right in valid_pairs:
                for previous in frontier:
                    if not _within_motion(previous, left, right, dt, config):
                        motion_rejected = True
                        continue
                    report = config.collision_checker.transition(
                        (previous.left.q, previous.right.q), (left.q, right.q)
                    )
                    if not report.valid:
                        transition_rejected.update(item.value for item in report.classes)
                        continue
                    cost = previous.cost + left.pose_cost + right.pose_cost + _transition_cost(previous, left, right, config)
                    next_frontier.append(_Node(left, right, cost, previous, row))
            if not next_frontier:
                finish_segment()
                frontier = []
                failures[row] = "velocity_jump_violation" if motion_rejected and not transition_rejected else "transition_collision"
                collisions[row] = tuple(sorted(transition_rejected))
                continue
        next_frontier.sort(key=lambda item: (item.cost, item.left.branch_index, item.right.branch_index))
        frontier = next_frontier[: max(1, config.beam_width)]
    finish_segment()

    left_q = np.full((count, left_dof), np.nan)
    right_q = np.full((count, right_dof), np.nan)
    left_tcp = np.full((count, 7), np.nan)
    right_tcp = np.full((count, 7), np.nan)
    left_pos = np.full(count, np.nan)
    right_pos = np.full(count, np.nan)
    left_rot = np.full(count, np.nan)
    right_rot = np.full(count, np.nan)
    branches: list[tuple[int, int] | None] = [None] * count
    for row, node in enumerate(selected):
        if node is None:
            continue
        left_q[row], right_q[row] = node.left.q, node.right.q
        branches[row] = (node.left.branch_index, node.right.branch_index)
        left_pos[row], right_pos[row] = node.left.position_error_m, node.right.position_error_m
        left_rot[row], right_rot[row] = node.left.orientation_error_rad, node.right.orientation_error_rad
        if node.left.actual_tcp is not None:
            left_tcp[row] = node.left.actual_tcp
        if node.right.actual_tcp is not None:
            right_tcp[row] = node.right.actual_tcp
    success = np.asarray([node is not None for node in selected], dtype=bool)
    return BimanualIKResult(left_q, right_q, left_tcp, right_tcp, left_pos, right_pos, left_rot, right_rot,
                             branches, success, failures, collisions, rows.copy())
