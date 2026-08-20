"""Episode-level trajectory-following metrics shared by search and reports."""
from __future__ import annotations

import numpy as np


FAILURE_REASON_ORDER = (
    "self_collision", "table_collision", "position_and_orientation",
    "position", "orientation", "joint_discontinuity",
)


def classify_solver_failures(
    *, rolling_success, position_feasible, independent_pose_feasible,
    collision, jump_violation,
) -> np.ndarray:
    """Separate physical infeasibility from rolling branch-tracking failure."""
    rolling = np.asarray(rolling_success, dtype=bool)
    position = np.asarray(position_feasible, dtype=bool)
    independent = np.asarray(independent_pose_feasible, dtype=bool)
    collision = np.asarray(collision, dtype=bool)
    jump = np.asarray(jump_violation, dtype=bool)
    if not (rolling.shape == position.shape == independent.shape
            == collision.shape == jump.shape):
        raise ValueError("solver diagnostic arrays must have matching shapes")
    reasons = np.full(rolling.shape, "none", dtype=object)
    failed = ~rolling
    reasons[failed & collision] = "collision"
    reasons[failed & (reasons == "none") & ~position] = "position_unreachable"
    reasons[failed & (reasons == "none") & position & ~independent] = "pose_infeasible"
    reasons[failed & (reasons == "none") & jump] = "jump_violation"
    reasons[failed & (reasons == "none") & independent] = "branch_lost"
    reasons[failed & (reasons == "none")] = "iteration_exhausted"
    return reasons.astype(str)


def planner_failure_diagnostics(*, rolling_success, collision, jump_violation,
                                recovery_mode) -> dict[str, object]:
    """Report planner causes without relabelling correlated pose errors as a cause."""
    success = np.asarray(rolling_success, dtype=bool)
    collision = np.asarray(collision, dtype=bool)
    jump = np.asarray(jump_violation, dtype=bool)
    recovery = np.asarray(recovery_mode).astype(str)
    if not (success.shape == collision.shape == jump.shape == recovery.shape):
        raise ValueError("planner diagnostic arrays must have matching shapes")
    reasons = np.full(success.shape, "none", dtype=object)
    failed = ~success
    reasons[failed & collision] = "collision"
    reasons[failed & (reasons == "none") & jump] = "jump_violation"
    reasons[failed & (reasons == "none") & (recovery == "hold")] = "branch_lost"
    reasons[failed & (reasons == "none")] = "tolerance_miss"
    order = ("collision", "jump_violation", "branch_lost", "tolerance_miss")
    return {"per_frame": reasons.astype(str),
            "counts": {reason: int(np.sum(reasons == reason)) for reason in order
                       if np.any(reasons == reason)}}


def failure_reason_diagnostics(*, position_error_m, orientation_error_rad,
                               table_collision, self_collision,
                               joint_discontinuity=None,
                               position_tolerance_m=1e-3,
                               orientation_tolerance_rad=np.deg2rad(1.5)) -> dict[str, object]:
    """Classify every failed frame with transparent primary/secondary causes."""
    position = np.asarray(position_error_m, dtype=float)
    orientation = np.asarray(orientation_error_rad, dtype=float)
    table = np.asarray(table_collision, dtype=bool)
    self_hit = np.asarray(self_collision, dtype=bool)
    jump = np.zeros_like(table) if joint_discontinuity is None else np.asarray(joint_discontinuity, dtype=bool)
    if not (position.shape == orientation.shape == table.shape == self_hit.shape == jump.shape):
        raise ValueError("failure diagnostic arrays must have matching shapes")
    masks = {
        "joint_discontinuity": jump,
        "self_collision": self_hit,
        "table_collision": table,
        "position": position > position_tolerance_m,
        "orientation": orientation > orientation_tolerance_rad,
    }
    masks["position_and_orientation"] = masks["position"] & masks["orientation"]
    primary = np.full(len(position), "none", dtype=object)
    # Priority reflects physical invalidity first, then tracking error.
    for reason in FAILURE_REASON_ORDER:
        primary[(primary == "none") & masks[reason]] = reason
    counts = {reason: int(mask.sum()) for reason, mask in masks.items()}
    primary_counts = {reason: int(np.sum(primary == reason)) for reason in FAILURE_REASON_ORDER}
    return {
        "primary_per_frame": primary.astype(str),
        "affected_frame_counts": counts,
        "primary_frame_counts": primary_counts,
    }


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def episode_follow_metrics(*, pose_success, collision, position_error_m,
                           orientation_error_rad) -> dict[str, object]:
    """Return binary whole-episode success plus frame-level diagnostics."""
    pose = np.asarray(pose_success, dtype=bool)
    collision = np.asarray(collision, dtype=bool)
    position = np.asarray(position_error_m, dtype=float)
    orientation = np.asarray(orientation_error_rad, dtype=float)
    if not (pose.ndim == 1 and pose.shape == collision.shape == position.shape == orientation.shape):
        raise ValueError("all episode arrays must be non-empty matching one-dimensional arrays")
    if len(pose) == 0:
        raise ValueError("episode must contain at least one frame")
    safe = pose & ~collision
    failure = ~safe
    return {
        "episode_success": bool(safe.all()),
        "frame_coverage": float(safe.mean()),
        "successful_frames": int(safe.sum()),
        "frames": int(len(safe)),
        "longest_failure_run_frames": _longest_true_run(failure),
        "collision_frames": int(collision.sum()),
        "position_rmse_m": float(np.sqrt(np.mean(position * position))),
        "orientation_rmse_rad": float(np.sqrt(np.mean(orientation * orientation))),
    }


def follow_rank(metrics: dict[str, object]) -> tuple[float, ...]:
    """Lexicographic objective: follow extent first, then continuity/errors."""
    return (
        float(metrics["episode_success"]),
        float(metrics["frame_coverage"]),
        -float(metrics["longest_failure_run_frames"]),
        -float(metrics["collision_frames"]),
        -float(metrics["position_rmse_m"]),
        -float(metrics["orientation_rmse_rad"]),
    )
