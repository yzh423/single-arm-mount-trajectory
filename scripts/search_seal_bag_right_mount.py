"""Targeted upright right-mount search for the seal-bag fixed-time failure."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


def visual_wrist_flip_mask(
    q, target_positions_m, *, maximum_joint_step_deg=35.0,
    maximum_wrist_norm_deg=45.0, maximum_target_step_m=.02,
):
    """Flag visually invalid wrist-branch switches independent of retiming."""
    joints = np.asarray(q, dtype=float)
    targets = np.asarray(target_positions_m, dtype=float)
    if joints.ndim != 2 or joints.shape[1] < 6:
        raise ValueError("q must have at least six joints per frame")
    if targets.shape != (len(joints), 3):
        raise ValueError("target positions must have shape (frames, 3)")
    delta = np.diff(joints, axis=0)
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    maximum = np.max(np.abs(delta[:, 3:6]), axis=1)
    wrist_norm = np.linalg.norm(delta[:, 3:6], axis=1)
    target_step = np.linalg.norm(np.diff(targets, axis=0), axis=1)
    invalid = ((maximum > np.deg2rad(maximum_joint_step_deg)) |
               (wrist_norm > np.deg2rad(maximum_wrist_norm_deg))) & \
              (target_step <= maximum_target_step_m)
    return np.r_[False, invalid]


def exhaustive_tabletop_right_mounts(
    *, table_half_size_m, mount_radius_m, xy_step_m, yaw_step_deg,
    left_xy, minimum_center_distance_m,
):
    """Enumerate every upright grid pose whose circular adapter fits the table."""
    half_x, half_y = (float(value) for value in table_half_size_m)
    radius = float(mount_radius_m)
    step = float(xy_step_m)
    yaw_step = float(yaw_step_deg)
    if min(half_x, half_y, radius, step, yaw_step,
           float(minimum_center_distance_m)) <= 0:
        raise ValueError("search dimensions and steps must be positive")
    if radius > min(half_x, half_y) or 360.0 / yaw_step % 1 > 1e-12:
        raise ValueError("invalid adapter size or yaw step")
    x_limit, y_limit = half_x - radius, half_y - radius
    xs = np.arange(-x_limit, x_limit + step * .5, step)
    ys = np.arange(-y_limit, y_limit + step * .5, step)
    yaws = np.arange(0.0, 360.0, yaw_step)
    left = np.asarray(left_xy, dtype=float)
    return tuple(
        (float(round(x, 12)), float(round(y, 12)), float(yaw))
        for x, y in itertools.product(xs, ys)
        if np.linalg.norm(np.asarray((x, y)) - left)
        >= float(minimum_center_distance_m) - 1e-12
        for yaw in yaws
    )


def longest_failure_run(failed) -> int:
    """Return the longest contiguous true run in a one-dimensional mask."""
    values = np.asarray(failed, dtype=bool)
    if values.ndim != 1:
        raise ValueError("failed must be one-dimensional")
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return int(np.max(stops - starts)) if len(starts) else 0


def whole_trajectory_window_indices(*, frame_count: int, window_count: int,
                                    window_length: int):
    """Return evenly spaced contiguous windows including both timeline ends."""
    if frame_count < 1 or window_count < 1 or window_length < 1:
        raise ValueError("frame_count, window_count and window_length must be positive")
    if window_length > frame_count:
        raise ValueError("window_length exceeds frame_count")
    starts = np.rint(np.linspace(0, frame_count - window_length, window_count)).astype(int)
    starts = np.unique(starts)
    return tuple(start + np.arange(window_length, dtype=int) for start in starts)


def generate_right_mount_candidates(
    *, left_xy, right_xy, left_yaw_deg, right_yaw_deg,
    xy_offsets_m, yaw_offsets_deg, shared_base_z_m,
):
    """Return deterministic candidates while keeping the left mount and heights fixed."""
    offsets = list(itertools.product(xy_offsets_m, xy_offsets_m, yaw_offsets_deg))
    offsets.sort(key=lambda item: (
        abs(float(item[0])) + abs(float(item[1])), abs(float(item[2])),
        float(item[0]), float(item[1]), float(item[2])))
    result = []
    for dx, dy, dyaw in offsets:
        result.append({
            "xy": {
                "left": tuple(float(value) for value in left_xy),
                "right": (float(right_xy[0] + dx), float(right_xy[1] + dy)),
            },
            "yaw": {
                "left": float(left_yaw_deg),
                "right": float((right_yaw_deg + dyaw) % 360.0),
            },
            "shared_base_z_m": float(shared_base_z_m),
            "tilt_deg": {"left": 0.0, "right": 0.0},
            "roll_deg": {"left": 0.0, "right": 0.0},
        })
    return result


def right_mount_score(
    *, strict_success, collision_free, selected_joint_margin_rad,
    candidate_counts, mean_pose_error, displacement_m, yaw_displacement_deg,
):
    """Lexicographic score: fixed-time safety first, then margin and small change."""
    success = np.asarray(strict_success, dtype=bool)
    safe = success & np.asarray(collision_free, dtype=bool)
    counts = np.asarray(candidate_counts, dtype=float)
    margins = np.asarray(selected_joint_margin_rad, dtype=float)
    finite_margins = margins[np.isfinite(margins) & safe]
    minimum_margin = float(np.min(finite_margins)) if len(finite_margins) else -np.inf
    failure_count = int(np.count_nonzero(~safe))
    longest_failure = longest_failure_run(~safe)
    return (
        failure_count,
        longest_failure,
        -minimum_margin,
        -int(np.count_nonzero(counts > 0)),
        float(mean_pose_error),
        float(displacement_m),
        abs(float(yaw_displacement_deg)),
    )


def main():
    """Persist the complete grid and deterministic geometric pruning evidence."""
    from factory_bimanual.registration import RigidTaskRegistration, register_task
    from factory_bimanual.source_data import load_factory_task
    from scripts.render_factory_dual_xarm6_se3_follow import CSV, ROOT, SELECTED_MOUNT

    source = load_factory_task(CSV, "seal_bag")
    points = np.vstack((source.left_position_m, source.right_position_m))
    translation = np.array([
        -points[:, 0].mean(), -points[:, 1].mean(), .9 - points[:, 2].min()])
    task = register_task(source, RigidTaskRegistration(np.eye(3), translation))
    mounts = exhaustive_tabletop_right_mounts(
        table_half_size_m=(.9, .7), mount_radius_m=.075,
        xy_step_m=.05, yaw_step_deg=15.0,
        left_xy=SELECTED_MOUNT["xy"]["left"],
        minimum_center_distance_m=.16,
    )
    target = np.asarray(task.right_position_m, dtype=float)
    base_z = float(SELECTED_MOUNT["shared_base_z_m"])
    rows = []
    for x, y, yaw in mounts:
        distances = np.linalg.norm(
            target - np.asarray((x, y, base_z))[None, :], axis=1)
        maximum = float(np.max(distances)); minimum = float(np.min(distances))
        valid = maximum <= .85 and minimum >= .08
        rows.append({
            "right_x_m": x, "right_y_m": y, "right_yaw_deg": yaw,
            "maximum_target_distance_m": maximum,
            "minimum_target_distance_m": minimum,
            "geometric_status": "candidate" if valid else "pruned_reach",
        })
    output = (ROOT / "reports/factory_bimanual/seal_bag_dual_xarm6/"
              "right_mount_exhaustive_grid_5cm_15deg.json")
    output.write_text(json.dumps({
        "scope": {
            "table_half_size_m": [.9, .7], "mount_radius_m": .075,
            "xy_step_m": .05, "yaw_step_deg": 15.0,
            "base_z_m": base_z, "tilt_deg": 0.0, "roll_deg": 0.0,
            "minimum_center_distance_from_left_m": .16,
            "necessary_reach_interval_m": [.08, .85],
        },
        "enumerated_pose_count": len(rows),
        "geometric_candidate_count": sum(
            row["geometric_status"] == "candidate" for row in rows),
        "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
