"""Pure contracts for bounded, resumable fixed-time mount searches."""
from __future__ import annotations

import numpy as np


MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M = 0.40


def rank_full_fixed_time_mount(record):
    """Apply the recommended collision-hard lexicographic Mount score."""
    collision = int(record.get("collision_frame_count", 10**9))
    return (
        collision,
        int(record.get("clearance_violation_frame_count", 0)),
        -float(record.get("synchronous_strict_coverage", 0.0)),
        int(record.get("longest_failure_run_frames", 10**9)),
        -float(record.get("connectable_safe_branch_ratio", 0.0)),
        -float(record.get("minimum_clearance_m", -np.inf)),
        -float(record.get("minimum_singularity_margin", -np.inf)),
        -float(record.get("minimum_joint_limit_margin_rad", -np.inf)),
        float(record.get("joint_travel_rad", np.inf)),
        int(record.get("required_retime_frame_count", 10**9)),
        float(record.get("position_mean_mm", np.inf)),
    )


def _is_complete_fixed_time(record, expected_rows):
    return (
        record.get("audit_scope") == "full_timeline"
        and int(record.get("source_row_count", -1)) == int(expected_rows)
        and int(record.get("audited_source_rows", -1)) == int(expected_rows)
        and record.get("timing_mode") == "fixed_source_time"
        and int(record.get("retimed_frame_count", -1)) == 0
        and isinstance(record.get("mount"), dict)
    )


def select_full_fixed_time_mount(records, expected_rows):
    """Select only collision-free complete immutable-source-time runs."""
    eligible = [record for record in records
                if _is_complete_fixed_time(record, expected_rows)
                and int(record.get("collision_frame_count", -1)) == 0
                and int(record.get(
                    "clearance_violation_frame_count", 0)) == 0
                and float(record.get("base_distance_m", 0.0)) >=
                MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M]
    if not eligible:
        raise RuntimeError(
            "no collision-free complete fixed-time mount evaluation")
    return min(eligible, key=rank_full_fixed_time_mount)


def targeted_source_indices(frame_count, failure_mask, maximum):
    """Cover the timeline and known failure regions with a bounded sample."""
    count = int(frame_count)
    limit = int(maximum)
    failed = np.asarray(failure_mask, dtype=bool)
    if count < 2 or limit < 2 or failed.shape != (count,):
        raise ValueError("invalid frame count, failure mask, or sample limit")
    uniform_count = max(2, limit // 2)
    uniform = np.rint(np.linspace(0, count - 1, uniform_count)).astype(int)
    failure_rows = np.flatnonzero(failed)
    if len(failure_rows):
        failure_count = max(1, limit - len(np.unique(uniform)))
        chosen = np.rint(np.linspace(
            0, len(failure_rows) - 1,
            min(failure_count, len(failure_rows)))).astype(int)
        failure_rows = failure_rows[chosen]
    indices = np.unique(np.r_[0, uniform, failure_rows, count - 1]).astype(int)
    if len(indices) > limit:
        keep = np.rint(np.linspace(0, len(indices) - 1, limit)).astype(int)
        indices = indices[keep]
    return indices


__all__ = [
    "rank_full_fixed_time_mount",
    "MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M",
    "select_full_fixed_time_mount",
    "targeted_source_indices",
]
