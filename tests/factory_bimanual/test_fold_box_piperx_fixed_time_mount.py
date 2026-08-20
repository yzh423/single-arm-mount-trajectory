import numpy as np
import pytest

from scripts.search_fold_box_piperx_fixed_time_mount import (
    select_full_audited_mount,
    synchronized_mount_candidates,
)


def test_synchronized_mount_candidates_are_deterministic_and_well_separated():
    first = synchronized_mount_candidates(maximum=32)
    second = synchronized_mount_candidates(maximum=32)

    assert first == second
    assert len(first) == 32
    for mount in first:
        left = np.asarray(mount["xy"]["left"])
        right = np.asarray(mount["xy"]["right"])
        assert np.linalg.norm(right - left) >= 0.60 - 1e-12
        assert mount["shared_base_z_m"] in (0.81, 0.86)
        assert mount["roll"] == {"left": 0.0, "right": 0.0}
        assert mount["pitch"] == {"left": 0.0, "right": 0.0}


def _record(*, coverage, status="valid_selection", state=0, edge=0):
    return {
        "mount": {
            "xy": {"left": [-0.3, 0.35], "right": [-0.3, -0.35]},
            "yaw": {"left": 15.0, "right": 15.0},
            "shared_base_z_m": 0.81,
        },
        "status": status,
        "audit_scope": "full_timeline",
        "source_row_count": 1964,
        "audited_source_rows": 1964,
        "audit_fingerprint": "a" * 64,
        "continuous_pair_coverage": coverage,
        "pair_collision_frames": state,
        "pair_edge_collision_frames": edge,
        "longest_hold_frames": 10,
        "relaxed_tier_frames": 20,
        "mean_pair_pose_error": 0.01,
        "p10_pair_singularity_margin": 0.1,
        "base_distance_m": 0.7,
    }


def test_full_audited_mount_selection_rejects_sparse_or_colliding_records():
    sparse = _record(coverage=1.0)
    sparse["audit_scope"] = "sampled"
    colliding = _record(coverage=0.99, state=1)
    safe = _record(coverage=0.8)

    selected = select_full_audited_mount([sparse, colliding, safe])
    assert selected["full_audit_metrics"]["continuous_pair_coverage"] == 0.8

    with pytest.raises(RuntimeError, match="full-audited"):
        select_full_audited_mount([sparse, colliding])
