import numpy as np
import pytest

from factory_bimanual.staged_mount_search import (
    rank_full_fixed_time_mount,
    select_full_fixed_time_mount,
    targeted_source_indices,
)
from scripts.search_piperx_fixed_time_mounts import (
    local_mount_candidates,
    select_stage_finalists,
    xarm6_style_mount_candidates,
)


def _full_record(*, rows=100, collision=0, coverage=.8,
                 timing="fixed_source_time", retimed=0):
    return {
        "mount": {"xy": {"left": [-.3, .35], "right": [-.3, -.35]},
                  "yaw": {"left": 15., "right": 15.},
                  "shared_base_z_m": .81},
        "audit_scope": "full_timeline",
        "source_row_count": rows,
        "audited_source_rows": rows,
        "timing_mode": timing,
        "retimed_frame_count": retimed,
        "collision_frame_count": collision,
        "synchronous_strict_coverage": coverage,
        "cannot_follow_frame_count": int(round(rows * (1 - coverage))),
        "position_mean_mm": 1.0,
        "base_distance_m": .5,
        "longest_failure_run_frames": 4,
        "connectable_safe_branch_ratio": .7,
        "minimum_clearance_m": .02,
        "minimum_singularity_margin": .05,
        "minimum_joint_limit_margin_rad": .1,
        "joint_travel_rad": 10.,
        "required_retime_frame_count": 0,
    }


def test_full_mount_selection_requires_complete_fixed_time_evidence():
    safe = _full_record(coverage=.75)
    colliding = _full_record(collision=1, coverage=.99)
    too_close = _full_record(coverage=1.0)
    too_close["base_distance_m"] = .39
    incomplete = _full_record(rows=99)
    incomplete["source_row_count"] = 100
    sampled = _full_record(coverage=1.0)
    sampled["audit_scope"] = "sampled"
    retimed = _full_record(coverage=1.0, retimed=1)

    selected = select_full_fixed_time_mount(
        [colliding, too_close, incomplete, sampled, retimed, safe],
        expected_rows=100)
    assert selected["synchronous_strict_coverage"] == .75

    with pytest.raises(RuntimeError, match="collision-free"):
        select_full_fixed_time_mount([colliding], expected_rows=100)

    with pytest.raises(RuntimeError, match="complete fixed-time"):
        select_full_fixed_time_mount(
            [incomplete, sampled, retimed], expected_rows=100)


def test_full_mount_ranking_is_collision_first():
    safe = _full_record(collision=0, coverage=.5)
    higher_coverage = _full_record(collision=1, coverage=.9)
    assert (rank_full_fixed_time_mount(safe) <
            rank_full_fixed_time_mount(higher_coverage))


def test_mount_quality_ranking_uses_recommended_lexicographic_order():
    baseline = _full_record(coverage=.8)
    higher_coverage = {**baseline, "synchronous_strict_coverage": .81,
                       "longest_failure_run_frames": 99}
    shorter_failure = {**baseline, "longest_failure_run_frames": 3}
    more_connectable = {**baseline, "connectable_safe_branch_ratio": .8}
    more_clearance = {**baseline, "minimum_clearance_m": .03}

    assert rank_full_fixed_time_mount(higher_coverage) < \
        rank_full_fixed_time_mount(baseline)
    assert rank_full_fixed_time_mount(shorter_failure) < \
        rank_full_fixed_time_mount(baseline)
    assert rank_full_fixed_time_mount(more_connectable) < \
        rank_full_fixed_time_mount(baseline)
    assert rank_full_fixed_time_mount(more_clearance) < \
        rank_full_fixed_time_mount(baseline)


def test_targeted_indices_are_deterministic_bounded_and_cover_failures():
    failed = np.zeros(100, dtype=bool)
    failed[40:50] = True
    first = targeted_source_indices(100, failed, maximum=20)
    second = targeted_source_indices(100, failed, maximum=20)

    assert np.array_equal(first, second)
    assert len(first) <= 20
    assert first[0] == 0 and first[-1] == 99
    assert np.any((first >= 40) & (first < 50))
    assert np.all(np.diff(first) > 0)


def test_xarm6_style_candidates_scan_upright_xy_spacing_yaw_and_height():
    first = xarm6_style_mount_candidates(maximum=32)
    second = xarm6_style_mount_candidates(maximum=32)
    assert first == second
    assert len(first) == 32
    assert len({tuple(mount["yaw"].values()) for mount in first}) > 2
    assert len({mount["shared_base_z_m"] for mount in first}) > 1
    for mount in first:
        left = np.asarray(mount["xy"]["left"])
        right = np.asarray(mount["xy"]["right"])
        assert np.linalg.norm(right - left) >= .40 - 1e-12
        assert np.linalg.norm(right - left) <= .80 + 1e-12
        assert .79 <= mount["shared_base_z_m"] <= 1.0
        assert mount["roll"] == {"left": 0.0, "right": 0.0}
        assert mount["pitch"] == {"left": 0.0, "right": 0.0}
    assert min(np.linalg.norm(
        np.asarray(mount["xy"]["right"]) -
        np.asarray(mount["xy"]["left"])) for mount in first) <= .45


def test_stage_finalists_reject_sampled_collision_before_tracking_rank():
    records = [
        {"stage": "sparse", "status": "complete", "mount_key": "a",
         "synchronous_strict_coverage": .7, "collision_frame_count": 0,
         "cannot_follow_frame_count": 3, "position_mean_mm": 1.,
         "base_distance_m": .5},
        {"stage": "sparse", "status": "complete", "mount_key": "b",
         "synchronous_strict_coverage": .9, "collision_frame_count": 1,
         "cannot_follow_frame_count": 1, "position_mean_mm": 1.,
         "base_distance_m": .5},
        {"stage": "sparse", "status": "complete", "mount_key": "close",
         "synchronous_strict_coverage": 1., "collision_frame_count": 0,
         "cannot_follow_frame_count": 0, "position_mean_mm": 0.,
         "base_distance_m": .39},
        {"stage": "dense", "status": "complete", "mount_key": "c",
         "synchronous_strict_coverage": 1., "collision_frame_count": 0,
         "cannot_follow_frame_count": 0, "position_mean_mm": 0.,
         "base_distance_m": .5},
    ]
    selected = select_stage_finalists(records, stage="sparse", maximum=2)
    assert [record["mount_key"] for record in selected] == ["a"]


def test_local_refinement_varies_both_mounts_yaw_and_shared_z_within_bounds():
    center = {
        "xy": {"left": [-.42, .25], "right": [-.40, -.52]},
        "yaw": {"left": -15., "right": 15.},
        "shared_base_z_m": 1.0,
        "roll": {"left": 0., "right": 0.},
        "pitch": {"left": 0., "right": 0.},
    }
    candidates = local_mount_candidates(center, maximum=12)

    assert candidates[0] == center
    assert len(candidates) == 12
    assert len({tuple(item["xy"]["left"]) for item in candidates}) > 1
    assert len({tuple(item["xy"]["right"]) for item in candidates}) > 1
    assert len({tuple(item["yaw"].values()) for item in candidates}) > 1
    assert len({item["shared_base_z_m"] for item in candidates}) > 1
    for mount in candidates:
        distance = np.linalg.norm(
            np.asarray(mount["xy"]["right"]) -
            np.asarray(mount["xy"]["left"]))
        assert .40 <= distance <= .80
        assert .79 <= mount["shared_base_z_m"] <= 1.0
