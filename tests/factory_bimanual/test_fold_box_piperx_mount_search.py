import numpy as np
import scripts.search_fold_box_piperx_mount as mount_search

from scripts.search_fold_box_piperx_mount import (
    BASE_Z_M,
    coarse_mount_candidates,
    layered_sample_indices,
    local_paired_refinements,
    rank_paired_mount_candidate,
    rank_mount_candidate,
    select_paired_mount,
    select_mount_pair,
)
from scripts.search_fold_box_piperx_paired_mount import (
    advance_connected_pairs, deterministic_pair_mounts,
)


def _full_audit(record):
    value = record | {
        "status": "valid_selection", "audit_scope": "full_timeline",
        "source_row_count": 10, "audited_source_rows": 10,
        "audit_fingerprint": "b" * 64,
        "structural_crossing_frames": 0,
        "structural_edge_crossing_frames": 0,
        "gripper_overlap_violation_frames": 0,
        "gripper_overlap_edge_violation_frames": 0,
        "maximum_gripper_overlap_m": 0.,
        "base_distance_m": .6,
    }
    value.setdefault("pair_edge_collision_frames", 0)
    return value


def test_layered_samples_include_endpoints_and_each_axis_extrema():
    position = np.asarray([
        [0, 0, 0], [2, 1, 2], [-3, 2, 1], [1, -4, 3], [0, 3, -5],
    ], dtype=float)
    indices = layered_sample_indices(5, position, uniform_count=3)
    assert {0, 4}.issubset(indices)
    for axis in range(3):
        assert int(np.argmin(position[:, axis])) in indices
        assert int(np.argmax(position[:, axis])) in indices


def test_coarse_mounts_are_upright_at_shared_capped_height():
    candidates = coarse_mount_candidates(np.asarray([0.0, 0.0]))
    assert candidates
    assert {record["base_z_m"] for record in candidates} == {BASE_Z_M}
    assert all(record["roll_deg"] == 0.0 and record["pitch_deg"] == 0.0
               for record in candidates)
    assert all(-180.0 <= record["yaw_deg"] <= 180.0 for record in candidates)


def test_mount_ranking_prioritizes_coverage_collision_then_required_time():
    base = {"coverage": .9, "collision_frames": 0, "required_time_s": 1.0,
            "p10_singularity_margin": .1, "mean_pose_error": .001}
    assert rank_mount_candidate(base | {"coverage": .95}) < rank_mount_candidate(base)
    assert rank_mount_candidate(base) < rank_mount_candidate(
        base | {"collision_frames": 1, "required_time_s": .1})
    assert rank_mount_candidate(base) < rank_mount_candidate(
        base | {"required_time_s": 2.0})


def test_pair_selection_rejects_overlapping_bases():
    left = [
        {"xy": [0.0, 0.0], "coverage": 1.0, "collision_frames": 0,
         "required_time_s": 1.0, "p10_singularity_margin": .1,
         "mean_pose_error": 0.0, "yaw_deg": 0.0},
        {"xy": [-.3, 0.0], "coverage": .95, "collision_frames": 0,
         "required_time_s": 1.0, "p10_singularity_margin": .1,
         "mean_pose_error": 0.0, "yaw_deg": 0.0},
    ]
    right = [left[0] | {"xy": [.01, 0.0]}, left[0] | {"xy": [.3, 0.0]}]
    selected = select_mount_pair(left, right, minimum_separation_m=.18)
    assert np.linalg.norm(np.subtract(selected["xy"]["left"],
                                      selected["xy"]["right"])) >= .18
    assert selected["shared_base_z_m"] == BASE_Z_M


def test_paired_ranking_prioritizes_collision_free_synchronous_coverage():
    high_independent_but_crossing = {
        "synchronous_pair_coverage": .5, "pair_collision_frames": 10,
        "left_coverage": .9, "right_coverage": .9,
        "p10_pair_singularity_margin": .2, "mean_pair_pose_error": .001,
    }
    lower_independent_but_safe = {
        "synchronous_pair_coverage": .8, "pair_collision_frames": 0,
        "left_coverage": .8, "right_coverage": .8,
        "p10_pair_singularity_margin": .1, "mean_pair_pose_error": .002,
    }
    assert rank_paired_mount_candidate(lower_independent_but_safe) < \
        rank_paired_mount_candidate(high_independent_but_crossing)


def test_paired_ranking_prefers_more_base_clearance_after_quality_ties():
    common = {
        "synchronous_pair_coverage": .8,
        "left_coverage": .8, "right_coverage": .8,
        "p10_pair_singularity_margin": .1,
        "mean_pair_pose_error": .002,
    }
    assert rank_paired_mount_candidate(common | {"base_distance_m": .75}) < \
        rank_paired_mount_candidate(common | {"base_distance_m": .60})


def test_select_paired_mount_uses_pair_metric_not_two_independent_scores():
    crossing = {
        "mount": {"xy": {"left": [-.3, -.2], "right": [-.2, .2]},
                  "yaw": {"left": 90., "right": -120.},
                  "shared_base_z_m": .76},
        "synchronous_pair_coverage": .5, "pair_collision_frames": 12,
        "left_coverage": .9, "right_coverage": .9,
        "p10_pair_singularity_margin": .2, "mean_pair_pose_error": .001,
    }
    safe = {
        "mount": {"xy": {"left": [-.35, .25], "right": [-.3, -.45]},
                  "yaw": {"left": 15., "right": 15.},
                  "shared_base_z_m": .81},
        "synchronous_pair_coverage": .8, "pair_collision_frames": 0,
        "left_coverage": .8, "right_coverage": .8,
        "p10_pair_singularity_margin": .1, "mean_pair_pose_error": .002,
    }
    selected = select_paired_mount([
        _full_audit(crossing), _full_audit(safe)])
    assert selected["xy"] == safe["mount"]["xy"]
    assert selected["paired_sparse_metrics"]["pair_collision_frames"] == 0


def test_paired_mount_selection_hard_rejects_swept_edge_collision():
    colliding = {
        "mount": {"xy": {"left": [0., 0.], "right": [.4, 0.]},
                  "yaw": {"left": 0., "right": 180.},
                  "shared_base_z_m": .81},
        "synchronous_pair_coverage": 1., "pair_collision_frames": 0,
        "pair_edge_collision_frames": 1, "left_coverage": 1.,
        "right_coverage": 1., "p10_pair_singularity_margin": .5,
        "mean_pair_pose_error": 0.,
    }
    safe = dict(colliding)
    safe["mount"] = {**colliding["mount"],
                     "xy": {"left": [-.1, 0.], "right": [.5, 0.]}}
    safe["synchronous_pair_coverage"] = .5
    safe["pair_edge_collision_frames"] = 0
    selected = select_paired_mount([
        _full_audit(colliding), _full_audit(safe)])
    assert selected["xy"] == safe["mount"]["xy"]


def test_local_paired_refinement_keeps_bases_upright_and_same_height():
    mount = {"xy": {"left": [-.3, .2], "right": [-.3, -.4]},
             "yaw": {"left": 15., "right": 15.},
             "shared_base_z_m": .81}
    candidates = local_paired_refinements(
        mount, side="right", xy_step_m=.05, yaw_step_deg=15.)
    assert mount in candidates
    assert all(candidate["shared_base_z_m"] == .81 for candidate in candidates)
    assert all(set(candidate) == {"xy", "yaw", "shared_base_z_m"}
               for candidate in candidates)


def test_collision_aware_refinement_moves_both_bases_apart_symmetrically():
    assert hasattr(mount_search, "collision_aware_pair_refinements")
    mount = {"xy": {"left": [-.2, .1], "right": [.2, -.1]},
             "yaw": {"left": -30., "right": 150.},
             "shared_base_z_m": .81}
    original_distance = np.linalg.norm(np.subtract(
        mount["xy"]["left"], mount["xy"]["right"]))

    candidates = mount_search.collision_aware_pair_refinements(
        mount, separation_step_m=.04, yaw_step_deg=10.)

    distances = [np.linalg.norm(np.subtract(
        candidate["xy"]["left"], candidate["xy"]["right"]))
        for candidate in candidates]
    np.testing.assert_allclose(
        max(distances), original_distance + .08, atol=1e-12)
    assert mount in candidates
    assert mount["xy"] == {"left": [-.2, .1], "right": [.2, -.1]}


def test_missing_or_disconnected_layer_does_not_reset_pair_history():
    previous = [("left_old", "right_old")]

    after_gap, disconnected = advance_connected_pairs(
        previous, [], lambda old, new: True)
    assert after_gap == previous
    assert disconnected

    proposed = [("left_new", "right_new")]
    blocked, disconnected = advance_connected_pairs(
        after_gap, proposed, lambda old, new: False)
    assert blocked == previous
    assert disconnected

    connected, disconnected = advance_connected_pairs(
        blocked, proposed, lambda old, new: old == previous[0])
    assert connected == proposed
    assert not disconnected


def test_expanded_pair_grid_is_deterministic_bounded_and_spatially_diverse():
    task = type("Task", (), {
        "left_position_m": np.tile([-.1, .1, .9], (10, 1)),
        "right_position_m": np.tile([.1, -.1, .9], (10, 1)),
    })()

    first = deterministic_pair_mounts(task, maximum=72)
    second = deterministic_pair_mounts(task, maximum=72)

    assert first == second
    assert len(first) == 72
    assert all(record["shared_base_z_m"] == .81 for record in first)
    left_xy = np.asarray([record["xy"]["left"] for record in first])
    assert np.ptp(left_xy[:, 0]) > .5
    assert np.ptp(left_xy[:, 1]) > .5
    yaw_pairs = {(record["yaw"]["left"], record["yaw"]["right"])
                 for record in first}
    assert len(yaw_pairs) >= 12
