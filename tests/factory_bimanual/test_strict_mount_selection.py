import pytest

from scripts.search_fold_box_piperx_mount import select_paired_mount


def _record(**changes):
    value = {
        "mount": {
            "xy": {"left": [-.3, .2], "right": [-.3, -.4]},
            "yaw": {"left": 15., "right": 15.},
            "shared_base_z_m": .81,
        },
        "status": "valid_selection",
        "audit_scope": "full_timeline",
        "audited_source_rows": 2439,
        "source_row_count": 2439,
        "audit_fingerprint": "a" * 64,
        "synchronous_pair_coverage": .8,
        "continuous_pair_coverage": .8,
        "left_coverage": .8,
        "right_coverage": .8,
        "pair_collision_frames": 0,
        "pair_edge_collision_frames": 0,
        "structural_crossing_frames": 0,
        "structural_edge_crossing_frames": 0,
        "gripper_overlap_violation_frames": 0,
        "gripper_overlap_edge_violation_frames": 0,
        "gripper_overlap_frames": 3,
        "maximum_gripper_overlap_m": .07,
        "p10_pair_singularity_margin": .1,
        "mean_pair_pose_error": .01,
        "longest_hold_frames": 20,
        "relaxed_tier_frames": 10,
        "base_distance_m": .6,
    }
    value.update(changes)
    return value


@pytest.mark.parametrize("changes", [
    {"status": "sparse_survivor"},
    {"audit_scope": "sparse"},
    {"audited_source_rows": 2438},
    {"audit_fingerprint": ""},
    {"pair_collision_frames": 1},
    {"pair_edge_collision_frames": 1},
])
def test_selection_rejects_records_without_complete_zero_crossing_audit(changes):
    with pytest.raises(RuntimeError, match="full-audited collision-free"):
        select_paired_mount([_record(**changes)])


def test_selection_allows_limited_contact_free_gripper_overlap():
    selected = select_paired_mount([_record()])
    assert selected["xy"]["left"] == [-.3, .2]
    assert selected["paired_sparse_metrics"]["gripper_overlap_frames"] == 3


def test_contact_free_projection_crossing_does_not_reject_reachable_mount():
    crossing = _record(
        synchronous_pair_coverage=1., continuous_pair_coverage=1.,
        structural_crossing_frames=1,
        structural_edge_crossing_frames=1,
        gripper_overlap_violation_frames=1,
        gripper_overlap_edge_violation_frames=1,
        maximum_structural_crossing_m=.12,
        maximum_gripper_overlap_m=.12)
    safe = _record(
        mount=_record()["mount"] | {
            "xy": {"left": [-.4, .2], "right": [-.4, -.4]}},
        synchronous_pair_coverage=.6, continuous_pair_coverage=.6)
    selected = select_paired_mount([crossing, safe])
    assert selected["xy"] == crossing["mount"]["xy"]


def test_equal_following_quality_prefers_less_projected_crossing():
    crossing = _record(maximum_structural_crossing_m=.12)
    clear = _record(
        mount=_record()["mount"] | {
            "xy": {"left": [-.4, .2], "right": [-.4, -.4]}},
        maximum_structural_crossing_m=.01)
    assert select_paired_mount([crossing, clear])["xy"] == clear["mount"]["xy"]
