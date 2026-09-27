import numpy as np
import pytest

from scripts import build_piperx_v4_mount_optimization_report as report


def test_mount_comparison_preserves_source_scope_and_counts_gain():
    baseline = {
        "source_sha256": "a" * 64,
        "controller_event_rows": 755,
        "both_accept_frames": 716,
    }
    optimized = {
        "source_sha256": "a" * 64,
        "controller_event_rows": 755,
        "both_accept_frames": 739,
    }

    comparison = report.mount_comparison(baseline, optimized)

    assert comparison["accepted_gain"] == 23
    assert comparison["coverage_gain_percentage_points"] == pytest.approx(
        100 * 23 / 755)


def test_mount_comparison_rejects_different_recording():
    baseline = {
        "source_sha256": "a" * 64,
        "controller_event_rows": 755,
        "both_accept_frames": 716,
    }
    optimized = {
        **baseline,
        "source_sha256": "b" * 64,
    }

    with pytest.raises(ValueError, match="same source"):
        report.mount_comparison(baseline, optimized)


def test_comparison_rejects_any_target_or_time_change():
    original = {
        "source_time_s": np.asarray([0.0, .02]),
        "source_poll_row_index": np.asarray([0, 2]),
        "left_target_position_m": np.zeros((2, 3)),
        "right_target_position_m": np.zeros((2, 3)),
        "left_target_quaternion_wxyz": np.tile([1., 0., 0., 0.], (2, 1)),
        "right_target_quaternion_wxyz": np.tile([1., 0., 0., 0.], (2, 1)),
    }
    changed = {key: value.copy() for key, value in original.items()}
    changed["right_target_position_m"][1, 0] = .001

    with pytest.raises(ValueError, match="target"):
        report.validate_identical_targets(original, changed)


def test_optimized_video_interpolation_has_no_safety_violation():
    _, arrays, scene, _ = report._read_shard(report.OPTIMIZED_ROOT)

    result = report.audit_video_interpolation_safety(scene, arrays)

    assert result["frame_count"] == 530
    assert result["invalid_collision_or_clearance_frames"] == 0
    assert result["invalid_topology_frames"] == 0
    assert result["minimum_named_clearance_m"] >= .015
