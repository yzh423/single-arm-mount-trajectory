import numpy as np
import pytest

from factory_bimanual.fixed_time_evidence_audit import (
    compare_target_tracks,
    pose_tracking_errors,
    strict_pair_accept,
    taskspace_segment_rates,
)
from scripts.audit_piperx_fixed_time_evidence import _aggregate


def test_quaternion_sign_does_not_create_false_angular_motion():
    time_s = np.asarray([0.0, 0.5, 1.0])
    position = np.zeros((3, 3))
    quaternion = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])

    rates = taskspace_segment_rates(time_s, position, quaternion)

    np.testing.assert_allclose(rates.linear_speed_m_s, 0.0)
    np.testing.assert_allclose(rates.angular_speed_rad_s, 0.0)


def test_taskspace_rates_use_source_intervals_without_retiming():
    time_s = np.asarray([2.0, 2.5, 3.5])
    position = np.asarray([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [1.0, 2.0, 0.0]])
    half = np.sqrt(0.5)
    quaternion = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [half, 0.0, 0.0, half],
                             [0.0, 0.0, 0.0, 1.0]])

    rates = taskspace_segment_rates(time_s, position, quaternion)

    np.testing.assert_allclose(rates.interval_s, [0.5, 1.0])
    np.testing.assert_allclose(rates.linear_speed_m_s, [2.0, 2.0])
    np.testing.assert_allclose(
        rates.angular_speed_rad_s, [np.pi, np.pi / 2.0])


def test_taskspace_rates_reject_non_increasing_timeline():
    with pytest.raises(ValueError, match="strictly increasing"):
        taskspace_segment_rates(
            [0.0, 0.0], np.zeros((2, 3)),
            [[1.0, 0.0, 0.0, 0.0]] * 2)


def test_pose_tracking_errors_measure_raw_target_not_solver_target():
    target_position = np.asarray([[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0]])
    target_quaternion = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0]])
    actual_pose = np.asarray([
        [0.0005, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0015, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])

    errors = pose_tracking_errors(
        actual_pose, target_position, target_quaternion)

    np.testing.assert_allclose(errors.position_error_m, [0.0005, 0.0015])
    np.testing.assert_allclose(errors.orientation_error_rad, 0.0)


def test_strict_pair_accept_requires_both_hands_and_valid_source():
    left_position = np.asarray([0.0005, 0.0005, 0.0005])
    right_position = np.asarray([0.0005, 0.0015, 0.0005])
    left_orientation = np.deg2rad([0.25, 0.25, 0.25])
    right_orientation = np.deg2rad([0.25, 0.25, 0.75])

    accepted = strict_pair_accept(
        left_position, right_position,
        left_orientation, right_orientation,
        source_valid=[True, True, False],
        position_tolerance_m=0.001,
        orientation_tolerance_rad=np.deg2rad(0.5),
    )

    np.testing.assert_array_equal(accepted, [True, False, False])


def test_compare_target_tracks_reports_conditioning_budget_overrun():
    raw_position = np.zeros((2, 3))
    conditioned_position = raw_position.copy()
    conditioned_position[1, 0] = 0.002
    raw_quaternion = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0, 0.0]])
    conditioned_quaternion = raw_quaternion.copy()

    comparison = compare_target_tracks(
        raw_position, raw_quaternion,
        conditioned_position, conditioned_quaternion,
        position_budget_m=0.001,
        orientation_budget_rad=np.deg2rad(0.5),
    )

    assert comparison.maximum_position_deviation_m == pytest.approx(0.002)
    assert comparison.position_budget_exceeded
    assert not comparison.orientation_budget_exceeded


def test_project_summary_weights_coverage_by_valid_source_frames():
    rows = []
    for mode in ("baseline", "upright_table", "horizontal_wall", "inverted"):
        rows.extend([
            {
                "mode": mode, "valid_pair_frames": 1,
                "conditioned_pair_coverage": 1.0, "raw_pair_coverage": 1.0,
                "solver_target_exceeds_strict_position_tolerance": True,
                "solver_target_exceeds_strict_orientation_tolerance": False,
                "stored_solver_target_matches_current_contract": True,
                "study_dynamics_pass": False,
                "official_ceiling_dynamics_pass": False,
                "collision_frames": 0, "edge_collision_frames": 0,
                "topology_invalid_frames": 0,
                "source_tcp_linear_speed_max_m_s": 2.0,
                "source_tcp_angular_speed_max_rad_s": 3.0,
            },
            {
                "mode": mode, "valid_pair_frames": 9,
                "conditioned_pair_coverage": 0.0, "raw_pair_coverage": 0.0,
                "solver_target_exceeds_strict_position_tolerance": False,
                "solver_target_exceeds_strict_orientation_tolerance": True,
                "stored_solver_target_matches_current_contract": True,
                "study_dynamics_pass": True,
                "official_ceiling_dynamics_pass": True,
                "collision_frames": 0, "edge_collision_frames": 0,
                "topology_invalid_frames": 0,
                "source_tcp_linear_speed_max_m_s": 1.0,
                "source_tcp_angular_speed_max_rad_s": 1.0,
            },
        ])

    summary = _aggregate(rows)

    assert summary["weighted_raw_pair_coverage"] == pytest.approx(0.1)
    assert summary["weighted_conditioned_pair_coverage"] == pytest.approx(0.1)
    assert summary["study_dynamics_pass_shards"] == 4
    assert summary["stored_solver_target_contract_match_shards"] == 8
    assert summary["solver_target_position_budget_exceeded_shards"] == 4
    assert summary["solver_target_orientation_budget_exceeded_shards"] == 4
    assert summary["nonzero_raw_coverage_and_study_dynamics_pass_shards"] == 0
    assert summary["nonzero_raw_coverage_and_official_ceiling_pass_shards"] == 0
