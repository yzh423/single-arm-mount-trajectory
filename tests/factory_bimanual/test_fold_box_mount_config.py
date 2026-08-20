import numpy as np
import inspect

from scripts.render_factory_dual_xarm6_fold_box import (
    SELECTED_MOUNT,
    bounded_smooth_pose_series,
    classify_ik_failure,
    retime_for_joint_acceleration,
    shared_retimed_intervals,
)
from scripts import render_factory_dual_xarm6_fold_box as fold_box_renderer


def test_fold_box_uses_dynamically_validated_dual_mount():
    np.testing.assert_allclose(
        SELECTED_MOUNT["xy"]["left"],
        (-0.3757733962920817, 0.2812066419065647),
    )
    np.testing.assert_allclose(
        SELECTED_MOUNT["xy"]["right"],
        (-0.29170913284743294, -0.17418189846667936),
    )
    assert SELECTED_MOUNT["yaw"] == {"left": 315.0, "right": 45.0}
    assert SELECTED_MOUNT["shared_base_z_m"] == 0.87
    distance = np.linalg.norm(
        np.asarray(SELECTED_MOUNT["xy"]["left"])
        - np.asarray(SELECTED_MOUNT["xy"]["right"])
    )
    assert np.isclose(distance, 0.4630826309545253)


def test_strict_success_has_no_recovery_failure_reason():
    reason = classify_ik_failure(
        strict_success=True, candidate_count=4,
        intrinsic_edge_feasible=False, collision_only_blocked=False,
        recovery_mode="limited_step", position_error_m=0.0005,
        orientation_error_rad=np.deg2rad(0.5),
        realized_velocity_violation=False,
    )
    assert reason == "ok"


def test_failure_reason_categories_are_specific():
    common = dict(
        strict_success=False, position_error_m=0.01,
        orientation_error_rad=np.deg2rad(3),
        realized_velocity_violation=False, recovery_mode="limited_step",
    )
    assert classify_ik_failure(
        candidate_count=0, intrinsic_edge_feasible=False,
        collision_only_blocked=False, **common) == "POSE UNREACHABLE"
    assert classify_ik_failure(
        candidate_count=4, intrinsic_edge_feasible=False,
        collision_only_blocked=False, **common) == "SOURCE TIMING INFEASIBLE"
    assert classify_ik_failure(
        candidate_count=4, intrinsic_edge_feasible=False,
        collision_only_blocked=True, **common) == "COLLISION BLOCKED"
    assert classify_ik_failure(
        candidate_count=4, intrinsic_edge_feasible=True,
        collision_only_blocked=False, **common) == "RECOVERY PROPAGATION"


def test_multibranch_solver_forwards_requested_horizon():
    source = inspect.getsource(fold_box_renderer.solve_multibranch_single_arm_method)
    assert "horizon=horizon" in source


def test_fold_box_main_uses_global_acceleration_retiming_without_dynamic_projection():
    source = inspect.getsource(fold_box_renderer.run_fold_box)
    assert "solve_multibranch_single_arm_method" in source
    assert "refine_fixed_time_bimanual_path" not in source
    assert "global_retimed=True" in source
    assert "shared_retimed_intervals" in source
    assert "acceleration_smoothed" in fold_box_renderer.OUT.stem
    assert "retime_for_joint_acceleration" in source


def test_fold_box_solver_supports_global_minimum_retiming():
    signature = inspect.signature(
        fold_box_renderer.solve_multibranch_single_arm_method)
    assert "global_retimed" in signature.parameters
    source = inspect.getsource(
        fold_box_renderer.solve_multibranch_single_arm_method)
    assert "select_minimum_retime_path" in source
    assert "path.required_dt_s" in source


def test_fold_box_helpers_accept_robot_contract_name():
    for function in (
        fold_box_renderer.mapped_quaternions_at_joint_midpoint,
        fold_box_renderer.solve_multibranch_single_arm_method,
        fold_box_renderer.audit_bimanual_collisions,
    ):
        assert "robot_name" in inspect.signature(function).parameters
        assert "ROBOT_CONTRACTS[robot_name]" in inspect.getsource(function)
    signature = inspect.signature(fold_box_renderer.run_fold_box)
    assert tuple(signature.parameters) == (
        "robot_name", "selected_mount", "output", "velocity_limit_rad_s")


def test_shared_retimed_intervals_use_both_arms_and_report_added_time():
    result = shared_retimed_intervals(
        source=np.array([.1, .1, .1]),
        left=np.array([.1, .15, .1]),
        right=np.array([.1, .12, .2]),
    )
    np.testing.assert_allclose(result.intervals_s, [.1, .15, .2])
    assert np.isclose(result.added_duration_s, .15)
    assert result.retimed.tolist() == [False, True, True]
    assert np.isclose(result.maximum_edge_addition_s, .1)


def test_bounded_pose_smoothing_reduces_impulse_without_exceeding_pose_caps():
    position = np.zeros(( nine := 9, 3))
    position[4, 0] = .02
    angle = np.zeros(nine)
    angle[4] = np.deg2rad(8)
    quaternion = np.column_stack((np.cos(angle / 2), np.zeros((nine, 2)),
                                  np.sin(angle / 2)))
    smoothed_position, smoothed_quaternion = bounded_smooth_pose_series(
        position, quaternion, position_cap_m=.003,
        orientation_cap_rad=np.deg2rad(1), sigma_frames=1.0)
    displacement = np.linalg.norm(smoothed_position - position, axis=1)
    orientation_change = 2 * np.arccos(np.clip(np.abs(np.sum(
        smoothed_quaternion * quaternion, axis=1)), 0.0, 1.0))
    assert displacement.max() <= .003 + 1e-12
    assert orientation_change.max() <= np.deg2rad(1) + 1e-12
    assert np.abs(np.diff(smoothed_position[:, 0], n=2)).max() < .04


def test_acceleration_retiming_only_adds_time_and_respects_limit():
    q = np.asarray([[0.0], [.3], [0.0]])
    source = np.asarray([.1, .1, .1])
    result = retime_for_joint_acceleration(
        q, source, acceleration_limit_rad_s2=10.0)
    assert np.all(result.intervals_s >= source)
    velocity = np.diff(q, axis=0) / result.intervals_s[1:, None]
    acceleration = 2 * np.diff(velocity, axis=0) / (
        result.intervals_s[1:-1, None] + result.intervals_s[2:, None])
    assert np.abs(acceleration).max() <= 10.0 + 1e-6


def test_fold_box_video_interpolates_joint_states():
    source = inspect.getsource(fold_box_renderer.run_fold_box)
    assert "interpolate_states=True" in source
    assert "task = replace(" in source
    assert '"pose_smoothing"' in source
    assert 'reconstructed_target_quaternion' in source
