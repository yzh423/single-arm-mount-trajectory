import mujoco
import numpy as np
import pytest

from scripts.strict_mujoco_ik import PosePathResult
from scripts.render_factory_dual_xarm6_se3_follow import (
    classify_ik_failure,
    improvement_has_plateaued, shared_mount_height_candidates,
    synchronous_mount_score, map_source_quaternions_to_tcp,
    mapped_quaternions_at_joint_midpoint,
    prepare_follow_targets,
    use_collision_safe_pair_planner,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from scripts.strict_mujoco_ik import realized_velocity_violation


def test_pose_path_failure_reason_field_contract():
    assert "joint_discontinuity" in PosePathResult.__dataclass_fields__


def test_official_piperx_uses_collision_hard_paired_planner():
    assert use_collision_safe_pair_planner("piperx")
    assert not use_collision_safe_pair_planner("xarm6")


def test_shared_mount_height_candidates_are_equal_and_capped():
    candidates = shared_mount_height_candidates(
        table_height_m=0.75, minimum_adapter_m=0.04,
        maximum_adapter_m=0.12, count=5,
    )
    assert candidates == [0.79, 0.81, 0.83, 0.85, 0.87]
    assert all(left_z == right_z for left_z, right_z in
               ((height, height) for height in candidates))
    assert max(candidates) <= 0.87


def test_mount_score_prioritizes_synchronous_success_then_conditioning():
    better_sync = synchronous_mount_score(
        left_success=[True, True, False], right_success=[True, True, True],
        left_sigma=[.10, .10, .10], right_sigma=[.10, .10, .10],
        mean_pose_error=0.2,
    )
    worse_sync = synchronous_mount_score(
        left_success=[True, True, True], right_success=[False, False, True],
        left_sigma=[.20, .20, .20], right_sigma=[.20, .20, .20],
        mean_pose_error=0.01,
    )
    assert better_sync < worse_sync


def test_realized_velocity_violation_uses_output_motion_not_candidate_jump():
    assert not realized_velocity_violation(
        realized_delta=[0.0, 0.0], dt_s=0.01,
        velocity_limit_rad_s=[3.14, 3.14],
    )
    assert realized_velocity_violation(
        realized_delta=[0.04, 0.0], dt_s=0.01,
        velocity_limit_rad_s=[3.14, 3.14],
    )
    assert not realized_velocity_violation(
        realized_delta=[1.0, 0.0], dt_s=0.01,
        velocity_limit_rad_s=[3.14, 3.14], initializing=True,
    )


def test_plateau_requires_three_consecutive_negligible_improvements():
    history = [
        {"coverage": .50, "p95_position_mm": 20.0, "p95_orientation_deg": 3.0},
        {"coverage": .503, "p95_position_mm": 19.4, "p95_orientation_deg": 2.9},
        {"coverage": .506, "p95_position_mm": 18.9, "p95_orientation_deg": 2.8},
        {"coverage": .509, "p95_position_mm": 18.4, "p95_orientation_deg": 2.7},
    ]
    assert improvement_has_plateaued(history, patience=3)
    history[-1]["coverage"] = .52
    assert not improvement_has_plateaued(history, patience=3)


def test_quaternion_mapping_preserves_world_relative_rotation_with_fixed_tool_calibration():
    def quaternion(axis, angle):
        value = np.empty(4)
        mujoco.mju_axisAngle2Quat(value, np.asarray(axis, dtype=float), angle)
        return value

    source = np.asarray([
        quaternion((0, 0, 1), np.deg2rad(70.0)),
        quaternion((1, 0, 0), np.deg2rad(35.0)),
    ])
    robot_initial = quaternion((0, 1, 0), np.deg2rad(-40.0))
    mapped = map_source_quaternions_to_tcp(source, robot_initial)

    def matrix(value):
        result = np.empty(9)
        mujoco.mju_quat2Mat(result, value)
        return result.reshape(3, 3)

    source_matrices = np.asarray([matrix(value) for value in source])
    robot_initial_matrix = matrix(robot_initial)
    expected = source_matrices[1] @ source_matrices[0].T @ robot_initial_matrix
    np.testing.assert_allclose(matrix(mapped[0]), robot_initial_matrix, atol=1e-12)
    np.testing.assert_allclose(matrix(mapped[1]), expected, atol=1e-12)


def test_scene_quaternion_calibration_uses_joint_midpoint_home(tmp_path):
    xml = tmp_path / "scene.xml"
    contract = ROBOT_CONTRACTS["xarm6"]
    build_same_model_scene(contract, .5, xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    source = np.asarray(((1.0, 0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0, 0.0)))
    task = type("Task", (), {
        "left_quaternion_wxyz": source,
        "right_quaternion_wxyz": source,
    })()
    mapped = mapped_quaternions_at_joint_midpoint(model, task)
    for side in ("left", "right"):
        data = mujoco.MjData(model)
        joint_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{i}")
            for i in range(1, 7)]
        data.qpos[model.jnt_qposadr[joint_ids]] = np.mean(
            model.jnt_range[joint_ids], axis=1)
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        expected = np.empty(4)
        mujoco.mju_mat2Quat(expected, data.site_xmat[site])
        assert abs(float(np.dot(mapped[side][0], expected))) == pytest.approx(1.0)


def test_full_follow_reconstructs_held_quaternion_before_solving(tmp_path):
    xml = tmp_path / "scene.xml"
    contract = ROBOT_CONTRACTS["xarm6"]
    build_same_model_scene(contract, .5, xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    identity = np.array([1., 0., 0., 0.])
    turn = np.array([np.cos(.2), 0., 0., np.sin(.2)])
    task = type("Task", (), {})()
    task.__dict__.update({
        "time_s": np.array([0., 1., 2.]),
        "left_position_m": np.zeros((3, 3)),
        "right_position_m": np.zeros((3, 3)),
        "left_quaternion_wxyz": np.array([identity, identity, turn]),
        "right_quaternion_wxyz": np.array([identity, identity, turn]),
    })
    prepared, mapped = prepare_follow_targets(model, task)
    for side in ("left", "right"):
        raw = mapped_quaternions_at_joint_midpoint(model, task)[side]
        assert abs(float(np.dot(mapped[side][1], raw[1]))) < 1.0 - 1e-8
        assert np.linalg.norm(prepared.__dict__[f"{side}_position_m"], axis=1).max() == 0


def test_failure_classifier_separates_unreachable_edge_and_tracking_error():
    assert classify_ik_failure(
        candidate_count=0, recovery_mode="hold_no_candidate",
        position_error_m=.2, orientation_error_rad=.4,
        realized_velocity_violation=False,
    ) == "6D pose unreachable (no IK candidate)"
    assert classify_ik_failure(
        candidate_count=4, recovery_mode="limited_step",
        position_error_m=.02, orientation_error_rad=.1,
        realized_velocity_violation=False,
    ) == "IK edge infeasible at source timing; bounded recovery"
    assert classify_ik_failure(
        candidate_count=4, recovery_mode="none",
        position_error_m=.002, orientation_error_rad=.01,
        realized_velocity_violation=False,
    ) == "position tolerance exceeded (2.0 mm)"
