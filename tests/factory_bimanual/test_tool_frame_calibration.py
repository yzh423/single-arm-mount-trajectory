import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from factory_bimanual.tool_frame_calibration import (
    CalibrationArtifact,
    apply_fixed_tool_rotation,
    fixed_offset_quaternion,
    proper_axis_rotations,
    rank_calibration_result,
    representative_quaternion_indices,
    validate_local_refinement_deg,
)
import scripts.render_factory_dual_xarm6_se3_follow as follower
import factory_bimanual.tool_frame_calibration as tool_frame


def _quat_z(degrees):
    angle = np.deg2rad(degrees) / 2.0
    return np.asarray([np.cos(angle), 0.0, 0.0, np.sin(angle)])


def test_proper_axis_rotations_are_the_24_right_handed_signed_permutations():
    rotations = proper_axis_rotations()

    assert rotations.shape == (24, 3, 3)
    assert len({tuple(matrix.ravel()) for matrix in rotations}) == 24
    for matrix in rotations:
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0)
        assert set(np.unique(matrix)).issubset({-1.0, 0.0, 1.0})


def test_fixed_tool_rotation_preserves_relative_rotation_without_frame_zero_anchor():
    offset = _quat_z(90.0)
    source_a = np.asarray([_quat_z(0.0), _quat_z(30.0), _quat_z(70.0)])
    source_b = np.asarray([_quat_z(150.0), _quat_z(30.0), _quat_z(70.0)])

    mapped_a = apply_fixed_tool_rotation(source_a, offset)
    mapped_b = apply_fixed_tool_rotation(source_b, offset)

    np.testing.assert_allclose(
        np.abs(mapped_a[1:]), np.abs(mapped_b[1:]), atol=1e-12)
    source_step = 2.0 * np.arccos(np.clip(abs(np.dot(
        source_a[1], source_a[2])), 0.0, 1.0))
    mapped_step = 2.0 * np.arccos(np.clip(abs(np.dot(
        mapped_a[1], mapped_a[2])), 0.0, 1.0))
    assert mapped_step == pytest.approx(source_step)


def test_fixed_tool_translation_rotates_one_constant_local_offset_per_frame():
    positions = np.zeros((2, 3), dtype=float)
    source = np.asarray([_quat_z(0.0), _quat_z(90.0)])

    mapped = tool_frame.apply_fixed_tool_translation(
        positions, source, [1.0, 0.0, 0.0])

    np.testing.assert_allclose(
        mapped, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], atol=1e-12)
    np.testing.assert_array_equal(
        tool_frame.apply_fixed_tool_translation(
            positions, source, [0.0, 0.0, 0.0]),
        positions,
    )


def test_bounded_wrist_adaptation_holds_then_returns_to_zero():
    source = np.repeat(_quat_z(0.0)[None], 4, axis=0)
    time_s = np.asarray([0.0, 0.5, 1.0, 1.5])
    spec = SimpleNamespace(
        axis="x", angle_deg=-10.0,
        hold_until_s=0.5, return_until_s=1.5,
    )

    mapped, angle_deg = tool_frame.apply_bounded_wrist_adaptation(
        source, time_s, spec)

    np.testing.assert_allclose(angle_deg, [-10.0, -10.0, -5.0, 0.0])
    relative = Rotation.from_quat(mapped[:, [1, 2, 3, 0]]).as_euler(
        "xyz", degrees=True)
    np.testing.assert_allclose(relative[:, 0], angle_deg, atol=1e-12)


def test_local_refinement_is_limited_to_fifteen_degrees_per_axis():
    np.testing.assert_allclose(
        validate_local_refinement_deg([15.0, -15.0, 0.0]),
        [15.0, -15.0, 0.0])
    with pytest.raises(ValueError, match="15"):
        validate_local_refinement_deg([15.01, 0.0, 0.0])


def test_axis_mapping_and_local_refinement_compose_to_one_unit_quaternion():
    base = fixed_offset_quaternion(4, [0.0, 0.0, 0.0])
    refined = fixed_offset_quaternion(4, [10.0, -5.0, 2.5])
    assert np.linalg.norm(base) == pytest.approx(1.0)
    assert np.linalg.norm(refined) == pytest.approx(1.0)
    assert not np.allclose(np.abs(base), np.abs(refined))
    with pytest.raises(ValueError, match="index"):
        fixed_offset_quaternion(24, [0.0, 0.0, 0.0])


def test_representative_quaternion_indices_are_bounded_and_cover_motion():
    source = np.asarray([_quat_z(value) for value in
                         (0, 1, 2, 3, 40, 41, 80, 81, 120, 121)])
    first = representative_quaternion_indices(source, maximum=6)
    second = representative_quaternion_indices(source, maximum=6)
    assert np.array_equal(first, second)
    assert first[0] == 0 and first[-1] == len(source) - 1
    assert len(first) == 6
    assert 4 in first or 6 in first or 8 in first


def test_calibration_ranking_is_coverage_then_failure_run_then_connectivity():
    base = {
        "synchronous_strict_coverage": .75,
        "longest_failure_run_frames": 5,
        "connectable_safe_branch_ratio": .7,
        "minimum_singularity_margin": .1,
        "minimum_joint_limit_margin_rad": .2,
        "mean_normalized_pose_error": .3,
    }
    assert rank_calibration_result(
        base | {"synchronous_strict_coverage": .8,
                "longest_failure_run_frames": 20}) < \
        rank_calibration_result(base)
    assert rank_calibration_result(
        base | {"longest_failure_run_frames": 2}) < \
        rank_calibration_result(base)
    assert rank_calibration_result(
        base | {"connectable_safe_branch_ratio": .8}) < \
        rank_calibration_result(base)


def test_calibration_artifact_is_cross_task_fixed_and_round_trips(tmp_path):
    artifact = CalibrationArtifact(
        version=1,
        robot="piperx",
        tasks=("fold_box", "seal_bag"),
        source_fingerprints={"fold_box": "fold-hash", "seal_bag": "seal-hash"},
        left_offset_quaternion_wxyz=tuple(_quat_z(90.0)),
        right_offset_quaternion_wxyz=tuple(_quat_z(-90.0)),
        left_axis_rotation_index=3,
        right_axis_rotation_index=7,
        left_local_xyz_deg=(5.0, -2.5, 0.0),
        right_local_xyz_deg=(-5.0, 0.0, 2.5),
        metrics={"synchronous_strict_coverage": 0.8},
    )
    path = tmp_path / "calibration.json"
    artifact.write(path)
    loaded = CalibrationArtifact.read(
        path,
        expected_source_fingerprints={
            "fold_box": "fold-hash", "seal_bag": "seal-hash"},
    )

    assert loaded == artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tasks"] == ["fold_box", "seal_bag"]
    with pytest.raises(ValueError, match="fingerprint"):
        CalibrationArtifact.read(
            path,
            expected_source_fingerprints={
                "fold_box": "changed", "seal_bag": "seal-hash"},
        )


def test_prepared_calibrated_targets_bypass_framewise_orientation_rewrite(
        monkeypatch):
    source = np.asarray([_quat_z(0.0), _quat_z(20.0), _quat_z(50.0)])
    positions = np.arange(9, dtype=float).reshape(3, 3)
    task = SimpleNamespace(
        time_s=np.asarray([0.0, 0.1, 0.2]),
        left_position_m=positions.copy(),
        right_position_m=-positions.copy(),
        left_quaternion_wxyz=source.copy(),
        right_quaternion_wxyz=source.copy(),
    )
    artifact = CalibrationArtifact(
        version=1, robot="piperx", tasks=("fold_box", "seal_bag"),
        source_fingerprints={"fold_box": "a", "seal_bag": "b"},
        left_offset_quaternion_wxyz=tuple(_quat_z(90.0)),
        right_offset_quaternion_wxyz=tuple(_quat_z(-90.0)),
        left_axis_rotation_index=0, right_axis_rotation_index=1,
        left_local_xyz_deg=(0.0, 0.0, 0.0),
        right_local_xyz_deg=(0.0, 0.0, 0.0), metrics={},
    )
    monkeypatch.setattr(
        follower, "reconstruct_held_quaternions",
        lambda *a, **k: pytest.fail("calibrated orientation must not reconstruct"))
    monkeypatch.setattr(
        follower, "bounded_smooth_pose_series",
        lambda *a, **k: pytest.fail("calibrated pose must not be smoothed"))

    prepared, mapped = follower.prepare_follow_targets(
        None, task, calibration=artifact)

    np.testing.assert_array_equal(prepared.left_position_m, positions)
    np.testing.assert_array_equal(prepared.right_position_m, -positions)
    np.testing.assert_allclose(
        np.abs(mapped["left"]),
        np.abs(apply_fixed_tool_rotation(source, _quat_z(90.0))))
