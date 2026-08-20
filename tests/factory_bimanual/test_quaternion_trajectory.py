import numpy as np
import pytest

from factory_bimanual.quaternion_trajectory import (
    quaternion_poses_equal,
    reconstruct_held_quaternions,
)


def _z_rotation(degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) / 2.0
    return np.asarray((np.cos(half), 0.0, 0.0, np.sin(half)))


def _orientation_difference_degrees(first, second) -> float:
    dot = abs(float(np.dot(first, second)))
    return float(np.rad2deg(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def test_reconstructs_hold_then_jump_at_nonuniform_time_fraction():
    time_s = np.asarray((0.0, 1.0, 3.0))
    quaternion = np.asarray((_z_rotation(0), _z_rotation(0), _z_rotation(30)))

    result = reconstruct_held_quaternions(time_s, quaternion)

    assert result.changed.tolist() == [False, True, False]
    assert _orientation_difference_degrees(result.quaternion_wxyz[1], _z_rotation(10)) < 1e-8
    assert np.isclose(np.rad2deg(result.change_rad[1]), 10.0)
    np.testing.assert_allclose(np.linalg.norm(result.quaternion_wxyz, axis=1), 1.0)


def test_preserves_endpoints_and_constant_rate_rotation():
    time_s = np.asarray((0.0, 1.0, 2.0))
    quaternion = np.asarray((_z_rotation(0), _z_rotation(10), _z_rotation(20)))

    result = reconstruct_held_quaternions(time_s, quaternion)

    assert not result.changed.any()
    np.testing.assert_allclose(result.quaternion_wxyz, quaternion, atol=1e-12)
    np.testing.assert_allclose(result.quaternion_wxyz[[0, -1]], quaternion[[0, -1]])


def test_uses_shortest_quaternion_path_across_sign_change():
    time_s = np.asarray((0.0, 1.0, 2.0))
    quaternion = np.asarray((_z_rotation(0), -_z_rotation(0), -_z_rotation(20)))

    result = reconstruct_held_quaternions(time_s, quaternion)

    assert result.changed[1]
    assert _orientation_difference_degrees(result.quaternion_wxyz[1], _z_rotation(10)) < 1e-8


def test_quaternion_pose_equality_accepts_opposite_signs():
    quaternion = _z_rotation(37)
    assert quaternion_poses_equal(quaternion, -quaternion)
    assert not quaternion_poses_equal(quaternion, _z_rotation(38))


@pytest.mark.parametrize(
    "time_s, quaternion",
    [
        (np.asarray((0.0, 0.0)), np.asarray((_z_rotation(0), _z_rotation(10)))),
        (np.asarray((0.0, 1.0)), np.zeros((3, 4))),
        (np.asarray((0.0, 1.0)), np.asarray((_z_rotation(0), np.zeros(4)))),
    ],
)
def test_rejects_invalid_inputs(time_s, quaternion):
    with pytest.raises(ValueError):
        reconstruct_held_quaternions(time_s, quaternion)
