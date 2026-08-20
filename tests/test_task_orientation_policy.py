import numpy as np

from scripts.strict_trajectory_sources import task_anchor_rotations


def test_task_orientation_uses_fixed_hand_to_tcp_calibration_without_clamping():
    quaternions = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.92387953, 0.38268343, 0.0, 0.0],
    ])

    rotations = task_anchor_rotations("cap-left", quaternions)

    np.testing.assert_allclose(rotations[0, :, 2], [0.0, -1.0, 0.0], atol=1e-8)
    # The recorded 45-degree relative rotation remains present; it is not
    # projected back onto a task-specific horizontal plane.
    relative = rotations[0].T @ rotations[1]
    assert not np.allclose(relative, np.eye(3))
    np.testing.assert_allclose(np.trace(relative), 1.0 + 2.0 * np.cos(np.deg2rad(45.0)), atol=1e-7)
