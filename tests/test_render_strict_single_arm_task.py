import numpy as np

from scripts.render_strict_single_arm_task import (
    PLAYBACK_SPEED,
    interpolate_quaternion_wxyz,
    interpolated_failure_reason,
    playback_frame_count,
)


def test_renderer_defaults_to_original_speed():
    assert PLAYBACK_SPEED == 1.0
    assert playback_frame_count(np.array([0.0, 10.0])) == 300


def test_interpolated_failure_reason_uses_the_failed_endpoint():
    success = np.array([True, False, True, False])
    reasons = np.array(["none", "solver_failure", "none", "position"])
    assert interpolated_failure_reason(success, reasons, 0, 1) == "solver_failure"
    assert interpolated_failure_reason(success, reasons, 1, 2) == "solver_failure"
    assert interpolated_failure_reason(success, reasons, 1, 3) == "solver_failure"
    assert interpolated_failure_reason(success, reasons, 0, 2) == "none"


def test_renderer_slerps_tcp_orientation_without_quaternion_sign_flip():
    identity = np.asarray((1.0, 0.0, 0.0, 0.0))
    half_turn_z = np.asarray((0.0, 0.0, 0.0, 1.0))

    halfway = interpolate_quaternion_wxyz(identity, half_turn_z, 0.5)

    assert np.linalg.norm(halfway) == 1.0
    assert abs(float(np.dot(halfway, np.asarray((2**-0.5, 0.0, 0.0, 2**-0.5))))) > 1 - 1e-12
