import numpy as np

from factory_bimanual.fixed_time_refinement import refine_fixed_time_joint_path


def test_refinement_preserves_time_and_enforces_derivative_limits():
    time_s = np.arange(7, dtype=float) * 0.1
    reference = np.array([[0.0], [0.1], [-0.1], [0.1], [-0.1], [0.1], [0.0]])
    result = refine_fixed_time_joint_path(
        reference, time_s,
        velocity_limit_rad_s=np.array([1.0]),
        acceleration_limit_rad_s2=np.array([2.0]),
        jerk_limit_rad_s3=np.array([20.0]),
        lower_rad=np.array([-1.0]), upper_rad=np.array([1.0]),
    )
    assert result.q.shape == reference.shape
    assert np.array_equal(result.time_s, time_s)
    assert result.success
    assert result.maximum_velocity_rad_s <= 1.0 + 1e-8
    assert result.maximum_acceleration_rad_s2 <= 2.0 + 1e-8
    assert result.maximum_jerk_rad_s3 <= 20.0 + 1e-7
    assert np.max(np.abs(result.q - reference)) > 0.01


def test_refinement_keeps_a_feasible_reference_unchanged():
    time_s = np.arange(6, dtype=float) * 0.1
    reference = (0.2 * time_s)[:, None]
    result = refine_fixed_time_joint_path(
        reference, time_s, velocity_limit_rad_s=1.0,
        acceleration_limit_rad_s2=2.0, jerk_limit_rad_s3=20.0,
        lower_rad=-1.0, upper_rad=1.0,
    )
    np.testing.assert_allclose(result.q, reference, atol=1e-9)
