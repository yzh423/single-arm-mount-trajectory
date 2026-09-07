import numpy as np

from factory_bimanual.fixed_time_tracking import (
    bounded_joint_step,
    fixed_time_derivatives,
)


def test_bounded_joint_step_enforces_velocity_and_acceleration_together():
    result = bounded_joint_step(
        previous_q=np.array([0.0]),
        desired_q=np.array([1.0]),
        previous_velocity_rad_s=np.array([0.0]),
        dt_s=0.1,
        previous_dt_s=0.1,
        velocity_limit_rad_s=1.0,
        acceleration_limit_rad_s2=2.0,
    )

    np.testing.assert_allclose(result.velocity_rad_s, [0.2])
    np.testing.assert_allclose(result.q, [0.02])
    assert result.limited


def test_bounded_joint_step_decelerates_instead_of_instant_hold():
    result = bounded_joint_step(
        previous_q=np.array([0.5]),
        desired_q=np.array([0.5]),
        previous_velocity_rad_s=np.array([0.4]),
        dt_s=0.1,
        previous_dt_s=0.1,
        velocity_limit_rad_s=1.0,
        acceleration_limit_rad_s2=2.0,
    )

    np.testing.assert_allclose(result.velocity_rad_s, [0.2])
    np.testing.assert_allclose(result.q, [0.52])
    assert result.limited


def test_bounded_joint_step_preserves_reachable_desired_state():
    result = bounded_joint_step(
        previous_q=np.array([0.5, -0.2]),
        desired_q=np.array([0.51, -0.19]),
        previous_velocity_rad_s=np.array([0.1, 0.1]),
        dt_s=0.1,
        previous_dt_s=0.1,
        velocity_limit_rad_s=3.0,
        acceleration_limit_rad_s2=5.0,
    )

    np.testing.assert_allclose(result.q, [0.51, -0.19])
    assert not result.limited


def test_bounded_joint_step_respects_physical_joint_range():
    result = bounded_joint_step(
        previous_q=np.array([0.99]),
        desired_q=np.array([1.2]),
        previous_velocity_rad_s=np.array([0.1]),
        dt_s=0.1,
        previous_dt_s=0.1,
        velocity_limit_rad_s=3.0,
        acceleration_limit_rad_s2=5.0,
        lower_rad=np.array([-1.0]),
        upper_rad=np.array([1.0]),
    )

    assert .99 < result.q[0] < 1.0
    assert 0.0 < result.velocity_rad_s[0] < .1
    assert result.limited


def test_fixed_time_derivatives_match_interval_command_contract():
    q = np.asarray([[0.0], [0.02], [0.06]])

    velocity, acceleration = fixed_time_derivatives(
        q, np.asarray([0.0, 0.1, 0.2]), initial_velocity_rad_s=0.0)

    np.testing.assert_allclose(velocity[:, 0], [0.0, 0.2, 0.4])
    np.testing.assert_allclose(acceleration[:, 0], [0.0, 2.0, 2.0])


def test_joint_range_adds_braking_viability_velocity_cap():
    result = bounded_joint_step(
        previous_q=np.array([0.9]),
        desired_q=np.array([1.0]),
        previous_velocity_rad_s=np.array([0.4]),
        dt_s=0.1,
        previous_dt_s=0.1,
        velocity_limit_rad_s=3.0,
        acceleration_limit_rad_s2=1.0,
        lower_rad=np.array([-1.0]),
        upper_rad=np.array([1.0]),
    )

    expected_cap = -0.1 + np.sqrt(0.21)
    assert result.velocity_rad_s[0] <= expected_cap + 1e-12
    assert result.q[0] < 0.94

    braking = bounded_joint_step(
        previous_q=result.q,
        desired_q=result.q,
        previous_velocity_rad_s=result.velocity_rad_s,
        dt_s=0.1,
        previous_dt_s=0.1,
        velocity_limit_rad_s=3.0,
        acceleration_limit_rad_s2=1.0,
        lower_rad=np.array([-1.0]),
        upper_rad=np.array([1.0]),
    )
    assert braking.q[0] <= 1.0
