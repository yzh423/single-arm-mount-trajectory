import numpy as np
import pytest

from factory_bimanual.rescue_v31 import (
    StrictGate,
    minimum_jerk_transition,
    shortest_joint_delta,
    trapezoidal_transition_time,
    wrist_branch_signature,
)


@pytest.mark.parametrize(
    "position_error_m,orientation_error_deg,joint_delta_rad,expected",
    [
        (0.001000, 0.500, 0.300, True),
        (0.001001, 0.500, 0.300, False),
        (0.001000, 0.501, 0.300, False),
        (0.001000, 0.500, 0.301, False),
    ],
)
def test_strict_gate_boundaries(
    position_error_m,
    orientation_error_deg,
    joint_delta_rad,
    expected,
):
    gate = StrictGate(0.001, np.deg2rad(0.5), 0.30)
    delta = np.array([joint_delta_rad, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert gate.accepts(
        position_error_m,
        np.deg2rad(orientation_error_deg),
        delta,
    ) is expected


def test_strict_gate_rejects_nonfinite_values():
    gate = StrictGate(0.001, np.deg2rad(0.5), 0.30)
    assert not gate.accepts(np.nan, 0.0, np.zeros(6))
    assert not gate.accepts(0.0, np.inf, np.zeros(6))
    assert not gate.accepts(0.0, 0.0, np.full(6, np.nan))


def test_shortest_joint_delta_wraps_only_periodic_joints():
    old = np.array([np.pi - 0.05, np.pi - 0.05])
    new = np.array([-np.pi + 0.05, -np.pi + 0.05])
    delta = shortest_joint_delta(new, old, np.array([True, False]))
    assert np.isclose(delta[0], 0.10)
    assert np.isclose(delta[1], -2 * np.pi + 0.10)


def test_wrist_signature_ignores_small_zero_crossing_but_detects_flip():
    near_zero_a = np.array([0, 0, 0, 0.04, -0.03, 0], dtype=float)
    near_zero_b = np.array([0, 0, 0, -0.04, 0.03, 0], dtype=float)
    flipped = np.array([0, 0, 0, -0.8, 0.9, 0], dtype=float)
    assert wrist_branch_signature(near_zero_a) == wrist_branch_signature(near_zero_b)
    assert wrist_branch_signature(near_zero_a) != wrist_branch_signature(flipped)


def test_trapezoidal_time_matches_triangular_and_cruise_cases():
    triangular = trapezoidal_transition_time(
        np.array([0.16]), velocity_rad_s=1.0, acceleration_rad_s2=4.0,
        settle_s=0.1, decision_s=0.015,
    )
    cruise = trapezoidal_transition_time(
        np.array([1.25]), velocity_rad_s=1.0, acceleration_rad_s2=4.0,
        settle_s=0.1, decision_s=0.015,
    )
    assert np.isclose(triangular, 2 * np.sqrt(0.16 / 4.0) + 0.115)
    assert np.isclose(cruise, 0.5 + (1.25 - 0.25) + 0.115)


def test_minimum_jerk_hits_recovery_branch_with_zero_seam():
    q0 = np.zeros(6)
    q1 = np.array([0.8, -0.2, 0.1, 1.2, -1.1, 0.3])
    path = minimum_jerk_transition(q0, q1, 19)
    assert np.array_equal(path[0], q0)
    assert np.array_equal(path[-1], q1)
    assert np.max(np.abs(path[-1] - q1)) <= 1e-12
    assert np.all(np.diff(path[:, 0]) >= 0)


@pytest.mark.parametrize("frames", [0, 1])
def test_minimum_jerk_requires_start_and_end_frames(frames):
    with pytest.raises(ValueError, match="at least two"):
        minimum_jerk_transition(np.zeros(6), np.ones(6), frames)
