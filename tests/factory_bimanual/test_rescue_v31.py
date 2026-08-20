import numpy as np
import pytest

from factory_bimanual.rescue_v31 import (
    CandidateFrame,
    RescueScheduleConfig,
    StrictGate,
    minimum_jerk_transition,
    schedule_rescue_v31,
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


PERIODIC = np.zeros(6, dtype=bool)


def candidate(
    value=0.0,
    *,
    branch=0,
    wrist4=None,
    wrist5=0.0,
    position_error_m=0.0002,
    orientation_error_deg=0.1,
    wrist_risk=0.0,
    joint_margin=0.5,
):
    q = np.full(6, float(value))
    q[3] = float(value if wrist4 is None else wrist4)
    q[4] = float(wrist5)
    return CandidateFrame(
        q=q,
        branch_index=branch,
        position_error_m=position_error_m,
        orientation_error_rad=np.deg2rad(orientation_error_deg),
        wrist_risk=wrist_risk,
        joint_limit_margin_rad=joint_margin,
    )


def schedule_config(**changes):
    values = {
        "gate": StrictGate(0.001, np.deg2rad(0.5), 0.30),
        "velocity_rad_s": 1.0,
        "acceleration_rad_s2": 4.0,
        "settle_s": 0.1,
        "decision_s": 0.015,
        "source_rate_hz": 60.0,
        "dwell_frames": 18,
        "lookahead_frames": 120,
        "early_trigger_frames": 0,
    }
    values.update(changes)
    return RescueScheduleConfig(**values)


def test_nonterminal_hold_resumes_from_last_command():
    layers = [(candidate(0.0),), (), (candidate(0.1),)]
    result = schedule_rescue_v31(
        layers,
        np.arange(3) / 60.0,
        PERIODIC,
        schedule_config(dwell_frames=99),
    )
    assert result.state.tolist() == ["FOLLOW", "HOLD", "FOLLOW"]
    assert np.allclose(result.command_q[1], result.command_q[0])
    assert result.source_accepted.tolist() == [True, False, True]


def test_mode_a_intercepts_future_target_and_counts_drops():
    layers = [(candidate(0.0),), ()]
    layers.extend([()] * 48)
    layers.append((candidate(0.0, branch=7, wrist4=0.4),))
    layers.extend([(candidate(0.0, branch=7, wrist4=0.4),)] * 4)

    result = schedule_rescue_v31(
        layers,
        np.arange(len(layers)) / 60.0,
        PERIODIC,
        schedule_config(dwell_frames=0),
    )

    event = result.events[0]
    assert event.mode == "A"
    assert event.start_frame == 1
    assert event.end_frame == 50
    assert np.array_equal(result.source_command_q[event.end_frame], event.recovery_q)
    assert event.seam_error_rad <= 1e-12
    assert result.dropped_source_frames == event.transition_frames == 49
    assert result.cycle_delay_s == 0.0


def test_mode_b_preserves_source_frames_and_adds_cycle_delay():
    layers = [
        (candidate(0.0),),
        (candidate(0.0, branch=8, wrist4=0.4),),
        (candidate(0.0, branch=8, wrist4=0.42),),
    ]
    result = schedule_rescue_v31(
        layers,
        np.arange(len(layers)) / 60.0,
        PERIODIC,
        schedule_config(dwell_frames=0, lookahead_frames=1),
    )
    assert result.events[0].mode == "B"
    assert set(result.source_index.tolist()) == {0, 1, 2}
    assert result.source_accepted.tolist() == [True, True, True]
    assert result.dropped_source_frames == 0
    assert result.cycle_delay_s > 0


def test_dwell_prevents_immediate_second_branch_jump():
    layers = [
        (candidate(0.0),),
        (candidate(0.0, branch=2, wrist4=0.4),),
        (candidate(0.0, branch=3, wrist4=-0.4),),
    ]
    result = schedule_rescue_v31(
        layers,
        np.arange(3) / 60.0,
        PERIODIC,
        schedule_config(dwell_frames=18, lookahead_frames=1),
    )
    assert [event.start_frame for event in result.events] == [1]
    assert not result.source_accepted[2]


def test_follow_never_changes_wrist_branch_without_rescue_event():
    layers = [
        (candidate(0.0),),
        (candidate(0.0, branch=4, wrist4=0.2),),
    ]
    result = schedule_rescue_v31(
        layers,
        np.arange(2) / 60.0,
        PERIODIC,
        schedule_config(dwell_frames=0, lookahead_frames=1),
    )
    assert result.state[1] != "FOLLOW"
    assert len(result.events) == 1
    assert result.events[0].from_wrist_signature != result.events[0].to_wrist_signature


def test_early_trigger_switches_before_current_branch_is_lost():
    layers = [
        (candidate(0.0, branch=0),),
        (
            candidate(0.05, branch=0),
            candidate(0.0, branch=9, wrist4=0.4),
        ),
        (
            candidate(0.10, branch=0),
            candidate(0.0, branch=9, wrist4=0.42),
        ),
        (candidate(0.0, branch=9, wrist4=0.44),),
        (candidate(0.0, branch=9, wrist4=0.46),),
    ]
    result = schedule_rescue_v31(
        layers,
        np.arange(len(layers)) / 60.0,
        PERIODIC,
        schedule_config(
            dwell_frames=0,
            lookahead_frames=4,
            early_trigger_frames=2,
            velocity_rad_s=10.0,
            acceleration_rad_s2=40.0,
            settle_s=0.0,
            decision_s=0.0,
        ),
    )
    assert result.events[0].start_frame == 1
    assert result.events[0].trigger == "early_intercept"


def test_scheduler_rejects_non_monotonic_source_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        schedule_rescue_v31(
            [(candidate(),), (candidate(0.1),)],
            np.array([0.0, 0.0]),
            PERIODIC,
            schedule_config(),
        )


def test_candidate_can_carry_bimanual_wrist_signature_override():
    pair = CandidateFrame(
        q=np.zeros(12), branch_index=101,
        position_error_m=0.0, orientation_error_rad=0.0,
        wrist_signature_value=(0, 0, 1, -1),
    )
    assert pair.wrist_signature == (0, 0, 1, -1)
