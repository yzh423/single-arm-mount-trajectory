import numpy as np

from scripts.rolling_multibranch_ik import (
    BranchCandidate, build_collision_free_pair_layers, select_minimum_retime_path)


def candidate(q, index):
    return BranchCandidate(np.asarray(q, float), True, True, 0., 0., .2, .1, index)


def test_joint_pairing_rejects_independently_preferred_colliding_state():
    left = ((candidate([0.], 0), candidate([1.], 1)),)
    right = ((candidate([0.], 0), candidate([1.], 1)),)
    layers = build_collision_free_pair_layers(
        left_layers=left, right_layers=right,
        state_valid=lambda l, r: not (l[0] == 0 and r[0] == 0))
    path = select_minimum_retime_path(
        layers=layers, initial_q=np.zeros(2), periodic=np.zeros(2, bool),
        dt_s=np.ones(1), velocity_limit_rad_s=np.ones(2) * 2,
        maximum_joint_step_rad=2., maximum_wrist_step_norm_rad=2.)
    assert path.pose_valid[0]
    assert not np.array_equal(path.q[0], [0., 0.])


def test_no_safe_pair_produces_explicit_hold():
    layers = build_collision_free_pair_layers(
        left_layers=((candidate([0.], 0),),),
        right_layers=((candidate([0.], 0),),),
        state_valid=lambda _l, _r: False)
    path = select_minimum_retime_path(
        layers=layers, initial_q=np.array([.2, -.2]),
        periodic=np.zeros(2, bool), dt_s=np.ones(1),
        velocity_limit_rad_s=np.ones(2))
    np.testing.assert_allclose(path.q[0], [.2, -.2])
    assert path.recovery_mode[0] == "hold_no_candidate"


def test_disconnected_safe_pair_uses_bounded_recovery_instead_of_teleport():
    layers = (tuple(), (candidate([1.0], 0),))
    path = select_minimum_retime_path(
        layers=layers, initial_q=np.array([0.]), periodic=np.zeros(1, bool),
        dt_s=np.array([.1, .1]), velocity_limit_rad_s=np.ones(1),
        maximum_joint_step_rad=.2, maximum_wrist_step_norm_rad=.2,
        transition_valid=lambda _old, new: new[0] <= .2 + 1e-12)
    np.testing.assert_allclose(path.q[1], [.2])
    assert not path.pose_valid[1]
    assert path.recovery_mode[1] == "limited_step"
