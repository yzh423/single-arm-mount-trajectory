from types import SimpleNamespace

import numpy as np

from factory_bimanual.bimanual_collision import CallbackCollisionChecker, CollisionClass, CollisionReport
from factory_bimanual.strict_bimanual_ik import (
    BimanualIKConfig,
    IKCandidate,
    solve_strict_bimanual_path,
)


def _task(times):
    n = len(times)
    return SimpleNamespace(time_s=np.asarray(times, dtype=float), source_row_indices=np.arange(10, 10 + n))


def _candidate(q, cost=0.0, branch=0):
    return IKCandidate(q=np.asarray([q], dtype=float), branch_index=branch, pose_cost=cost)


def test_colliding_branch_pair_is_not_selected():
    candidates = lambda _m, _c, _t, _row, side: [_candidate(0, 0, 0), _candidate(1, 1, 1)]
    collision = CallbackCollisionChecker(
        pair_evaluator=lambda left, right: CollisionReport(
            classes=(CollisionClass.CROSS_ARM,) if left[0] == 0 and right[0] == 0 else ()
        )
    )
    result = solve_strict_bimanual_path(None, None, _task([0.0]), BimanualIKConfig(candidates, collision))
    assert result.success.tolist() == [True]
    assert result.paired_branch_indices[0] != (0, 0)


def test_future_safe_pair_beats_lower_cost_greedy_pair():
    def candidates(_m, _c, _t, row, _side):
        if row == 0:
            return [_candidate(0, 0.0, 0), _candidate(1, 0.2, 1)]
        return [_candidate(1, 0.0, 1)]

    # A transition from branch 0 to branch 1 is blocked; keeping branch 1 is safe.
    collision = CallbackCollisionChecker(
        transition_evaluator=lambda previous, current: CollisionReport(
            classes=(CollisionClass.CROSS_ARM,)
            if previous[0][0] == 0 and current[0][0] == 1
            else ()
        )
    )
    config = BimanualIKConfig(candidates, collision, beam_width=4, max_velocity_rad_s=100.0)
    result = solve_strict_bimanual_path(None, None, _task([0.0, 1.0]), config)
    assert result.success.tolist() == [True, True]
    assert result.paired_branch_indices == [(1, 1), (1, 1)]


def test_velocity_limit_uses_source_timestamps():
    candidates = lambda _m, _c, _t, row, _side: [_candidate(float(row), branch=row)]
    config = BimanualIKConfig(candidates, CallbackCollisionChecker(), max_velocity_rad_s=2.0)

    slow = solve_strict_bimanual_path(None, None, _task([0.0, 1.0]), config)
    fast = solve_strict_bimanual_path(None, None, _task([0.0, 0.1]), config)

    assert slow.success.tolist() == [True, True]
    assert fast.success.tolist() == [True, False]
    assert fast.failures[1] == "velocity_jump_violation"


def test_acceleration_limit_uses_adjacent_source_intervals():
    positions = [0.0, 1.0, 3.0]
    candidates = lambda _m, _c, _t, row, _side: [
        _candidate(positions[row], branch=row)]
    config = BimanualIKConfig(
        candidates, CallbackCollisionChecker(),
        max_velocity_rad_s=10.0, max_acceleration_rad_s2=0.5)

    result = solve_strict_bimanual_path(
        None, None, _task([0.0, 1.0, 2.0]), config)

    assert result.success.tolist() == [True, True, False]
    assert result.failures[2] == "acceleration_violation"


def test_failures_and_outputs_remain_aligned_to_source_rows():
    def candidates(_m, _c, _t, row, _side):
        return [] if row == 1 else [_candidate(row, branch=row)]

    result = solve_strict_bimanual_path(
        None, None, _task([0.0, 1.0, 2.0]), BimanualIKConfig(candidates, CallbackCollisionChecker())
    )
    assert result.source_row_indices.tolist() == [10, 11, 12]
    assert len(result.left_q) == len(result.right_q) == len(result.failures) == 3
    assert result.success.tolist() == [True, False, True]
    assert result.failures[1] == "branch_lost"
    assert np.isnan(result.left_q[1]).all()


def test_wide_limited_joint_transition_does_not_wrap_across_hard_stops():
    values = (np.pi - .01, -np.pi + .01)
    candidates = lambda _m, _c, _t, row, _side: [_candidate(values[row], branch=row)]
    config = BimanualIKConfig(
        candidates, CallbackCollisionChecker(), max_velocity_rad_s=.25,
        max_jump_rad=.025, periodic_joints=np.asarray((False,)),
    )
    result = solve_strict_bimanual_path(None, None, _task([0.0, 0.1]), config)
    assert result.success.tolist() == [True, False]
    assert result.failures[1] == "velocity_jump_violation"
