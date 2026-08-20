from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from factory_bimanual.bimanual_collision import (
    CallbackCollisionChecker, CollisionClass, CollisionReport,
)
from factory_bimanual.collision_safe_follow import (
    CollisionSafeFollowConfig, PoseToleranceTier,
    _bounded_tier_diverse_pairs,
    _bounded_recovery_diverse_frontier,
    _orientation_continuity_cost,
    solve_collision_safe_follow,
)
from factory_bimanual.strict_bimanual_ik import IKCandidate


def _candidate(q, branch):
    return IKCandidate(
        q=np.asarray([q], float), branch_index=branch,
        actual_tcp=np.asarray([q, 0., 0., 1., 0., 0., 0.]),
        position_error_m=0., orientation_error_rad=0.,
        joint_limit_margin_rad=1., singularity_margin=1.,
    )


def _task(count=2):
    zeros = np.zeros((count, 3)); quaternion = np.tile([1., 0., 0., 0.], (count, 1))
    return SimpleNamespace(
        time_s=np.arange(count, dtype=float), source_row_indices=np.arange(count),
        left_position_m=zeros.copy(), right_position_m=zeros.copy(),
        left_quaternion_wxyz=quaternion.copy(),
        right_quaternion_wxyz=quaternion.copy(),
    )


def test_relaxed_pair_is_selected_when_every_exact_pair_collides():
    tiers = (
        PoseToleranceTier("exact", .001, np.deg2rad(1.5), 1., 0., 0),
        PoseToleranceTier("clearance", .005, np.deg2rad(10), .25, .006, 1),
    )

    def candidates(_model, _contract, _task, row, side, tier):
        value = 0. if tier.name == "exact" else (-1. if side == "left" else 1.)
        return [_candidate(value, row)]

    checker = CallbackCollisionChecker(
        pair_evaluator=lambda left, right: CollisionReport(
            (CollisionClass.CROSS_ARM,) if left[0] == right[0] == 0. else ()))
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=checker, tiers=tiers,
            candidate_provider=candidates, initial_left_q=np.array([-1.]),
            initial_right_q=np.array([1.]), max_velocity_rad_s=10.,
            generate_all_tiers=False,
        ))

    assert result.collision_count == 0
    assert result.followed.tolist() == [True, True]
    assert result.left_tier.tolist() == ["clearance", "clearance"]
    assert result.right_tier.tolist() == ["clearance", "clearance"]
    np.testing.assert_allclose(result.left_q, -1.)
    np.testing.assert_allclose(result.right_q, 1.)


def test_collision_only_row_uses_previous_safe_hold():
    tier = PoseToleranceTier("exact", .001, np.deg2rad(1.5), 1., 0., 0)

    def candidates(_model, _contract, _task, row, side, _tier):
        if row == 0:
            return [_candidate(-1. if side == "left" else 1., row)]
        return [_candidate(0., row)]

    checker = CallbackCollisionChecker(
        pair_evaluator=lambda left, right: CollisionReport(
            (CollisionClass.CROSS_ARM,) if left[0] == right[0] == 0. else ()))
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=checker, tiers=(tier,),
            candidate_provider=candidates, initial_left_q=np.array([-1.]),
            initial_right_q=np.array([1.]), max_velocity_rad_s=10.,
        ))

    assert result.collision_count == 0
    assert result.followed.tolist() == [True, False]
    assert result.target_state_safe.tolist() == [True, False]
    assert result.recovery_mode[0] == "target"
    assert result.recovery_mode[1] in {
        "bounded_collision_approach", "left_only_collision_avoidance",
        "right_only_collision_avoidance"}
    assert result.failure_reason.tolist() == [
        "ok", "state_collision_or_clearance"]
    assert not (np.array_equal(result.left_q[1], result.left_q[0]) and
                np.array_equal(result.right_q[1], result.right_q[0]))


def test_relaxed_pair_bridges_an_exact_transition_collision():
    tiers = (
        PoseToleranceTier("exact", .001, .01, 1., 0., 0),
        PoseToleranceTier("bridge", .005, .2, .25, .005, 1),
    )

    def candidates(_model, _contract, _task, row, side, tier):
        if row == 0 or tier.name == "bridge":
            value = -1. if side == "left" else 1.
        else:
            value = 1. if side == "left" else -1.
        return [_candidate(value, row)]

    checker = CallbackCollisionChecker(
        transition_evaluator=lambda previous, current: CollisionReport(
            (CollisionClass.CROSS_ARM,) if previous != current else ()))
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=checker, tiers=tiers,
            candidate_provider=candidates, initial_left_q=np.array([-1.]),
            initial_right_q=np.array([1.]), max_velocity_rad_s=10.,
        ))

    assert result.followed.tolist() == [True, True]
    assert result.left_tier.tolist() == ["exact", "bridge"]
    assert result.collision_count == 0


def test_missing_ik_is_not_mislabeled_as_collision_free_success():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    def candidates(_model, _contract, _task, row, side, _tier):
        if row == 1 and side == "right":
            return []
        return [_candidate(-1. if side == "left" else 1., row)]

    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=CallbackCollisionChecker(), tiers=(tier,),
            candidate_provider=candidates,
            initial_left_q=np.array([-1.]), initial_right_q=np.array([1.]),
            max_velocity_rad_s=10.,
        ))

    assert result.followed.tolist() == [True, False]
    assert result.target_state_safe.tolist() == [True, False]
    assert result.failure_reason.tolist() == ["ok", "right_pose_unreachable"]


def test_one_arm_yields_then_pair_catches_up_without_swept_collision():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    def candidates(_model, _contract, _task, row, side, _tier):
        if row == 0:
            return [_candidate(-1. if side == "left" else 1., row)]
        return [_candidate(1. if side == "left" else -1., row)]

    def transition(previous, current):
        changed = int(previous[0][0] != current[0][0]) + int(
            previous[1][0] != current[1][0])
        return CollisionReport(
            (CollisionClass.CROSS_ARM,) if changed > 1 else ())

    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(3),
        CollisionSafeFollowConfig(
            collision_checker=CallbackCollisionChecker(
                transition_evaluator=transition),
            tiers=(tier,), candidate_provider=candidates,
            initial_left_q=np.array([-1.]), initial_right_q=np.array([1.]),
            max_velocity_rad_s=10., max_jump_rad=3.,
        ))

    assert result.followed.tolist() == [True, False, True]
    assert result.recovery_mode[1] in {
        "left_only_collision_avoidance", "right_only_collision_avoidance"}
    assert result.collision_count == 0


def test_selected_path_reports_state_and_swept_clearance_metrics():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    def candidates(_model, _contract, _task, row, side, _tier):
        return [_candidate(-1. if side == "left" else 1., row)]

    checker = CallbackCollisionChecker()
    checker.clearance = lambda left, right: SimpleNamespace(
        minimum_m=.025, limiting_pair="gripper_base__gripper_base")
    checker.transition_clearance = lambda previous, current: SimpleNamespace(
        minimum_m=.018, limiting_pair="fingers__opposite_wrist")
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(3),
        CollisionSafeFollowConfig(
            collision_checker=checker, tiers=(tier,),
            candidate_provider=candidates,
            initial_left_q=np.array([-1.]), initial_right_q=np.array([1.]),
            max_velocity_rad_s=10.,
        ))

    np.testing.assert_allclose(result.state_clearance_m, .025)
    assert np.isnan(result.swept_clearance_m[0])
    np.testing.assert_allclose(result.swept_clearance_m[1:], .018)
    assert result.minimum_clearance_m == .018
    assert result.limiting_clearance_pair == "fingers__opposite_wrist"


def test_pair_frontier_prefers_more_clearance_within_same_strict_tier():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)
    low = (SimpleNamespace(tier=tier, candidate=_candidate(0., 0)),
           SimpleNamespace(tier=tier, candidate=_candidate(0., 0)))
    high = (SimpleNamespace(tier=tier, candidate=_candidate(1., 1)),
            SimpleNamespace(tier=tier, candidate=_candidate(1., 1)))

    selected = _bounded_tier_diverse_pairs(
        [low, high], (tier,), 1,
        clearance_evaluator=lambda pair: pair[0].candidate.q[0])

    assert selected == [high]


def test_orientation_continuity_cost_penalizes_offset_velocity_and_swing():
    previous = IKCandidate(
        q=np.zeros(1), branch_index=0, tool_axis_offset_deg=15.0,
        swing_offset_deg=0.0)
    unchanged = IKCandidate(
        q=np.zeros(1), branch_index=1, tool_axis_offset_deg=15.0,
        swing_offset_deg=0.0)
    changed = IKCandidate(
        q=np.zeros(1), branch_index=2, tool_axis_offset_deg=-15.0,
        swing_offset_deg=7.5)

    assert _orientation_continuity_cost(previous, unchanged) == 0.0
    assert _orientation_continuity_cost(previous, changed) == pytest.approx(
        30.0 ** 2 + 7.5 ** 2)


def test_bounded_recovery_reenters_a_distant_target_branch():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    def candidates(_model, _contract, _task, row, side, _tier):
        value = 0. if row == 0 else (2. if side == "left" else -2.)
        return [_candidate(value, row)]

    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(6),
        CollisionSafeFollowConfig(
            collision_checker=CallbackCollisionChecker(), tiers=(tier,),
            candidate_provider=candidates, initial_left_q=np.array([0.]),
            initial_right_q=np.array([0.]), max_velocity_rad_s=np.inf,
            max_jump_rad=.5,
        ))

    assert result.recovery_mode[1] == "bounded_collision_safe_recovery"
    assert result.followed[-1]
    assert result.left_q[-1, 0] == 2.
    assert result.right_q[-1, 0] == -2.
    assert np.max(np.abs(np.diff(result.left_q[:, 0]))) <= .5 + 1e-12
    assert result.collision_count == 0


def test_recovery_keeps_moving_when_a_receding_target_cannot_be_caught():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    def candidates(_model, _contract, _task, row, side, _tier):
        value = 0. if row == 0 else row * (2. if side == "left" else -2.)
        return [_candidate(value, row)]

    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(6),
        CollisionSafeFollowConfig(
            collision_checker=CallbackCollisionChecker(), tiers=(tier,),
            candidate_provider=candidates,
            initial_left_q=np.array([0.]), initial_right_q=np.array([0.]),
            max_velocity_rad_s=np.inf, max_jump_rad=.5,
        ))

    assert np.all(np.diff(result.left_q[:, 0]) >= 0.)
    assert result.left_q[-1, 0] > 0.
    assert result.recovery_mode[-1] != "collision_safe_hold"


def test_global_path_may_skip_trapping_first_target_to_preserve_future():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    def candidates(_model, _contract, _task, row, side, _tier):
        if row == 0:
            return [_candidate(float(value), value) for value in range(80)]
        return [_candidate(-1. if side == "left" else 1., row)]

    def transition(previous, current):
        trapped = previous[0][0] == previous[1][0]
        return CollisionReport(
            (CollisionClass.CROSS_ARM,) if trapped and previous != current else ())

    checker = CallbackCollisionChecker(
        pair_evaluator=lambda left, right: CollisionReport(
            (CollisionClass.CROSS_ARM,)
            if left[0] != right[0] and not (left[0] == -1. and right[0] == 1.)
            else ()),
        transition_evaluator=transition)
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(3),
        CollisionSafeFollowConfig(
            collision_checker=checker,
            tiers=(tier,), candidate_provider=candidates,
            initial_left_q=np.array([-1.]), initial_right_q=np.array([1.]),
            max_velocity_rad_s=np.inf, max_jump_rad=3.,
        ))

    assert result.followed.tolist() == [False, True, True]
    assert result.recovery_mode[0] == "collision_safe_hold"
    assert result.collision_count == 0


def test_hold_crowding_cannot_prune_every_moving_recovery_branch():
    def node(index, mode):
        candidate = SimpleNamespace(branch_index=index)
        tagged = SimpleNamespace(candidate=candidate)
        return SimpleNamespace(
            cost=(1, 0, float(index), 0.), left=tagged, right=tagged,
            followed=False, recovery_mode=mode)

    holds = [node(index, "collision_safe_hold") for index in range(80)]
    moving = [node(100 + index, "bounded_collision_safe_recovery")
              for index in range(12)]
    selected = _bounded_recovery_diverse_frontier(holds + moving, 16)

    assert any(item.recovery_mode == "bounded_collision_safe_recovery"
               for item in selected)
    assert any(item.recovery_mode == "collision_safe_hold"
               for item in selected)


def test_branch_discontinuity_requests_a_nearby_diverse_ik_branch():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    class Provider:
        def __init__(self):
            self.diversify_calls = []

        def __call__(self, _model, _contract, _task, row, _side, _tier):
            return [_candidate(0. if row == 0 else 1., row)]

        def diversify(self, _model, _contract, _task, row, side, _tier):
            self.diversify_calls.append((row, side))
            return [_candidate(.4, 100 + row)]

    provider = Provider()
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=CallbackCollisionChecker(), tiers=(tier,),
            candidate_provider=provider,
            initial_left_q=np.array([0.]), initial_right_q=np.array([0.]),
            max_velocity_rad_s=np.inf, max_jump_rad=.5,
        ))

    assert provider.diversify_calls == [(1, "left"), (1, "right")]
    assert result.followed.tolist() == [True, True]
    np.testing.assert_allclose(result.left_q[:, 0], [0., .4])
    np.testing.assert_allclose(result.right_q[:, 0], [0., .4])


def test_branch_discontinuity_requests_reference_aware_tolerance_box_ik():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    class Provider:
        def __init__(self):
            self.reference_calls = []

        def __call__(self, _model, _contract, _task, row, _side, _tier):
            return [_candidate(0. if row == 0 else 1., row)]

        def reference(self, _model, _contract, _task, row, side, _tier,
                      reference_q):
            self.reference_calls.append((row, side, reference_q.copy()))
            return [_candidate(.4, 100 + row)]

    provider = Provider()
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=CallbackCollisionChecker(), tiers=(tier,),
            candidate_provider=provider, reference_aware_enabled=True,
            initial_left_q=np.array([0.]), initial_right_q=np.array([0.]),
            max_velocity_rad_s=np.inf, max_jump_rad=.5,
        ))

    assert [(row, side) for row, side, _ in provider.reference_calls] == [
        (1, "left"), (1, "right")]
    assert result.followed.tolist() == [True, True]
    np.testing.assert_allclose(result.left_q[:, 0], [0., .4])
    np.testing.assert_allclose(result.right_q[:, 0], [0., .4])


def test_orientation_adaptation_runs_only_after_exact_connection_fails():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    class Provider:
        def __init__(self):
            self.adaptation_calls = []

        def __call__(self, _model, _contract, _task, row, _side, _tier):
            return [_candidate(0. if row == 0 else 1., row)]

        def adapted(self, _model, _contract, _task, row, side, _tier,
                    references):
            self.adaptation_calls.append((row, side, len(references)))
            item = _candidate(.4, 200 + row)
            return [replace(
                item, adaptation_level="tool_axis",
                adapted_target_quaternion_wxyz=np.asarray([.99, 0., 0., .1]),
                tool_axis_offset_deg=15.0, swing_offset_deg=0.0)]

    provider = Provider()
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=CallbackCollisionChecker(), tiers=(tier,),
            candidate_provider=provider, orientation_adaptation_enabled=True,
            initial_left_q=np.array([0.]), initial_right_q=np.array([0.]),
            max_velocity_rad_s=np.inf, max_jump_rad=.5,
        ))

    assert provider.adaptation_calls == [(1, "left", 1), (1, "right", 1)]
    assert result.followed.tolist() == [True, True]
    assert result.candidate_count["left"].tolist() == [1, 2]
    assert result.candidate_count["right"].tolist() == [1, 2]
    assert result.safe_pair_count.tolist() == [1, 4]
    assert result.orientation_adaptation_level["left"].tolist() == [
        "original", "tool_axis"]
    assert result.tool_axis_offset_deg["left"].tolist() == [0.0, 15.0]


def test_colliding_only_pair_requests_diverse_safe_ik_branches():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    class Provider:
        def __init__(self):
            self.diversify_calls = []

        def __call__(self, _model, _contract, _task, row, side, _tier):
            if row == 0:
                return [_candidate(-1. if side == "left" else 1., row)]
            return [_candidate(0., row)]

        def diversify(self, _model, _contract, _task, row, side, _tier):
            self.diversify_calls.append((row, side))
            return [_candidate(-1. if side == "left" else 1., 100 + row)]

    provider = Provider()
    checker = CallbackCollisionChecker(
        pair_evaluator=lambda left, right: CollisionReport(
            (CollisionClass.CROSS_ARM,)
            if left[0] == right[0] == 0. else ()))
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=checker, tiers=(tier,),
            candidate_provider=provider,
            initial_left_q=np.array([-1.]), initial_right_q=np.array([1.]),
            max_velocity_rad_s=np.inf, max_jump_rad=3.,
        ))

    assert provider.diversify_calls == [(1, "left"), (1, "right")]
    assert result.followed.tolist() == [True, True]
    assert result.collision_count == 0


def test_swept_collision_requests_a_transition_safe_ik_branch():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    class Provider:
        def __init__(self):
            self.diversify_calls = []

        def __call__(self, _model, _contract, _task, row, side, _tier):
            if row == 0:
                return [_candidate(-1. if side == "left" else 1., row)]
            return [_candidate(1. if side == "left" else -1., row)]

        def diversify(self, _model, _contract, _task, row, side, _tier):
            self.diversify_calls.append((row, side))
            return [_candidate(-1. if side == "left" else 1., 100 + row)]

    provider = Provider()
    checker = CallbackCollisionChecker(
        transition_evaluator=lambda previous, current: CollisionReport(
            (CollisionClass.CROSS_ARM,) if previous != current else ()))
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=checker, tiers=(tier,),
            candidate_provider=provider,
            initial_left_q=np.array([-1.]), initial_right_q=np.array([1.]),
            max_velocity_rad_s=np.inf, max_jump_rad=3.,
        ))

    assert provider.diversify_calls == [(1, "left"), (1, "right")]
    assert result.followed.tolist() == [True, True]
    assert result.collision_count == 0


def test_colliding_target_pair_allows_a_safe_bounded_two_arm_approach():
    tier = PoseToleranceTier("exact", .001, .01, 1., 0., 0)

    def candidates(_model, _contract, _task, row, side, _tier):
        if row == 0:
            return [_candidate(0., row)]
        return [_candidate(2. if side == "left" else -2., row)]

    checker = CallbackCollisionChecker(
        pair_evaluator=lambda left, right: CollisionReport(
            (CollisionClass.CROSS_ARM,)
            if left[0] >= 1. and right[0] <= -1. else ()))
    result = solve_collision_safe_follow(
        None, SimpleNamespace(dof_per_arm=1), _task(),
        CollisionSafeFollowConfig(
            collision_checker=checker, tiers=(tier,),
            candidate_provider=candidates,
            initial_left_q=np.array([0.]), initial_right_q=np.array([0.]),
            max_velocity_rad_s=np.inf, max_jump_rad=.5,
        ))

    assert result.recovery_mode[1] == "bounded_collision_approach"
    np.testing.assert_allclose(result.left_q[1], [.5])
    np.testing.assert_allclose(result.right_q[1], [-.5])
    assert result.collision_count == 0
