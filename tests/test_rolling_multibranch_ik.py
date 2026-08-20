import unittest

import numpy as np

from scripts.rolling_multibranch_ik import (
    BranchCandidate,
    select_global_feasible_path,
    select_minimum_retime_path,
    select_rolling_branch,
    transition_limit_rad,
)


def candidate(value: float, index: int) -> BranchCandidate:
    return BranchCandidate(
        q=np.asarray((value,), dtype=float),
        pose_valid=True,
        collision_free=True,
        position_error_m=0.0,
        orientation_error_rad=0.0,
        joint_limit_margin=1.0,
        singularity_margin=1.0,
        index=index,
    )


class RollingMultiBranchIKTest(unittest.TestCase):
    def test_receding_horizon_separates_pose_and_collision_blocked_layers(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path

        pose_invalid = BranchCandidate(
            q=np.asarray((0.0,)), pose_valid=False, collision_free=True,
            position_error_m=.1, orientation_error_rad=.2,
            joint_limit_margin=1.0, singularity_margin=1.0, index=0)
        collision_blocked = BranchCandidate(
            q=np.asarray((0.1,)), pose_valid=True, collision_free=False,
            position_error_m=0.0, orientation_error_rad=0.0,
            joint_limit_margin=1.0, singularity_margin=1.0, index=0)

        result = select_receding_horizon_path(
            layers=((pose_invalid,), (collision_blocked,)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.asarray((.1, .1)), velocity_limit_rad_s=np.asarray((1.0,)),
            cap_rad=np.asarray((.2,)), horizon=2, beam_width=2)

        self.assertEqual(result.recovery_mode.tolist(), [
            "hold_no_pose_candidate", "hold_state_collision_blocked"])

    def test_minimum_retime_chooses_branch_that_avoids_late_wrist_flip(self) -> None:
        result = select_minimum_retime_path(
            layers=((candidate(0.0, 0), candidate(1.0, 1)),
                    (candidate(0.1, 0), candidate(1.1, 1)),
                    (candidate(1.2, 1),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.asarray((.1, .1, .1)),
            velocity_limit_rad_s=np.asarray((1.0,)),
        )
        self.assertEqual(result.selected_indices.tolist(), [1, 1, 1])
        self.assertTrue(result.pose_valid.all())
        self.assertAlmostEqual(result.required_dt_s[2], .125)

    def test_minimum_retime_expands_only_infeasible_edge(self) -> None:
        result = select_minimum_retime_path(
            layers=((candidate(0.0, 0),), (candidate(.3, 1),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.asarray((.1, .1)),
            velocity_limit_rad_s=np.asarray((1.0,)), safety_fraction=.8,
        )
        np.testing.assert_allclose(result.required_dt_s, (.1, .375))

    def test_minimum_retime_prefers_lower_velocity_variation_when_time_is_equal(self) -> None:
        result = select_minimum_retime_path(
            layers=((candidate(.2, 0),),
                    (candidate(0.0, 0), candidate(.4, 1)),
                    (candidate(.2, 0),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.ones(3), velocity_limit_rad_s=np.ones(1),
        )
        self.assertEqual(result.selected_indices.tolist(), [0, 1, 0])

    def test_minimum_retime_cannot_legalize_visual_branch_flip(self) -> None:
        def arm(q, index):
            return BranchCandidate(
                q=np.asarray(q, dtype=float), pose_valid=True,
                collision_free=True, position_error_m=0.0,
                orientation_error_rad=0.0, joint_limit_margin=1.0,
                singularity_margin=1.0, index=index)

        safe = arm((0, 0, 0, .1, .1, .1), 0)
        flipped = arm((0, 0, 0, 3.0, -2.0, 2.0), 1)
        result = select_minimum_retime_path(
            layers=((safe,), (flipped,)), initial_q=np.zeros(6),
            periodic=np.zeros(6, dtype=bool), dt_s=np.asarray((.1, .1)),
            velocity_limit_rad_s=np.ones(6),
            maximum_joint_step_rad=np.deg2rad(35),
            maximum_wrist_step_norm_rad=np.deg2rad(45),
        )
        self.assertEqual(result.selected_indices.tolist(), [0, -1])
        self.assertFalse(result.pose_valid[1])

    def test_minimum_retime_holds_instead_of_entering_dangerous_branch(self) -> None:
        dangerous = BranchCandidate(
            q=np.asarray((1.0,)), pose_valid=True, collision_free=True,
            position_error_m=0.0, orientation_error_rad=0.0,
            joint_limit_margin=np.deg2rad(2), singularity_margin=.01, index=3)
        result = select_minimum_retime_path(
            layers=((candidate(0.0, 0),), (dangerous,), (candidate(.1, 1),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.asarray((.1, .1, .1)), velocity_limit_rad_s=np.asarray((1.0,)),
            minimum_joint_limit_margin_rad=np.deg2rad(10),
            minimum_singularity_margin=.05,
        )
        self.assertEqual(result.selected_indices.tolist(), [0, -1, 1])
        self.assertEqual(result.recovery_mode[1], "hold_low_quality_candidate")

    def test_global_selector_keeps_the_only_future_connected_branch(self) -> None:
        result = select_global_feasible_path(
            layers=((candidate(0.0, 0), candidate(1.0, 1)),
                    (candidate(0.1, 0), candidate(1.1, 1)),
                    (candidate(1.2, 1),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.ones(3), velocity_limit_rad_s=np.asarray((0.25,)),
            cap_rad=np.asarray((1.0,)),
        )
        self.assertEqual(result.selected_indices.tolist(), [1, 1, 1])
        self.assertTrue(result.pose_valid.all())

    def test_global_selector_never_selects_collision_candidate(self) -> None:
        colliding = BranchCandidate(
            q=np.asarray((0.1,)), pose_valid=True, collision_free=False,
            position_error_m=0.0, orientation_error_rad=0.0,
            joint_limit_margin=1.0, singularity_margin=1.0, index=0)
        result = select_global_feasible_path(
            layers=((candidate(0.0, 0),), (colliding, candidate(0.2, 1))),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.ones(2), velocity_limit_rad_s=np.asarray((1.0,)),
            cap_rad=np.asarray((1.0,)),
        )
        self.assertEqual(result.selected_indices.tolist(), [0, 1])

    def test_transition_limit_uses_timestamp_and_safety_cap(self) -> None:
        self.assertAlmostEqual(
            transition_limit_rad(0.1, np.deg2rad(90.0), np.deg2rad(25.0)),
            np.deg2rad(9.0),
        )
        self.assertAlmostEqual(
            transition_limit_rad(1.0, np.deg2rad(90.0), np.deg2rad(25.0)),
            np.deg2rad(25.0),
        )

    def test_future_safe_branch_beats_greedy_nearest_candidate(self) -> None:
        result = select_rolling_branch(
            layers=((candidate(0.1, 0), candidate(0.4, 1)), (candidate(0.8, 0),)),
            initial_q=np.asarray((0.0,)),
            periodic=np.asarray((False,)),
            dt_s=np.asarray((1.0, 1.0)),
            velocity_limit_rad_s=np.asarray((0.5,)),
            cap_rad=np.asarray((0.5,)),
            beam_width=8,
        )
        self.assertTrue(result.complete)
        self.assertEqual([item.index for item in result.path], [1, 0])

    def test_periodic_transition_uses_shortest_rotation(self) -> None:
        result = select_rolling_branch(
            layers=((candidate(-np.pi + 0.05, 0),),),
            initial_q=np.asarray((np.pi - 0.05,)),
            periodic=np.asarray((True,)),
            dt_s=np.asarray((1.0,)),
            velocity_limit_rad_s=np.asarray((0.2,)),
            cap_rad=np.asarray((0.2,)),
            beam_width=2,
        )
        self.assertTrue(result.complete)

    def test_equal_pose_paths_prefer_lower_acceleration(self) -> None:
        result = select_rolling_branch(
            layers=(
                (candidate(0.1, 0), candidate(0.2, 1)),
                (candidate(0.3, 0),),
                (candidate(0.5, 0),),
            ),
            initial_q=np.asarray((0.0,)),
            periodic=np.asarray((False,)),
            dt_s=np.ones(3),
            velocity_limit_rad_s=np.asarray((1.0,)),
            cap_rad=np.asarray((1.0,)),
            beam_width=8,
        )
        self.assertEqual(result.path[0].index, 0)

    def test_initial_velocity_prefers_smooth_continuation(self) -> None:
        result = select_rolling_branch(
            layers=((candidate(0.2, 0), candidate(-0.1, 1)),),
            initial_q=np.asarray((0.0,)),
            initial_velocity_rad_s=np.asarray((0.2,)),
            periodic=np.asarray((False,)),
            dt_s=np.ones(1), velocity_limit_rad_s=np.asarray((1.0,)),
            cap_rad=np.asarray((1.0,)), beam_width=2,
        )
        self.assertEqual(result.path[0].index, 0)

    def test_initial_velocity_shape_must_match_joints(self) -> None:
        with self.assertRaises(ValueError):
            select_rolling_branch(
                layers=((candidate(0.1, 0),),), initial_q=np.asarray((0.0,)),
                initial_velocity_rad_s=np.asarray((0.0, 0.0)),
                periodic=np.asarray((False,)), dt_s=np.ones(1),
                velocity_limit_rad_s=np.ones(1), cap_rad=np.ones(1),
            )

    def test_hard_acceleration_limit_rejects_velocity_reversal(self) -> None:
        result = select_rolling_branch(
            layers=((candidate(-1.0, 0), candidate(0.5, 1)),),
            initial_q=np.asarray((0.0,)),
            initial_velocity_rad_s=np.asarray((1.0,)),
            initial_acceleration_rad_s2=np.asarray((0.0,)),
            periodic=np.asarray((False,)), dt_s=np.ones(1),
            velocity_limit_rad_s=np.asarray((2.0,)), cap_rad=np.asarray((2.0,)),
            acceleration_limit_rad_s2=np.asarray((0.5,)),
            jerk_limit_rad_s3=np.asarray((10.0,)), beam_width=2,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.path[0].index, 1)

    def test_hard_jerk_limit_rejects_acceleration_sign_flip(self) -> None:
        result = select_rolling_branch(
            layers=((candidate(0.0, 0), candidate(1.5, 1)),),
            initial_q=np.asarray((0.0,)),
            initial_velocity_rad_s=np.asarray((1.0,)),
            initial_acceleration_rad_s2=np.asarray((1.0,)),
            periodic=np.asarray((False,)), dt_s=np.ones(1),
            velocity_limit_rad_s=np.asarray((2.0,)), cap_rad=np.asarray((2.0,)),
            acceleration_limit_rad_s2=np.asarray((2.0,)),
            jerk_limit_rad_s3=np.asarray((0.5,)), beam_width=2,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.path[0].index, 1)

    def test_transition_checker_rejects_colliding_edge(self) -> None:
        result = select_rolling_branch(
            layers=((candidate(0.4, 0), candidate(-0.4, 1)),),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.ones(1), velocity_limit_rad_s=np.ones(1), cap_rad=np.ones(1),
            beam_width=2,
            transition_valid=lambda _previous, current: bool(current[0] < 0.0),
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.path[0].index, 1)

    def test_blocked_future_returns_best_safe_partial_path(self) -> None:
        result = select_rolling_branch(
            layers=((candidate(0.1, 0),), (candidate(0.9, 0),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.ones(2), velocity_limit_rad_s=np.asarray((0.2,)),
            cap_rad=np.asarray((0.2,)), beam_width=2,
        )
        self.assertFalse(result.complete)
        self.assertEqual([item.index for item in result.path], [0])

    def test_receding_horizon_restarts_after_an_empty_candidate_layer(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        result = select_receding_horizon_path(
            layers=((candidate(0.0, 0),), (), (candidate(0.1, 1),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.ones(3), velocity_limit_rad_s=np.asarray((0.2,)),
            cap_rad=np.asarray((0.2,)), horizon=2, beam_width=2,
        )
        self.assertEqual(result.selected_indices.tolist(), [0, -1, 1])
        np.testing.assert_allclose(result.q, ((0.0,), (0.0,), (0.1,)))
        self.assertEqual(result.recovery_mode.tolist(), ["initialize", "hold_no_candidate", "none"])

    def test_receding_horizon_uses_future_safe_branch(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        result = select_receding_horizon_path(
            layers=((candidate(0.1, 0), candidate(0.4, 1)), (candidate(0.8, 2),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.ones(2), velocity_limit_rad_s=np.asarray((0.5,)),
            cap_rad=np.asarray((0.5,)), horizon=2, beam_width=4,
        )
        self.assertEqual(result.selected_indices.tolist(), [1, 2])
        self.assertTrue(result.pose_valid.all())

    def test_receding_horizon_infers_source_initial_velocity(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        result = select_receding_horizon_path(
            layers=((candidate(0.0, 0),), (candidate(0.1, 1),),
                    (candidate(0.2, 2),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.asarray((.1, .1, .1)),
            velocity_limit_rad_s=np.asarray((2.0,)), cap_rad=np.asarray((2.0,)),
            acceleration_limit_rad_s2=np.asarray((.1,)),
            jerk_limit_rad_s3=np.asarray((1.0,)), horizon=3, beam_width=4,
        )
        self.assertTrue(result.pose_valid.all())
        self.assertEqual(result.recovery_mode.tolist(), ["initialize", "none", "none"])

    def test_receding_horizon_initial_pose_is_not_treated_as_a_timed_transition(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        result = select_receding_horizon_path(
            layers=((candidate(1.0, 3),),), initial_q=np.asarray((0.0,)),
            periodic=np.asarray((False,)), dt_s=np.asarray((.01,)),
            velocity_limit_rad_s=np.asarray((.1,)), cap_rad=np.asarray((.1,)),
            horizon=1, beam_width=2,
        )
        np.testing.assert_allclose(result.q[0], (1.0,))
        self.assertEqual(result.recovery_mode[0], "initialize")
        self.assertTrue(result.pose_valid[0])

    def test_receding_horizon_takes_a_bounded_step_toward_unreachable_edge(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        result = select_receding_horizon_path(
            layers=((candidate(0.0, 0),), (candidate(0.2, 1),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.asarray((.1, .1)), velocity_limit_rad_s=np.asarray((.5,)),
            cap_rad=np.asarray((.5,)), horizon=2, beam_width=2,
        )
        np.testing.assert_allclose(result.q[:, 0], (0.0, 0.05), atol=1e-12)
        self.assertEqual(result.recovery_mode.tolist(), ["initialize", "limited_step"])
        self.assertEqual(result.selected_indices.tolist(), [0, 1])
        self.assertFalse(result.pose_valid[1])

    def test_receding_horizon_recovery_obeys_acceleration_and_jerk(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        result = select_receding_horizon_path(
            layers=((candidate(0.0, 0),), (candidate(0.1, 1),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.asarray((.1, .1)), velocity_limit_rad_s=np.asarray((2.0,)),
            cap_rad=np.asarray((2.0,)), acceleration_limit_rad_s2=np.asarray((2.0,)),
            jerk_limit_rad_s3=np.asarray((20.0,)), horizon=1, beam_width=2,
        )
        np.testing.assert_allclose(result.q[:, 0], (0.0, 0.02), atol=1e-12)
        self.assertEqual(result.recovery_mode[-1], "limited_step")

    def test_receding_horizon_holds_when_bounded_recovery_edge_collides(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        result = select_receding_horizon_path(
            layers=((candidate(0.0, 0),), (candidate(0.2, 1),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.asarray((.1, .1)), velocity_limit_rad_s=np.asarray((.5,)),
            cap_rad=np.asarray((.5,)), horizon=2, beam_width=2,
            transition_valid=lambda _old, new: bool(new[0] <= .01),
        )
        np.testing.assert_allclose(result.q[:, 0], (0.0, 0.0), atol=1e-12)
        self.assertEqual(result.recovery_mode[-1], "hold_recovery_collision")

    def test_receding_horizon_can_reseed_after_a_long_unreachable_gap(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        result = select_receding_horizon_path(
            layers=((candidate(0.0, 0),), (), (), (candidate(1.0, 4),)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.ones(4), velocity_limit_rad_s=np.asarray((0.1,)),
            cap_rad=np.asarray((0.1,)), horizon=2, beam_width=2,
            reseed_after_empty_frames=2,
        )
        np.testing.assert_allclose(result.q[:, 0], (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(result.recovery_mode[-1], "reinitialize_after_gap")
        self.assertTrue(result.pose_valid[-1])

    def test_receding_horizon_rejects_low_quality_wrist_flip_target(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        weak = BranchCandidate(
            q=np.asarray((1.0,)), pose_valid=True, collision_free=True,
            position_error_m=0.0, orientation_error_rad=0.0,
            joint_limit_margin=np.deg2rad(2), singularity_margin=.01, index=9)
        result = select_receding_horizon_path(
            layers=((candidate(0.0, 0),), (weak,)), initial_q=np.asarray((0.0,)),
            periodic=np.asarray((False,)), dt_s=np.asarray((.1, .1)),
            velocity_limit_rad_s=np.asarray((1.0,)), cap_rad=np.asarray((1.0,)),
            minimum_joint_limit_margin_rad=np.deg2rad(10),
            minimum_singularity_margin=.05,
        )
        np.testing.assert_allclose(result.q[:, 0], (0.0, 0.0))
        self.assertEqual(result.recovery_mode[-1], "hold_low_quality_candidate")

    def test_initial_branch_prefers_joint_limit_clearance_before_pose_tie(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        near_limit = BranchCandidate(
            np.asarray((0.0,)), True, True, 0.0, 0.0,
            np.deg2rad(2), .2, 0)
        clear = BranchCandidate(
            np.asarray((1.0,)), True, True, 5e-4, 0.0,
            np.deg2rad(40), .1, 1)
        result = select_receding_horizon_path(
            layers=((near_limit, clear),), initial_q=np.asarray((0.0,)),
            periodic=np.asarray((False,)), dt_s=np.asarray((.1,)),
            velocity_limit_rad_s=np.asarray((1.0,)), cap_rad=np.asarray((1.0,)),
            initial_soft_joint_limit_margin_rad=np.deg2rad(10),
        )
        self.assertEqual(result.selected_indices[0], 1)

    def test_initial_clearance_precedes_lower_future_motion_cost(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        near_limit = BranchCandidate(
            np.asarray((0.0,)), True, True, 0.0, 0.0,
            np.deg2rad(2), .2, 0)
        clear = BranchCandidate(
            np.asarray((1.0,)), True, True, 0.0, 0.0,
            np.deg2rad(40), .1, 1)
        common_future = BranchCandidate(
            np.asarray((0.1,)), True, True, 0.0, 0.0,
            np.deg2rad(40), .1, 2)
        result = select_receding_horizon_path(
            layers=((near_limit, clear), (common_future,)),
            initial_q=np.asarray((0.0,)), periodic=np.asarray((False,)),
            dt_s=np.asarray((1.0, 1.0)),
            velocity_limit_rad_s=np.asarray((1.0,)), cap_rad=np.asarray((1.0,)),
            initial_soft_joint_limit_margin_rad=np.deg2rad(10),
        )
        self.assertEqual(result.selected_indices[0], 1)

    def test_receding_horizon_holds_instead_of_recovering_to_wrist_flip(self) -> None:
        from scripts.rolling_multibranch_ik import select_receding_horizon_path
        flip = BranchCandidate(
            q=np.asarray((0., 0., 0., 1., 1., 1.)), pose_valid=True,
            collision_free=True, position_error_m=0., orientation_error_rad=0.,
            joint_limit_margin=1., singularity_margin=1., index=8)
        result = select_receding_horizon_path(
            layers=((BranchCandidate(np.zeros(6), True, True, 0., 0., 1., 1., 0),),
                    (flip,)), initial_q=np.zeros(6),
            periodic=np.zeros(6, bool), dt_s=np.asarray((.1, .1)),
            velocity_limit_rad_s=np.ones(6), cap_rad=np.ones(6),
            maximum_recovery_wrist_distance_rad=np.deg2rad(45),
        )
        np.testing.assert_allclose(result.q[1], np.zeros(6))
        self.assertEqual(result.recovery_mode[1], "hold_wrist_branch_switch")


if __name__ == "__main__":
    unittest.main()
