import unittest

import numpy as np

from design_optimization.episode_follow_metrics import (
    classify_solver_failures,
    failure_reason_diagnostics,
)


def test_planner_failure_diagnostics_does_not_call_pose_error_solver_failure():
    from design_optimization.episode_follow_metrics import planner_failure_diagnostics

    result = planner_failure_diagnostics(
        rolling_success=np.array([True, False, False, False]),
        collision=np.array([False, True, False, False]),
        jump_violation=np.array([False, False, True, False]),
        recovery_mode=np.array(["none", "hold", "limited_step", "hold"]),
    )
    assert result["per_frame"].tolist() == ["none", "collision", "jump_violation", "branch_lost"]
    assert result["counts"] == {"collision": 1, "jump_violation": 1, "branch_lost": 1}


def test_failure_reason_diagnostics_does_not_invent_solver_failure_from_two_errors():
    result = failure_reason_diagnostics(
        position_error_m=np.asarray([0.01]),
        orientation_error_rad=np.asarray([0.2]),
        table_collision=np.asarray([False]),
        self_collision=np.asarray([False]),
    )
    assert "solver_failure" not in result["affected_frame_counts"]
    assert result["primary_per_frame"].tolist() == ["position_and_orientation"]


class FailureReasonDiagnosticsTest(unittest.TestCase):
    def test_independent_pose_success_identifies_branch_loss(self) -> None:
        reasons = classify_solver_failures(
            rolling_success=np.asarray((False, False, False)),
            position_feasible=np.asarray((False, True, True)),
            independent_pose_feasible=np.asarray((False, False, True)),
            collision=np.asarray((False, False, False)),
            jump_violation=np.asarray((False, False, False)),
        )
        self.assertEqual(
            reasons.tolist(),
            ["position_unreachable", "pose_infeasible", "branch_lost"],
        )

    def test_pose_error_is_primary_when_discontinuity_candidate_was_held(self) -> None:
        result = failure_reason_diagnostics(
            position_error_m=np.asarray((0.10,)),
            orientation_error_rad=np.deg2rad(np.asarray((20.0,))),
            table_collision=np.asarray((False,)),
            self_collision=np.asarray((False,)),
            joint_discontinuity=np.asarray((True,)),
        )

        self.assertEqual(result["primary_per_frame"].tolist(), ["position_and_orientation"])
        self.assertEqual(result["affected_frame_counts"]["joint_discontinuity"], 1)


if __name__ == "__main__":
    unittest.main()
