import numpy as np

from design_optimization.incremental_mount_evaluator import IncrementalMountEvaluator


class CountingSolver:
    def __init__(self):
        self.calls = []

    def __call__(self, frame_index, seed_q, restart_limit):
        self.calls.append((frame_index, None if seed_q is None else tuple(seed_q), restart_limit))
        return {
            "q": np.array([float(frame_index)]),
            "success": True,
            "position_error_m": frame_index / 1000.0,
            "orientation_error_rad": frame_index / 100.0,
            "table_collision": False,
            "self_collision": False,
            "restarts": 0,
            "iterations": 1,
        }


def test_promotion_only_solves_new_frames():
    solver = CountingSolver()
    evaluator = IncrementalMountEvaluator(frame_count=8)
    first = evaluator.evaluate(
        candidate_id=1, frame_indices=np.array([0, 4, 7]),
        solver=solver, restart_limit=2,
    )
    second = evaluator.evaluate(
        candidate_id=1, frame_indices=np.arange(8),
        solver=solver, restart_limit=3,
    )
    assert [call[0] for call in solver.calls] == [0, 4, 7, 1, 2, 3, 5, 6]
    assert first.new_frames == 3 and first.reused_frames == 0
    assert second.new_frames == 5 and second.reused_frames == 3


def test_child_uses_parent_same_frame_solution_as_first_seed():
    solver = CountingSolver()
    evaluator = IncrementalMountEvaluator(frame_count=4)
    evaluator.evaluate(
        candidate_id=1, frame_indices=np.array([0, 2]),
        solver=solver, restart_limit=2,
    )
    evaluator.evaluate(
        candidate_id=2, parent_id=1, frame_indices=np.array([0, 2]),
        solver=solver, restart_limit=2,
    )
    child_calls = solver.calls[-2:]
    assert child_calls[0][1] == (0.0,)
    assert child_calls[1][1] == (2.0,)


def test_child_recomputes_metrics_instead_of_copying_parent_results():
    solver = CountingSolver()
    evaluator = IncrementalMountEvaluator(frame_count=2)
    parent = evaluator.evaluate(
        candidate_id=1, frame_indices=np.array([1]), solver=solver, restart_limit=2,
    )
    child = evaluator.evaluate(
        candidate_id=2, parent_id=1, frame_indices=np.array([1]),
        solver=solver, restart_limit=2,
    )
    assert len(solver.calls) == 2
    assert parent.frames[1] is not child.frames[1]


def test_current_candidate_nearest_success_precedes_parent_nearest_success():
    solver = CountingSolver()
    evaluator = IncrementalMountEvaluator(frame_count=6)
    evaluator.evaluate(
        candidate_id=1, frame_indices=np.array([0, 5]), solver=solver, restart_limit=2,
    )
    evaluator.evaluate(
        candidate_id=2, parent_id=1, frame_indices=np.array([2]),
        solver=solver, restart_limit=2,
    )
    evaluator.evaluate(
        candidate_id=2, parent_id=1, frame_indices=np.array([3]),
        solver=solver, restart_limit=2,
    )
    assert solver.calls[-1][1] == (2.0,)


def test_invalid_requested_frame_is_rejected():
    evaluator = IncrementalMountEvaluator(frame_count=2)
    solver = CountingSolver()
    try:
        evaluator.evaluate(
            candidate_id=1, frame_indices=np.array([2]), solver=solver, restart_limit=2,
        )
    except ValueError as exc:
        assert "frame" in str(exc)
    else:
        raise AssertionError("out-of-range frame must be rejected")
