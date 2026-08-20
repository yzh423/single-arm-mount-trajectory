from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.executors import ProductionExecutors
from factory_bimanual.run_experiment import ExperimentJob


ROOT = Path(__file__).resolve().parents[2]


def _job(stage="ik", mode="strict_a", rows=2):
    return ExperimentJob(stage, "xarm6", "screw_cap", mode, rows, "fingerprint")


def test_executor_rejects_missing_selected_spacing(tmp_path):
    executor = ProductionExecutors(ROOT, spacing_by_robot={})
    with pytest.raises(RuntimeError, match="selected spacing"):
        executor.ik(_job())


def test_prefix_preserves_source_rows_and_registration():
    executor = ProductionExecutors(ROOT, spacing_by_robot={"xarm6": .8})
    task = executor.load_task("screw_cap", prefix_rows=2)
    assert len(task.time_s) == 2
    assert task.source_row_indices.tolist() == [0, 1]
    assert hasattr(task, "registration")


def test_executor_mapping_exposes_both_stages():
    executor = ProductionExecutors(ROOT, spacing_by_robot={"xarm6": .8})
    assert executor.mapping()["ik"] == executor.ik
    assert executor.mapping()["mpc"] == executor.mpc


def test_executor_rejects_lookalike_output_outside_supplied_root(tmp_path):
    lookalike = ROOT.parent / "outside_executor_test" / "reports/factory_bimanual"
    with pytest.raises(ValueError, match="inside project root"):
        ProductionExecutors(ROOT, output_root=lookalike)


def test_ik_artifact_requires_source_row_zero_to_succeed(tmp_path):
    executor = ProductionExecutors(
        tmp_path, spacing_by_robot={"xarm6": .8}
    )
    task = type("Task", (), {
        "time_s": np.array([0.0, 0.01]),
        "source_row_indices": np.array([0, 1]),
        "source_path": Path("task.csv"),
    })()
    result = type("Result", (), {
        "left_q": np.zeros((2, 6)), "right_q": np.zeros((2, 6)),
        "left_actual_tcp": np.zeros((2, 7)), "right_actual_tcp": np.zeros((2, 7)),
        "left_position_error_m": np.zeros(2), "right_position_error_m": np.zeros(2),
        "left_orientation_error_rad": np.zeros(2), "right_orientation_error_rad": np.zeros(2),
        "success": np.array([False, True]),
    })()

    summary = executor._write_ik(_job(), task, result)

    assert summary["initializer_valid"] is False
    assert "initializer_row" not in summary
    assert "initializer_source_index" not in summary


def test_common_branch_comparison_requires_row_zero_from_both_artifacts():
    strict = {"summary": {"initializer_valid": True, "initializer_row": 1,
                           "initializer_source_index": 1}}
    easy = {"summary": {"initializer_valid": True, "initializer_row": 0,
                         "initializer_source_index": 0}}

    with pytest.raises(RuntimeError, match="source row 0"):
        ProductionExecutors._common_initialization_row(strict, easy)


def test_mpc_rejects_initializer_from_later_row_even_if_marked_valid(tmp_path):
    executor = ProductionExecutors(
        tmp_path, spacing_by_robot={"xarm6": .8}
    )
    executor._load_ik = lambda *_: {
        "summary": {"initializer_valid": True, "initializer_row": 1,
                    "initializer_source_index": 1},
        "arrays": {"left_q": np.zeros((2, 6)), "right_q": np.zeros((2, 6))},
    }

    with pytest.raises(RuntimeError, match="source row 0"):
        executor.mpc(_job(stage="mpc", mode="mpc_easyik"))
