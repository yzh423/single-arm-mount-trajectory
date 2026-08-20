import csv
import json
from pathlib import Path

from factory_bimanual.report import write_comparison_report


def test_report_separates_ik_planning_and_mpc_execution(tmp_path: Path):
    root = tmp_path / "reports" / "factory_bimanual"
    rows = [{
        "robot": "panda", "task": "pour", "mode": "strict_a",
        "ik_planning_success": True, "mpc_execution_success": False,
        "bimanual_coverage": 0.9, "longest_failure_s": 0.2,
    }]
    paths = write_comparison_report(root, rows, provenance={"note": "synthetic"})
    matrix = json.loads(paths.matrix_json.read_text(encoding="utf-8"))
    assert matrix["rows"][0]["ik_planning_success"] is True
    assert matrix["rows"][0]["mpc_execution_success"] is False
    with paths.matrix_csv.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "ik_planning_success" in header
    assert "mpc_execution_success" in header
    text = paths.markdown.read_text(encoding="utf-8")
    assert "best observed under the declared fixed-spacing and solver/control budgets" in text
    assert "unmodelled contact" in text.lower()
