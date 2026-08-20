"""Common comparison matrix and conservative human-readable report."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ReportPaths:
    matrix_json: Path
    matrix_csv: Path
    markdown: Path


def write_comparison_report(
    root: Path, rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> ReportPaths:
    root = Path(root).resolve()
    if not root.as_posix().rstrip("/").lower().endswith("/reports/factory_bimanual"):
        raise ValueError("report root must end with reports/factory_bimanual")
    root.mkdir(parents=True, exist_ok=True)
    normalized = [dict(row) for row in rows]
    required = {"robot", "task", "mode", "ik_planning_success", "mpc_execution_success"}
    for row in normalized:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"report row missing fields: {sorted(missing)}")
    matrix_json = root / "matrix.json"
    matrix_csv = root / "matrix.csv"
    markdown = root / "REPORT.md"
    matrix_json.write_text(json.dumps({"schema_version": 1, "provenance": dict(provenance),
                                       "rows": normalized}, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    fields = sorted(set().union(*(row.keys() for row in normalized))) if normalized else sorted(required)
    with matrix_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)
    lines = [
        "# Factory Bimanual Comparison Report", "",
        "Results are the best observed under the declared fixed-spacing and solver/control budgets.", "",
        "IK planning success and MPC execution success are reported independently.", "",
        "## Comparison matrix", "",
        "| Robot | Task | Mode | IK planning success | MPC execution success |",
        "|---|---|---|---:|---:|",
    ]
    for row in normalized:
        lines.append(f"| {row['robot']} | {row['task']} | {row['mode']} | "
                     f"{row['ik_planning_success']} | {row['mpc_execution_success']} |")
    lines += ["", "## Limitations", "",
              "MuJoCo results do not establish real-hardware feasibility. Force/torque and material, fluid, cap, container, and other unmodelled contact effects are outside scope.", "",
              "Failed runs and their original-timeline diagnostics are retained in the run artifacts.", ""]
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return ReportPaths(matrix_json, matrix_csv, markdown)
