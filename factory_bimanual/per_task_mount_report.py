"""Evidence-preserving metric tables for per-task mount searches."""
from __future__ import annotations

import csv
from pathlib import Path


METRICS = (
    "source_row_count", "audited_source_rows",
    "pair_collision_frames", "pair_edge_collision_frames",
    "synchronous_strict_coverage", "longest_failure_frames",
    "mean_pair_pose_error", "p95_pair_pose_error",
    "minimum_joint_limit_margin_rad", "p10_pair_singularity_margin",
    "base_distance_m",
)


def _row(task_name, scope, mode, status, layout):
    row = {"task_name": task_name, "scope": scope,
           "mode": mode or "", "status": status}
    for key in METRICS:
        row[key] = "" if layout is None else layout.get(key, "")
    return row


def build_metric_rows(selected_layouts):
    mode_rows = []
    final_rows = []
    for task_name in sorted(selected_layouts):
        task = selected_layouts[task_name]
        for mode in ("upright_table", "horizontal_forward", "inverted"):
            value = task.get("modes", {}).get(mode, {
                "status": "missing", "selected_layout": None})
            mode_rows.append(_row(
                task_name, "mode", mode, value["status"],
                value.get("selected_layout")))
        layout = task.get("selected_layout")
        final_rows.append(_row(
            task_name, "selected",
            None if layout is None else layout.get("mode"),
            task.get("status", "missing"), layout))
    return mode_rows, final_rows


def write_comparison_metrics(mode_rows, final_rows, output: Path):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fields = ("task_name", "scope", "mode", "status", *METRICS)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*mode_rows, *final_rows])
    temporary.replace(output)
