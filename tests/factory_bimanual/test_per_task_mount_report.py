import csv

from factory_bimanual.per_task_mount_report import (
    build_metric_rows, write_comparison_metrics,
)


def _selected():
    result = {}
    for task_index in range(11):
        modes = {}
        for mode in ("upright_table", "horizontal_forward", "inverted"):
            layout = {
                "mode": mode, "source_row_count": 10,
                "audited_source_rows": 10, "pair_collision_frames": 0,
                "pair_edge_collision_frames": 0,
                "synchronous_strict_coverage": .8,
                "mean_pair_pose_error": .002,
                "minimum_joint_limit_margin_rad": .1,
                "p10_pair_singularity_margin": .05,
            }
            modes[mode] = {"status": "complete", "selected_layout": layout}
        result[f"task{task_index}"] = {
            "status": "complete", "modes": modes,
            "selected_layout": modes["upright_table"]["selected_layout"]}
    return result


def test_report_rows_have_33_modes_and_11_final_selections(tmp_path):
    mode_rows, final_rows = build_metric_rows(_selected())
    assert len(mode_rows) == 33
    assert len(final_rows) == 11
    output = tmp_path / "metrics.csv"
    write_comparison_metrics(mode_rows, final_rows, output)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 44
    assert {row["scope"] for row in rows} == {"mode", "selected"}


def test_report_keeps_explicit_no_safe_layout():
    selected = _selected()
    selected["task0"] = {"status": "no_safe_layout", "modes": {
        mode: {"status": "no_safe_layout", "selected_layout": None}
        for mode in ("upright_table", "horizontal_forward", "inverted")},
        "selected_layout": None}
    mode_rows, final_rows = build_metric_rows(selected)
    assert [row for row in final_rows if row["task_name"] == "task0"][0][
        "status"] == "no_safe_layout"
