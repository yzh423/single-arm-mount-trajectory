"""Generate report figures for the multitask four-mount study."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from factory_bimanual.mount_comparison_visuals import (
    MOUNT_COLORS,
    MOUNT_LABELS,
    STUDY_PANEL_ORDER,
    audited_mount_rank,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/piperx_multitask_fixed_time_mount_study"


def _preferred_font_family():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
            "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"):
        if candidate in installed:
            return candidate
    return "sans-serif"


def coverage_matrix(rows, field="both_accept_coverage"):
    trajectories = tuple(sorted({row["trajectory"] for row in rows}))
    def numeric(value):
        return np.nan if value in (None, "") else float(value)
    lookup = {(row["trajectory"], row["mode"]): numeric(row[field])
              for row in rows}
    expected = {(trajectory, mode) for trajectory in trajectories
                for mode in STUDY_PANEL_ORDER}
    missing = expected - set(lookup)
    if missing:
        raise ValueError(f"aggregate matrix is incomplete: {sorted(missing)[:3]}")
    return trajectories, np.asarray([
        [lookup[(trajectory, mode)] for mode in STUDY_PANEL_ORDER]
        for trajectory in trajectories], dtype=float)


def winner_counts(rows):
    trajectories = sorted({row["trajectory"] for row in rows})
    counts = {mode: 0 for mode in STUDY_PANEL_ORDER}
    for trajectory in trajectories:
        candidates = [row for row in rows if row["trajectory"] == trajectory]
        winner = min(candidates, key=audited_mount_rank)
        counts[winner["mode"]] += 1
    return counts


def _read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _labels(trajectories):
    return [value.replace("/", " · ") for value in trajectories]


def _mount_tick_labels():
    """Keep the bilingual installation labels legible in narrow columns."""
    return [
        MOUNT_LABELS[mode].replace(" / ", "\n/ ")
        for mode in STUDY_PANEL_ORDER
    ]


def _heatmap(rows, field, output, *, title, color_label,
             percent=False, cmap="viridis", vmin=None, vmax=None,
             formatter=None):
    trajectories, matrix = coverage_matrix(rows, field)
    display = 100.0 * matrix if percent else matrix
    fig_height = max(7.0, 0.34 * len(trajectories) + 2.6)
    fig, axis = plt.subplots(figsize=(9.2, fig_height), constrained_layout=True)
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#D7DEE7")
    image = axis.imshow(
        np.ma.masked_invalid(display), aspect="auto", cmap=color_map,
        vmin=vmin, vmax=vmax)
    axis.set_xticks(range(4), _mount_tick_labels(), fontsize=8)
    axis.set_yticks(range(len(trajectories)), _labels(trajectories), fontsize=8)
    axis.set_title(title, fontsize=15, pad=14)
    finite = display[np.isfinite(display)]
    midpoint = float(np.median(finite)) if finite.size else 0.0
    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            value = display[row, column]
            if formatter is not None:
                label = formatter(value)
            elif not np.isfinite(value):
                label = "HOLD / N/A"
            else:
                label = f"{value:.1f}%" if percent else f"{value:.2f}"
            axis.text(column, row, label, ha="center", va="center",
                      fontsize=7, color=(
                          "white" if np.isfinite(value) and value > midpoint
                          else "black"))
    bar = fig.colorbar(image, ax=axis, shrink=.74)
    bar.set_label(color_label)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate_figures(aggregate_csv, output_dir):
    rows = _read_rows(aggregate_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": _preferred_font_family(),
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
    })
    coverage_cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#9B1C31", "#F2C94C", "#219653"])
    _heatmap(
        rows, "both_accept_coverage", output_dir / "coverage_heatmap.png",
        title="27 条双手轨迹的 Fixed-Time 双臂同时 ACCEPT",
        color_label="双臂同时 ACCEPT (%)", percent=True,
        cmap=coverage_cmap)
    _heatmap(
        rows, "longest_hold_ratio", output_dir / "longest_hold_heatmap.png",
        title="最长连续 HOLD 占源轨迹比例", color_label="轨迹比例 (%)",
        percent=True, cmap="magma_r", vmin=0., vmax=100.)
    collision_rows = []
    for row in rows:
        collision_rows.append({
            **row,
            "safety_pass": float(
                int(float(row["collision_frames"]))
                + int(float(row["edge_collision_frames"]))
                + int(float(row["topology_invalid_frames"])) == 0),
        })
    _heatmap(
        collision_rows, "safety_pass",
        output_dir / "collision_topology_heatmap.png",
        title="碰撞、扫掠边碰撞与拓扑硬门",
        color_label="0 = FAIL, 1 = PASS", cmap="RdYlGn", vmin=0., vmax=1.,
        formatter=lambda value: "PASS" if value == 1. else "FAIL")
    _heatmap(
        rows, "maximum_accepted_position_error_mm",
        output_dir / "maximum_position_error_heatmap.png",
        title="ACCEPT 帧最大 TCP 位置误差", color_label="mm", cmap="magma",
        vmin=0., vmax=1.)
    _heatmap(
        rows, "maximum_accepted_orientation_error_deg",
        output_dir / "maximum_orientation_error_heatmap.png",
        title="ACCEPT 帧最大 TCP 姿态误差", color_label="deg", cmap="magma",
        vmin=0., vmax=.5)
    dynamic_rows = [{
        **row,
        "display_velocity": (
            row["maximum_velocity_rad_s"]
            if float(row["both_accept_coverage"]) > 0. else None),
        "display_acceleration": (
            row["maximum_acceleration_rad_s2"]
            if float(row["both_accept_coverage"]) > 0. else None),
        "deployable": float(
            float(row["both_accept_coverage"]) >= 1.0 - 1e-12
            and str(row["limits_passed"]).lower() == "true"
            and int(float(row["collision_frames"])) == 0
            and int(float(row["edge_collision_frames"])) == 0
            and int(float(row["topology_invalid_frames"])) == 0),
    } for row in rows]
    _heatmap(
        dynamic_rows, "display_velocity",
        output_dir / "velocity_heatmap.png",
        title="原始节奏最大关节速度", color_label="rad/s", cmap="plasma")
    _heatmap(
        dynamic_rows, "display_acceleration",
        output_dir / "acceleration_heatmap.png",
        title="原始节奏最大关节加速度", color_label="rad/s²", cmap="plasma")
    _heatmap(
        dynamic_rows, "deployable", output_dir / "deployability_heatmap.png",
        title="100% 覆盖 + 动力学 + 安全联合可部署门",
        color_label="0 = FAIL, 1 = PASS", cmap="RdYlGn", vmin=0., vmax=1.,
        formatter=lambda value: "PASS" if value == 1. else "FAIL")

    counts = winner_counts(rows)
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    modes = list(STUDY_PANEL_ORDER)
    bars = axis.bar(
        _mount_tick_labels(),
        [counts[mode] for mode in modes],
        color=[MOUNT_COLORS[mode] for mode in modes])
    axis.set_title("严格位姿覆盖率构型胜者（不包含动力学门）", fontsize=15)
    axis.set_ylabel("轨迹数量")
    axis.set_ylim(0, max(counts.values(), default=0) + 3)
    axis.bar_label(bars, fontsize=12)
    fig.savefig(output_dir / "winner_counts.png", dpi=180)
    plt.close(fig)
    return tuple(sorted(output_dir.glob("*.png")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for path in generate_figures(
            args.output / "aggregate.csv", args.output / "figures"):
        print(path)


if __name__ == "__main__":
    main()
