"""Generate report figures for the multitask four-mount study."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
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


def coverage_matrix(rows, field="both_accept_coverage"):
    trajectories = tuple(sorted({row["trajectory"] for row in rows}))
    lookup = {(row["trajectory"], row["mode"]): float(row[field])
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


def _heatmap(rows, field, output, *, title, color_label,
             percent=False, cmap="viridis"):
    trajectories, matrix = coverage_matrix(rows, field)
    display = 100.0 * matrix if percent else matrix
    fig_height = max(7.0, 0.34 * len(trajectories) + 2.6)
    fig, axis = plt.subplots(figsize=(9.2, fig_height), constrained_layout=True)
    image = axis.imshow(display, aspect="auto", cmap=cmap)
    axis.set_xticks(range(4), [MOUNT_LABELS[mode] for mode in STUDY_PANEL_ORDER])
    axis.set_yticks(range(len(trajectories)), _labels(trajectories), fontsize=8)
    axis.set_title(title, fontsize=15, pad=14)
    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            value = display[row, column]
            label = f"{value:.1f}%" if percent else f"{value:.2f}"
            axis.text(column, row, label, ha="center", va="center",
                      fontsize=7, color=("white" if value > np.nanmedian(display) else "black"))
    bar = fig.colorbar(image, ax=axis, shrink=.74)
    bar.set_label(color_label)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate_figures(aggregate_csv, output_dir):
    rows = _read_rows(aggregate_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Microsoft YaHei",
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
        rows, "longest_hold_frames", output_dir / "longest_hold_heatmap.png",
        title="最长连续 HOLD 帧数", color_label="帧",
        cmap="magma_r")
    collision_rows = []
    for row in rows:
        collision_rows.append({
            **row,
            "unsafe_frames": int(float(row["collision_frames"]))
            + int(float(row["edge_collision_frames"]))
            + int(float(row["topology_invalid_frames"])),
        })
    _heatmap(
        collision_rows, "unsafe_frames",
        output_dir / "collision_topology_heatmap.png",
        title="碰撞、扫掠边碰撞与拓扑违规帧",
        color_label="不安全帧/边", cmap="Reds")
    _heatmap(
        rows, "maximum_position_error_mm",
        output_dir / "maximum_position_error_heatmap.png",
        title="最大 TCP 位置误差", color_label="mm", cmap="magma")
    _heatmap(
        rows, "maximum_velocity_rad_s",
        output_dir / "velocity_heatmap.png",
        title="原始节奏最大关节速度", color_label="rad/s", cmap="plasma")
    _heatmap(
        rows, "maximum_acceleration_rad_s2",
        output_dir / "acceleration_heatmap.png",
        title="原始节奏最大关节加速度", color_label="rad/s²", cmap="plasma")

    counts = winner_counts(rows)
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    modes = list(STUDY_PANEL_ORDER)
    bars = axis.bar(
        [MOUNT_LABELS[mode] for mode in modes],
        [counts[mode] for mode in modes],
        color=[MOUNT_COLORS[mode] for mode in modes])
    axis.set_title("安全门优先、再按双臂同时 ACCEPT 选择的构型胜者", fontsize=15)
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
