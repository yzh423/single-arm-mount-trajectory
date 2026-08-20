"""Publication-ready figures for the formal ten-arm, two-task experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/single_arm/ten_arm_two_single_tasks"
FIGURES = OUT / "figures_publication"
ROBOTS = ("doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5",
          "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "openarm", "piperx")
TASKS = ("cap-left", "open-box-2")
TASK_COLORS = {"cap-left": "#4C78A8", "open-box-2": "#E28E2C"}
REASONS = ("joint_discontinuity", "self_collision", "table_collision",
           "position", "orientation", "position_and_orientation")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})


def load_rows() -> list[dict]:
    rows = []
    for robot in ROBOTS:
        for task in TASKS:
            path = ROOT / "videos/single_arm/strict_cache/local" / robot / f"{task}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            row.update(robot=robot, task=task, audit_path=str(path))
            rows.append(row)
    return rows


def export(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def write_source_data(rows: list[dict]) -> None:
    fields = ["robot", "task", "frames", "episode_success", "frame_coverage",
              "position_mean_mm", "position_p95_mm", "orientation_mean_deg", "orientation_p95_deg",
              "base_x_m", "base_y_m", "base_z_m", "tilt_deg", "yaw_deg", "roll_deg", *REASONS]
    with (FIGURES / "source_data.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            affected = row.get("failure_reasons", {}).get("affected_frame_counts", {})
            base = row["base_xyz_m"]
            writer.writerow({
                "robot": row["robot"], "task": row["task"], "frames": row.get("frames", 0),
                "episode_success": bool(row.get("episode_success")), "frame_coverage": row.get("frame_coverage", 0),
                "position_mean_mm": 1000*row["position_error_m"]["mean"], "position_p95_mm": 1000*row["position_error_m"]["p95"],
                "orientation_mean_deg": row["orientation_error_deg"]["mean"], "orientation_p95_deg": row["orientation_error_deg"]["p95"],
                "base_x_m": base[0], "base_y_m": base[1], "base_z_m": base[2],
                "tilt_deg": row.get("tilt_deg", 0), "yaw_deg": row.get("yaw_deg", 0), "roll_deg": row.get("roll_deg", 0),
                **{reason: affected.get(reason, 0) for reason in REASONS},
            })


def robot_order(rows: list[dict]) -> list[str]:
    return sorted(ROBOTS, key=lambda robot: (
        sum(bool(r.get("episode_success")) for r in rows if r["robot"] == robot),
        np.mean([r.get("frame_coverage", 0) for r in rows if r["robot"] == robot]),
    ), reverse=True)


def plot_ranking(rows: list[dict], order: list[str]) -> None:
    success = [50*sum(bool(r.get("episode_success")) for r in rows if r["robot"] == robot) for robot in order]
    coverage = [100*np.mean([r.get("frame_coverage", 0) for r in rows if r["robot"] == robot]) for robot in order]
    y = np.arange(len(order)); fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(y+.18, coverage, .34, color="#9CBAD4", label="Mean frame coverage")
    ax.barh(y-.18, success, .34, color="#285F8F", label="Complete-episode success")
    for yi, value in enumerate(success): ax.text(value+1, yi-.18, f"{value:.0f}%", va="center", fontsize=7)
    ax.set(yticks=y, yticklabels=order, xlim=(0, 108), xlabel="Rate (%)")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2); ax.legend(loc="lower right")
    ax.set_title("Ten-arm trajectory-following ranking (two tasks)", loc="left", weight="bold")
    fig.tight_layout(); export(fig, "01_episode_success_ranking")


def plot_coverage_heatmap(rows: list[dict], order: list[str]) -> None:
    matrix = np.array([[next(r for r in rows if r["robot"] == robot and r["task"] == task)["frame_coverage"] for robot in order] for task in TASKS])
    success = np.array([[next(r for r in rows if r["robot"] == robot and r["task"] == task)["episode_success"] for robot in order] for task in TASKS])
    fig, ax = plt.subplots(figsize=(8.0, 2.8)); image = ax.imshow(100*matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    for i in range(len(TASKS)):
        for j in range(len(order)):
            label = f"{100*matrix[i,j]:.1f}%" + ("\nPASS" if success[i,j] else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=6.5, color="white" if matrix[i,j] > .58 else "#17202A", weight="bold" if success[i,j] else "normal")
    ax.set(xticks=range(len(order)), xticklabels=order, yticks=range(len(TASKS)), yticklabels=TASKS)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right"); fig.colorbar(image, ax=ax, label="Frame coverage (%)", fraction=.025, pad=.02)
    ax.set_title("Task-by-robot frame coverage", loc="left", weight="bold"); fig.tight_layout(); export(fig, "02_task_robot_coverage_heatmap")


def plot_errors(rows: list[dict], order: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True)
    x = np.arange(len(order))
    for task, offset in zip(TASKS, (-.12, .12)):
        group = [next(r for r in rows if r["robot"] == robot and r["task"] == task) for robot in order]
        axes[0].scatter(x+offset, [1000*r["position_error_m"]["mean"] for r in group], s=30, color=TASK_COLORS[task], label=task, zorder=3)
        axes[1].scatter(x+offset, [r["orientation_error_deg"]["mean"] for r in group], s=30, color=TASK_COLORS[task], label=task, zorder=3)
    axes[0].set_ylabel("Mean position error (mm)"); axes[1].set_ylabel("Mean orientation error (deg)")
    for ax in axes:
        ax.set_xticks(x, order, rotation=45, ha="right"); ax.grid(axis="y", alpha=.2)
    axes[0].legend(); axes[0].set_title("Position error", loc="left", weight="bold"); axes[1].set_title("Orientation error", loc="left", weight="bold")
    fig.suptitle("Per-episode tracking-error distribution", x=.07, ha="left", weight="bold"); fig.tight_layout(); export(fig, "03_position_orientation_error_distribution")


def plot_failures(rows: list[dict], order: list[str]) -> None:
    matrix = []
    for robot in order:
        group = [r for r in rows if r["robot"] == robot]
        total_frames = sum(max(1, int(r.get("frames", 1))) for r in group)
        matrix.append([100*sum(r.get("failure_reasons", {}).get("affected_frame_counts", {}).get(reason, 0) for r in group)/total_frames for reason in REASONS])
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7.4, 4.8)); image = ax.imshow(matrix, vmin=0, vmax=max(1, np.percentile(matrix, 95)), cmap="magma", aspect="auto")
    for i in range(len(order)):
        for j in range(len(REASONS)): ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=6, color="white" if matrix[i,j] > .55*max(1, np.percentile(matrix, 95)) else "#17202A")
    labels = ("Joint discontinuity", "Self collision", "Table collision", "Position", "Orientation", "Position + orientation")
    ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(order)), yticklabels=order); plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(image, ax=ax, label="Affected frames / total frames (%)", fraction=.035, pad=.02)
    ax.set_title("Failure-reason severity", loc="left", weight="bold"); fig.tight_layout(); export(fig, "04_failure_reason_heatmap")


def plot_mounts(rows: list[dict], order: list[str]) -> None:
    keys = ("base_x", "base_y", "base_z", "tilt", "yaw", "roll")
    labels = ("X (m)", "Y (m)", "Z (m)", "Tilt (deg)", "Yaw (deg)", "Roll (deg)")
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.2), sharex=True); x = np.arange(len(order))
    for ax, key, label in zip(axes.flat, keys, labels):
        for task, offset in zip(TASKS, (-.12, .12)):
            group = [next(r for r in rows if r["robot"] == robot and r["task"] == task) for robot in order]
            if key.startswith("base_"):
                idx = {"base_x": 0, "base_y": 1, "base_z": 2}[key]; values = [r["base_xyz_m"][idx] for r in group]
            else: values = [r.get(f"{key}_deg", 0) for r in group]
            ax.scatter(x+offset, values, s=25, color=TASK_COLORS[task], label=task, zorder=3)
        ax.axhline(0, color="#999999", lw=.6, zorder=1); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2)
        ax.set_xticks(x, order, rotation=48, ha="right", fontsize=6.5)
    axes[0,0].legend(); fig.suptitle("Task-specific optimal six-dimensional mounts", x=.06, ha="left", weight="bold")
    fig.tight_layout(); export(fig, "05_optimal_mount_distribution")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_rows(); order = robot_order(rows); write_source_data(rows)
    plot_ranking(rows, order); plot_coverage_heatmap(rows, order); plot_errors(rows, order)
    plot_failures(rows, order); plot_mounts(rows, order)
    (FIGURES / "figure_manifest.json").write_text(json.dumps({"robots": list(ROBOTS), "tasks": list(TASKS), "order": order, "formats": ["png", "svg", "pdf"], "source_data": "source_data.csv"}, indent=2), encoding="utf-8")
    print(FIGURES)


if __name__ == "__main__":
    main()
