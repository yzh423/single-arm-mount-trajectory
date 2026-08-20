"""Build tables, figures, and a PDF for the ten-arm three-episode study."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]


def provenance_for_report(path: Path | None = None) -> dict:
    path = path or ROOT / "reports/single_arm/official_model_provenance_gate.json"
    gate = json.loads(path.read_text(encoding="utf-8"))
    unqualified = list(gate.get("unqualified_ranking_robots", []))
    return {
        "model_gate_status": gate.get("status"),
        "geometry_policy": gate.get("geometry_policy"),
        "tcp_policy": gate.get("tcp_policy"),
        "unqualified_ranking_robots": unqualified,
        "ranking_disclosure_required": bool(unqualified),
    }


def load_rows(directory: Path) -> list[dict]:
    rows = []
    for path in sorted((directory / "cache").glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        arrays = np.load(path.with_suffix(".npz"), allow_pickle=False)
        failures = data.get("failure_reasons", {}).get("primary_frame_counts", {})
        rows.append({
            "robot": path.parent.name, "episode": path.stem,
            "episode_success": int(bool(data.get("episode_success"))),
            "frame_coverage": float(data.get("frame_coverage", 0.0)),
            "position_rmse_mm": 1000.0 * float(np.sqrt(np.mean(arrays["position_error_m"] ** 2))),
            "orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(arrays["orientation_error_rad"] ** 2)))),
            "base_x_m": data["base_xyz_m"][0], "base_y_m": data["base_xyz_m"][1],
            "base_z_m": data["base_xyz_m"][2], "tilt_deg": data["tilt_deg"],
            "yaw_deg": data["yaw_deg"], "roll_deg": data["roll_deg"],
            **{f"fail_{key}": int(value) for key, value in failures.items()},
        })
    return rows


def matrix(rows, robots, episodes, key):
    lookup = {(r["robot"], r["episode"]): r[key] for r in rows}
    return np.asarray([[lookup.get((robot, episode), np.nan) for robot in robots]
                       for episode in episodes], dtype=float)


def build_report(directory: Path, baseline: Path | None = None) -> None:
    rows = load_rows(directory)
    if not rows:
        raise ValueError(f"no completed cache audits in {directory}")
    robots = sorted({row["robot"] for row in rows})
    episodes = sorted({row["episode"] for row in rows})
    figures = directory / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (directory / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    success = matrix(rows, robots, episodes, "episode_success")
    coverage = matrix(rows, robots, episodes, "frame_coverage")
    rank = np.nanmean(success, axis=0)
    provenance = provenance_for_report()
    unqualified = set(provenance["unqualified_ranking_robots"])
    order = np.argsort(-rank)
    labels = [f"{robot}*" if robot in unqualified else robot for robot in np.asarray(robots)[order]]
    fig, ax = plt.subplots(figsize=(11, 5)); ax.bar(labels, rank[order])
    ax.set_ylim(0, 1); ax.set_ylabel("Complete-episode success rate"); ax.tick_params(axis="x", rotation=45)
    if unqualified:
        ax.set_title("* joint limits unverified; diagnostic result, excluded from unqualified ranking")
    fig.tight_layout(); fig.savefig(figures / "episode_success_ranking.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4)); image=ax.imshow(coverage, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(robots)), robots, rotation=45, ha="right"); ax.set_yticks(range(len(episodes)), episodes)
    fig.colorbar(image, ax=ax, label="Frame coverage"); fig.tight_layout()
    fig.savefig(figures / "episode_robot_coverage_heatmap.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([[r["position_rmse_mm"] for r in rows if r["robot"] == robot] for robot in robots], tick_labels=robots)
    axes[1].boxplot([[r["orientation_rmse_deg"] for r in rows if r["robot"] == robot] for robot in robots], tick_labels=robots)
    axes[0].set_ylabel("Position error (mm)"); axes[1].set_ylabel("Orientation error (deg)")
    for ax in axes: ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(figures / "pose_error_distributions.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key, label in zip(axes.ravel(), ("base_x_m","base_y_m","base_z_m","yaw_deg"), ("X (m)","Y (m)","Z (m)","Yaw (deg)")):
        for robot in robots:
            values=[r[key] for r in rows if r["robot"] == robot]; ax.scatter([robot]*len(values), values, s=22)
        ax.set_ylabel(label); ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(figures / "independent_mount_distributions.png", dpi=180); plt.close(fig)
    fail_keys=sorted({key for row in rows for key in row if key.startswith("fail_")})
    fail=np.asarray([[sum(r.get(key,0) for r in rows if r["robot"]==robot) for robot in robots] for key in fail_keys])
    fig,ax=plt.subplots(figsize=(12,5)); im=ax.imshow(fail,cmap="magma")
    ax.set_xticks(range(len(robots)),robots,rotation=45,ha="right");ax.set_yticks(range(len(fail_keys)),[k[5:] for k in fail_keys])
    fig.colorbar(im,ax=ax,label="Primary failed frames");fig.tight_layout();fig.savefig(figures/"failure_reason_heatmap.png",dpi=180);plt.close(fig)

    baseline_rows = load_rows(baseline) if baseline and baseline.is_dir() else []
    summary = {"jobs": len(rows), "episode_successes": int(success.sum()),
               "mean_frame_coverage": float(np.nanmean(coverage)),
               "per_robot_episode_success_rate": {robot: float(rank[i]) for i, robot in enumerate(robots)},
               "baseline_jobs": len(baseline_rows), "tilt_fixed_zero": all(abs(r["tilt_deg"]) < 1e-12 for r in rows),
               "roll_fixed_zero": all(abs(r["roll_deg"]) < 1e-12 for r in rows)}
    summary["model_provenance"] = provenance
    if baseline_rows:
        summary["baseline_episode_successes"] = sum(r["episode_success"] for r in baseline_rows)
        summary["baseline_mean_frame_coverage"] = float(np.mean([r["frame_coverage"] for r in baseline_rows]))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with PdfPages(directory / "report.pdf") as pdf:
        for name in ("episode_success_ranking.png", "episode_robot_coverage_heatmap.png",
                     "pose_error_distributions.png", "failure_reason_heatmap.png",
                     "independent_mount_distributions.png"):
            image=plt.imread(figures/name); fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(image);ax.axis("off");pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path);parser.add_argument("--baseline",type=Path)
    args=parser.parse_args();build_report(args.directory,args.baseline)


if __name__ == "__main__": main()
