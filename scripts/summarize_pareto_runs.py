from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("doosan", "xarm6", "ur5", "kinova")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "pareto_quick")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "learning" / "pareto")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    payload, histories = {}, {}
    for name in NAMES:
        payload[name] = json.loads((args.run_root / name / "pareto_finalists.json").read_text())
        histories[name] = [json.loads(line) for line in (args.run_root / name / "history.jsonl").read_text().splitlines()]
    (args.output_dir / "pareto_finalists.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, name in zip(axes.flat, NAMES):
        candidates = payload[name]["pareto_candidates"]
        objective = np.asarray([row["objectives"] for row in candidates])
        success = 100 * (1 - objective[:, 0])
        scatter = ax.scatter(1000 * objective[:, 1], np.degrees(objective[:, 2]), c=success,
                             cmap="viridis", vmin=0, vmax=100, s=90, edgecolor="black")
        ax.set_title(f"{name}: retained Pareto finalists")
        ax.set_xlabel("position RMSE (mm), lower is better")
        ax.set_ylabel("orientation RMSE (deg), lower is better")
        ax.grid(alpha=.25); fig.colorbar(scatter, ax=ax, label="success rate (%)")
    front_path = args.output_dir / "pareto_fronts.png"; fig.savefig(front_path, dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    fields = (("best_success_rate", "best success (%)", 100),
              ("best_position_rmse_mm", "best position RMSE (mm)", 1),
              ("best_orientation_rmse_deg", "best orientation RMSE (deg)", 1))
    for ax, (field, ylabel, factor) in zip(axes, fields):
        for name in NAMES:
            rows = histories[name]
            ax.plot([r["generation"] + 1 for r in rows], [factor * r[field] for r in rows], marker="o", label=name)
        ax.set_xlabel("generation"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend()
    convergence_path = args.output_dir / "pareto_convergence.png"
    fig.savefig(convergence_path, dpi=220); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, name in zip(axes.flat, NAMES):
        candidates = payload[name]["pareto_candidates"]
        roll = np.asarray([row["mount_roll_deg"] for row in candidates])
        success = 100 * np.asarray([1 - row["objectives"][0] for row in candidates])
        collision = 100 * np.asarray([row["objectives"][4] for row in candidates])
        scatter = ax.scatter(roll, success, c=collision, cmap="magma_r", s=100,
                             edgecolor="black", vmin=0, vmax=max(1, collision.max()))
        ax.set_title(name); ax.set_xlim(0, 180); ax.set_xticks((0, 45, 90, 135, 180))
        ax.set_xlabel("mount roll (deg)"); ax.set_ylabel("IK success (%)"); ax.grid(alpha=.25)
        fig.colorbar(scatter, ax=ax, label="reported collision frames (%)")
    mount_path = args.output_dir / "mount_roll_pareto.png"
    fig.savefig(mount_path, dpi=220); plt.close(fig)
    print(front_path); print(convergence_path); print(mount_path)


if __name__ == "__main__": main()
