from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ("doosan", "xarm6", "ur5", "kinova")


def main() -> None:
    report_dir = ROOT / "reports/learning/collision_formal"
    rows = []
    for robot in ROBOTS:
        test_path = report_dir / f"{robot}_test.json"
        finalist_path = ROOT / "runs/collision_formal" / robot / "pareto_finalists.json"
        if not test_path.exists() or not finalist_path.exists():
            continue
        test = json.loads(test_path.read_text()); payload = json.loads(finalist_path.read_text())
        candidate = sorted(payload["pareto_candidates"], key=lambda row: row["objectives"][0])[0]
        audit_path = report_dir / f"{robot}_episode563_candidate_audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        metrics = test["metrics"]
        rows.append({
            "robot": robot, "test_success_pct": 100 * metrics["success_rate"],
            "position_rmse_mm": metrics["position_rmse_mm"],
            "orientation_rmse_deg": metrics["orientation_rmse_deg"],
            "sigma_mean": metrics["sigma_mean"],
            "collision_pct": 100 * metrics["proxy_collision_fraction"],
            "minimum_clearance_mm": metrics["proxy_minimum_clearance_mm"],
            "base_spacing_m": candidate["base_spacing_m"], "base_y_m": candidate["base_y_m"],
            "base_z_m": candidate["base_z_m"], "mount_roll_deg": candidate["mount_roll_deg"],
            "episode_safe_pair_pct": 100 * audit.get("any_collision_free_pair_fraction", float("nan")),
            "episode_safe_success_pair_pct": 100 * audit.get(
                "any_collision_free_success_pair_fraction", float("nan")),
        })
    csv_path = report_dir / "summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=180)
        labels = [row["robot"] for row in rows]
        for axis, key, title in zip(
                axes, ("test_success_pct", "position_rmse_mm", "minimum_clearance_mm"),
                ("Safe pointwise success (%)", "Position RMSE (mm)", "Minimum clearance (mm)")):
            values = [row[key] for row in rows]; axis.bar(labels, values)
            axis.set_title(title); axis.grid(axis="y", alpha=.25)
            for index, value in enumerate(values): axis.text(index, value, f"{value:.2f}", ha="center")
        fig.tight_layout(); fig.savefig(report_dir / "summary.png", bbox_inches="tight")
    print(csv_path)


if __name__ == "__main__":
    main()
