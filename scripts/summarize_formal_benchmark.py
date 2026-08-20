"""Summarize episode-disjoint formal topology evaluations into CSV/JSON.

The script intentionally reports the kinematic proxy-collision column as an
audit field rather than mixing it into the reachability ranking.  The generic
single-capsule proxy is topology biased; mesh-labelled collision results are a
separate benchmark stage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROBOTS = ("doosan", "xarm6", "ur5", "kinova")


def collect(report_dir: Path, prefix: str, split: str) -> list[dict]:
    rows: list[dict] = []
    for robot in ROBOTS:
        path = report_dir / f"{prefix}_{robot}_formal_{split}.json"
        guided = report_dir / f"1100_{robot}_guided_{split}.json"
        if guided.exists():
            path = guided
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            {
                "robot": robot,
                "search": "surrogate-guided" if "guided" in path.name else "nsga2",
                "split": split,
                "frames": int(payload["frames"]),
                "success_rate": float(metrics["success_rate"]),
                "position_rmse_mm": float(metrics["position_rmse_mm"]),
                "orientation_rmse_deg": float(metrics["orientation_rmse_deg"]),
                "sigma_mean": float(metrics["sigma_mean"]),
                "mount_roll_deg": float(payload["mount_roll_deg"]),
                "base_spacing_m": float(payload["base_spacing_m"]),
                "proxy_collision_fraction_audit_only": float(
                    metrics["proxy_collision_fraction"]
                ),
                "source": str(path),
            }
        )
    return sorted(rows, key=lambda row: (-row["success_rate"], row["position_rmse_mm"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path("reports/learning/hourly"))
    parser.add_argument("--prefix", default="1000")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = collect(args.report_dir, args.prefix, args.split)
    if not rows:
        raise SystemExit("No completed formal reports found")
    output = args.output or args.report_dir / f"{args.prefix}_formal_{args.split}_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(output)


if __name__ == "__main__":
    main()
