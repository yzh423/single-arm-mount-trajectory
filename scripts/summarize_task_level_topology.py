"""Export per-task metrics for the final fixed-upright topology variants."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "xarm_recommended": "optimized_700_750_xarm6_full_mpc_aggressive_branchvel0p02_densebase_localscene_finalbranchvel.json",
    "doosan_precision": "optimized_700_750_doosan_full_mpc_densebase.json",
    "doosan_coverage": "optimized_700_750_doosan_full_mpc_densebase_localscene_dense2joint.json",
    "ur5_recommended": "optimized_700_750_ur5_full_mpc_densebase_localscene_jointbase.json",
}


def main() -> None:
    data = {name: json.loads((ROOT / "offline_results" / filename).read_text()) for name, filename in FILES.items()}
    rows = []
    for task_index in range(18):
        for variant, result in data.items():
            row = result["results"][task_index]
            rows.append({
                "task_index": task_index,
                "instruction": row["instruction"],
                "variant": variant,
                "success_fraction": row["success_fraction"],
                "position_rmse_mm": row["position_rmse_mm"],
                "orientation_rmse_deg": row["orientation_rmse_deg"],
                "sigma_min": row["sigma_min"],
                "penetrating_frames": row["penetrating_frames"],
            })
    dst = ROOT / "reports" / "700_750_task_level_comparison.csv"
    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(dst)


if __name__ == "__main__":
    main()
