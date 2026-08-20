"""Summarize the link/offset plus universal-base validation layer."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOPS = ("doosan", "xarm6", "ur5", "kinova")


def result_path(topology: str) -> Path:
    aggressive = "_aggressive" if topology == "xarm6" else ""
    return ROOT / "offline_results" / (
        f"optimized_700_750_{topology}_full_mpc{aggressive}"
        "_densebase_localscene_jointbase.json"
    )


def main() -> None:
    rows = []
    for topology in TOPS:
        geometry = json.loads((ROOT / "offline_results" / f"geometry_700_750_{topology}_local_dense.json").read_text())
        base = json.loads((ROOT / "offline_results" / f"geometry_700_750_{topology}_universal_base_local_dense_joint.json").read_text())
        result = json.loads(result_path(topology).read_text())
        best = geometry["best"]
        lengths_mm = 1000 * np.linalg.norm(np.asarray(best["joint_and_tcp_deltas_m"]), axis=1)
        rows.append({
            "topology": topology,
            "reach_m": best["reach_m"],
            "axis_delta_lengths_mm": "/".join(f"{x:.1f}" for x in lengths_mm),
            "base_xyz_droid_m": "/".join(f"{x:.6f}" for x in base["best"]["base_xyz_droid_m"]),
            "ik_proxy_score": base["best"]["score"],
            "mpc_success_fraction": result["success_fraction"],
            "position_rmse_mm": result["position_rmse_mm"],
            "orientation_rmse_deg": result["orientation_rmse_deg"],
            "sigma_min": result["sigma_min"],
            "penetrating_frames": result["penetrating_frames"],
            "samples": result["samples"],
        })
    dst = ROOT / "reports" / "700_750_joint_geometry_base_validation.csv"
    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(dst)


if __name__ == "__main__":
    main()
