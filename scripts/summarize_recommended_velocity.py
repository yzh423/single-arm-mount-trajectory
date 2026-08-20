"""Check velocity feasibility of the final fixed-upright recommendations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "xarm_recommended": ("xarm6", "optimized_700_750_xarm6_full_mpc_aggressive_branchvel0p02_densebase_localscene_finalbranchvel.npz"),
    "doosan_precision": ("doosan_m0609", "optimized_700_750_doosan_full_mpc_densebase.npz"),
    "doosan_coverage": ("doosan_m0609", "optimized_700_750_doosan_full_mpc_densebase_localscene_dense2joint.npz"),
    "ur5_recommended": ("ur5_classic", "optimized_700_750_ur5_full_mpc_densebase_localscene_jointbase.npz"),
    "kinova_control": ("kinova_gen3_lite", "optimized_700_750_kinova_full_mpc_densebase.npz"),
}


def main() -> None:
    limits = json.loads((ROOT / "configs" / "robot_limits.json").read_text())
    rows = []
    for variant, (limit_key, filename) in CASES.items():
        with np.load(ROOT / "offline_results" / filename) as data:
            q = data["q"]
            time_s = data["time_s"]
        dt = np.diff(time_s)
        valid = (dt > 1e-4) & (dt < .2)
        dq = (np.diff(q, axis=0) + np.pi) % (2 * np.pi) - np.pi
        qdot = np.abs(dq[valid] / dt[valid, None])
        configured = limits[limit_key]
        joint_limits = np.asarray(configured["joint_velocity_rad_s"])
        ratio = qdot / joint_limits
        position_fraction = ""
        if "joint_lower_rad" in configured and "joint_upper_rad" in configured:
            lower = np.asarray(configured["joint_lower_rad"])
            upper = np.asarray(configured["joint_upper_rad"])
            position_fraction = float(np.mean(np.all((q >= lower) & (q <= upper), axis=1)))
        rows.append({
            "variant": variant,
            **{f"p99_j{i+1}_rad_s": float(value) for i, value in enumerate(np.percentile(qdot, 99, axis=0))},
            "fraction_frames_all_joints_within_limit": float(np.mean(np.all(ratio <= 1, axis=1))),
            "minimum_p99_retime_factor": float(np.percentile(np.max(ratio, axis=1), 99)),
            "fraction_frames_all_joints_within_position_limits": position_fraction,
            "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(qdot * qdot))),
            "mean_sum_absolute_joint_velocity_rad_s": float(np.mean(np.sum(qdot, axis=1))),
            "total_joint_path_l2_rad": float(np.sum(np.linalg.norm(dq[valid], axis=1))),
            "valid_velocity_frames": int(valid.sum()),
        })
    destination = ROOT / "reports" / "700_750_recommended_velocity.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(destination)


if __name__ == "__main__":
    main()
