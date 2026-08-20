"""Fine-yaw and orientation-feasibility audit for the PiperX Fold Box mount."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.search_fold_box_piperx_mount import (
    _evaluate_side, _registered_task, rank_mount_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
WORKDIR = ROOT / ".tmp/piperx_fold_box_mount_refine"
OUTPUT = ROOT / ".tmp/piperx_mount_fine_audit.json"

CENTERS = {
    "left": {"xy": [-0.36121036783248495, -0.256828386388344],
             "yaw_deg": 90.0},
    "right": {"xy": [-0.21878963216751526, 0.22754635408558901],
              "yaw_deg": -120.0},
}


def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    task = _registered_task()
    strict = {}
    serial = 0
    for side in ("left", "right"):
        records = []
        center = CENTERS[side]
        for yaw in np.arange(center["yaw_deg"] - 20.0,
                             center["yaw_deg"] + 20.1, 5.0):
            mount = {"xy": center["xy"], "yaw_deg": float(yaw),
                     "base_z_m": .76, "roll_deg": 0.0, "pitch_deg": 0.0}
            record = _evaluate_side(
                task, side, mount, serial, WORKDIR, max_iterations=120,
                global_seed_count=8, uniform_count=30)
            serial += 1
            records.append(record)
            print(side, yaw, record["coverage"],
                  len(record["failed_ik_rows"]),
                  len(record["collision_only_rows"]), flush=True)
        records.sort(key=rank_mount_candidate)
        strict[side] = records

    relaxed = {}
    for side in ("left", "right"):
        mount = {key: strict[side][0][key]
                 for key in ("xy", "yaw_deg", "base_z_m", "roll_deg", "pitch_deg")}
        relaxed[side] = []
        for degrees in (1.5, 3.0, 5.0, 10.0):
            record = _evaluate_side(
                task, side, mount, serial, WORKDIR, max_iterations=160,
                global_seed_count=12, uniform_count=30,
                orientation_tolerance_rad=np.deg2rad(degrees))
            serial += 1
            relaxed[side].append(record)
            print(side, "orientation", degrees, record["coverage"], flush=True)

    payload = {"strict": strict, "orientation_audit": relaxed,
               "selected_mount": {
                   "xy": {side: strict[side][0]["xy"] for side in strict},
                   "yaw": {side: strict[side][0]["yaw_deg"] for side in strict},
                   "shared_base_z_m": .76,
               }}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["selected_mount"]), flush=True)


if __name__ == "__main__":
    main()
