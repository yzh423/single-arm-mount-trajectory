"""Search low, upright PiperX mounts that do not geometrically cross sides."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.search_fold_box_piperx_mount import (
    _evaluate_side, _registered_task, coarse_mount_candidates,
    rank_mount_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
WORKDIR = ROOT / ".tmp/piperx_fold_box_noncrossing"
OUTPUT = ROOT / ".tmp/piperx_mount_noncrossing.json"


def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    task = _registered_task()
    records = {}
    serial = 0
    for side in ("left", "right"):
        center = getattr(task, f"{side}_position_m")[:, :2].mean(axis=0)
        candidates = coarse_mount_candidates(center)
        candidates = [m | {"base_z_m": .76} for m in candidates
                      if ((side == "left" and m["xy"][1] >= center[1] + .15) or
                          (side == "right" and m["xy"][1] <= center[1] - .15))]
        side_records = []
        for mount in candidates:
            record = _evaluate_side(
                task, side, mount, serial, WORKDIR, max_iterations=100,
                global_seed_count=6, uniform_count=24)
            serial += 1
            side_records.append(record)
            print(side, mount["xy"], mount["yaw_deg"], record["coverage"],
                  record["collision_frames"], flush=True)
        side_records.sort(key=rank_mount_candidate)
        records[side] = side_records
    OUTPUT.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print("best", json.dumps({s: records[s][0] for s in records}), flush=True)


if __name__ == "__main__":
    main()
