"""Resume Fold Box PiperX mount refinement at the final collision fidelity."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.search_fold_box_piperx_mount import (
    _registered_task, rank_paired_mount_candidate, select_paired_mount)
from scripts.search_fold_box_piperx_paired_mount import (
    OUT, evaluate_pair, local_paired_refinements, _valid_mount)


START = {
    "xy": {"left": [-0.2712103678324848, 0.0269022832011202],
           "right": [0.0412103678324847, -0.3569022832011202]},
    "yaw": {"left": 15.0, "right": 45.0},
    "shared_base_z_m": 0.81,
}


def safe(record):
    return (record["pair_collision_frames"] == 0
            and record["pair_edge_collision_frames"] == 0)


def main():
    task = _registered_task(); records = []; serial = 200
    seeds = []
    for side in ("left", "right"):
        for mount in local_paired_refinements(
                START, side=side, xy_step_m=.04, yaw_step_deg=15.0):
            if not _valid_mount(mount):
                continue
            record = evaluate_pair(
                task, mount, serial, uniform_count=40, global_seed_count=8,
                max_iterations=120, maximum_candidates=4)
            serial += 1; records.append(record)
            if safe(record): seeds.append(record)
            print("final-neighborhood", side,
                  record["synchronous_pair_coverage"],
                  record["pair_collision_frames"],
                  record["pair_edge_collision_frames"], flush=True)
    if not seeds:
        raise RuntimeError("no collision-free mount in final XY/yaw neighborhood")
    seed = min(seeds, key=rank_paired_mount_candidate)
    verification = evaluate_pair(
        task, seed["mount"], serial, uniform_count=60, global_seed_count=12,
        max_iterations=140, maximum_candidates=6)
    records.append(verification)
    if not safe(verification):
        raise RuntimeError("best mount failed 60-frame collision verification")
    selected = select_paired_mount([verification])
    payload = {
        "robot": "piperx", "task": "fold_box",
        "search_method": (
            "paired strict 6D IK mount search with hard state and swept-edge "
            "collision rejection; resumed final-fidelity XY/yaw refinement"),
        "selected_mount": selected, "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("selected", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
