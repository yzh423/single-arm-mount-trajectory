"""Search a well-separated paired PiperX mount for fixed-time Fold Box."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from factory_bimanual.fixed_time_run_contract import (
    validate_synchronized_mount,
)
from scripts.search_fold_box_piperx_mount import (
    _registered_task,
    rank_paired_mount_candidate,
)
from scripts.search_fold_box_piperx_paired_mount import (
    evaluate_full_pair,
    evaluate_pair,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (ROOT / "reports/factory_bimanual/fold_box_dual_piperx/"
          "fold_box_piperx_fixed_time_mount_search.json")


def synchronized_mount_candidates(*, maximum=32):
    """Return a deterministic diverse subset of paired tabletop mounts."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    raw = []
    yaw_pairs = (
        (15.0, 15.0), (0.0, 0.0), (30.0, 30.0), (-15.0, -15.0),
        (15.0, -15.0), (-15.0, 15.0), (30.0, 0.0), (0.0, 30.0),
    )
    for center_x in (-0.35, -0.25, -0.15, -0.05):
        for center_y in (-0.10, 0.0, 0.10):
            for separation in (0.60, 0.70, 0.80):
                for skew_x in (-0.10, 0.0, 0.10):
                    for left_yaw, right_yaw in yaw_pairs:
                        for shared_z in (0.81, 0.86):
                            mount = {
                                "xy": {
                                    "left": [center_x - 0.5 * skew_x,
                                             center_y + 0.5 * separation],
                                    "right": [center_x + 0.5 * skew_x,
                                              center_y - 0.5 * separation],
                                },
                                "yaw": {"left": left_yaw,
                                        "right": right_yaw},
                                "shared_base_z_m": shared_z,
                                "roll": {"left": 0.0, "right": 0.0},
                                "pitch": {"left": 0.0, "right": 0.0},
                            }
                            validate_synchronized_mount(mount)
                            raw.append(mount)
    raw.sort(key=lambda mount: (
        abs(np.mean([mount["xy"]["left"][0],
                     mount["xy"]["right"][0]]) + 0.30),
        abs(validate_synchronized_mount(mount) - 0.70),
        abs(mount["yaw"]["left"] - 15.0) +
        abs(mount["yaw"]["right"] - 15.0),
        mount["shared_base_z_m"],
        json.dumps(mount, sort_keys=True),
    ))
    if len(raw) <= maximum:
        return raw
    features = np.asarray([[
        mount["xy"]["left"][0] / 0.4,
        mount["xy"]["left"][1] / 0.5,
        mount["xy"]["right"][0] / 0.4,
        mount["xy"]["right"][1] / 0.5,
        np.sin(np.deg2rad(mount["yaw"]["left"])),
        np.cos(np.deg2rad(mount["yaw"]["left"])),
        np.sin(np.deg2rad(mount["yaw"]["right"])),
        np.cos(np.deg2rad(mount["yaw"]["right"])),
        (mount["shared_base_z_m"] - 0.81) / 0.05,
    ] for mount in raw])
    chosen = [0]
    minimum_distance = np.sum((features - features[0]) ** 2, axis=1)
    minimum_distance[0] = -np.inf
    while len(chosen) < maximum:
        index = int(np.argmax(minimum_distance))
        chosen.append(index)
        distance = np.sum((features - features[index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[chosen] = -np.inf
    return [raw[index] for index in chosen]


def _full_audited_safe(record):
    return (
        record.get("status") == "valid_selection"
        and record.get("audit_scope") == "full_timeline"
        and int(record.get("source_row_count", 0)) > 0
        and int(record.get("audited_source_rows", -1)) ==
        int(record.get("source_row_count", 0))
        and len(str(record.get("audit_fingerprint", ""))) == 64
        and int(record.get("pair_collision_frames", -1)) == 0
        and int(record.get("pair_edge_collision_frames", -1)) == 0
    )


def select_full_audited_mount(records):
    eligible = [record for record in records if _full_audited_safe(record)]
    if not eligible:
        raise RuntimeError("no full-audited collision-free synchronized mount")
    best = min(eligible, key=lambda record: (
        -float(record["continuous_pair_coverage"]),
        int(record.get("longest_hold_frames", 0)),
        int(record.get("relaxed_tier_frames", 0)),
        float(record.get("mean_pair_pose_error", np.inf)),
        -float(record.get("base_distance_m", 0.0)),
    ))
    mount = json.loads(json.dumps(best["mount"]))
    mount["selection_method"] = (
        "well-separated paired PiperX fixed-time search; selected only after "
        "complete 1964-row state and swept-edge collision audit")
    mount["full_audit_metrics"] = {
        key: value for key, value in best.items() if key != "mount"}
    return mount


def _write(records, selected_mount=None):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "robot": "piperx",
        "task": "fold_box",
        "timing_mode": "fixed_source_time",
        "minimum_base_separation_m": 0.60,
        "selected_mount": selected_mount,
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def main():
    task = _registered_task()
    records = []
    sparse_records = []
    candidates = synchronized_mount_candidates(maximum=32)
    for serial, mount in enumerate(candidates):
        record = evaluate_pair(
            task, mount, 20_000 + serial,
            uniform_count=20, global_seed_count=2,
            max_iterations=65, maximum_candidates=2,
        )
        record["stage"] = "sparse_screen"
        records.append(record)
        if (record["pair_collision_frames"] == 0 and
                record["pair_edge_collision_frames"] == 0):
            sparse_records.append(record)
        _write(records)
        print(
            "sparse", serial + 1, len(candidates),
            f"coverage={record['synchronous_pair_coverage']:.4f}",
            f"distance={record['base_distance_m']:.3f}",
            flush=True,
        )
    finalists = sorted(
        sparse_records, key=rank_paired_mount_candidate)[:3]
    full_records = []
    for serial, finalist in enumerate(finalists):
        record = evaluate_full_pair(
            task, finalist["mount"], 30_000 + serial)
        record["stage"] = "full_timeline_audit"
        records.append(record)
        full_records.append(record)
        _write(records)
        print(
            "full", serial + 1, len(finalists), record["status"],
            f"coverage={record['continuous_pair_coverage']:.4f}",
            f"distance={record['base_distance_m']:.3f}",
            flush=True,
        )
    selected = select_full_audited_mount(full_records)
    _write(records, selected)
    print(json.dumps(selected, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
