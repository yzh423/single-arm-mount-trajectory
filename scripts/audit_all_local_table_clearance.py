"""Audit every cleaned Local episode for tabletop penetration after scene placement."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from design_optimization.local_pose_sampling import trim_episode_edges
from scripts.strict_trajectory_sources import local_task_anchor_z

TABLE_HALF_X_M = 0.95
TABLE_HALF_Y_M = 0.72
TABLE_TOP_Z_M = 0.0
TCP_RENDER_RADIUS_M = 0.008
REQUIRED_TCP_CLEARANCE_M = 0.10


def audit_episode(root: Path, episode: dict[str, object]) -> dict[str, object]:
    path = root / str(episode["artifact"])
    values = np.load(path, allow_pickle=False)
    keep = trim_episode_edges(values["time_s"], seconds=0.15)
    positions = np.asarray(values["position_m"][keep], dtype=float)
    if len(positions) < 2:
        raise ValueError(f"{path}: fewer than two frames after trimming")
    anchor_z = local_task_anchor_z(str(episode["task"]), REQUIRED_TCP_CLEARANCE_M)
    anchor = np.asarray(((-0.18 if episode["hand"] == "left" else 0.18), -0.45, anchor_z))
    targets = anchor + positions - positions[0]
    inside_footprint = ((np.abs(targets[:, 0]) <= TABLE_HALF_X_M)
                        & (np.abs(targets[:, 1]) <= TABLE_HALF_Y_M))
    center_penetration = inside_footprint & (targets[:, 2] < TABLE_TOP_Z_M)
    sphere_contact = inside_footprint & (targets[:, 2] - TCP_RENDER_RADIUS_M < TABLE_TOP_Z_M)
    source_path = str(episode.get("source_path", ""))
    collection = source_path.split("/")[0] if "/" in source_path else Path(source_path).parts[0] if source_path else "unknown"
    return {
        "episode_id": episode["episode_id"], "collection": collection,
        "task": episode["task"], "hand": episode["hand"], "split": episode["split"],
        "trajectory_edge_trim_eligible": bool(episode.get("trajectory_edge_trim_eligible", True)),
        "artifact": str(episode["artifact"]), "frames_after_trim": int(len(targets)),
        "task_anchor_z_m": float(anchor_z),
        "source_min_z_m": float(positions[:, 2].min()),
        "target_min_z_m": float(targets[:, 2].min()),
        "target_max_z_m": float(targets[:, 2].max()),
        "center_penetration_frames": int(center_penetration.sum()),
        "tcp_sphere_contact_frames": int(sphere_contact.sum()),
        "outside_table_footprint_frames": int((~inside_footprint).sum()),
    }


def main() -> None:
    root = ROOT / "data/processed/local_pose_benchmark"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    errors = []
    for episode in manifest["episodes"]:
        try:
            rows.append(audit_episode(root, episode))
        except Exception as error:
            errors.append({"episode_id": episode.get("episode_id"), "error": f"{type(error).__name__}: {error}"})
    by_collection = defaultdict(lambda: {"episodes": 0, "frames": 0, "penetrating_episodes": 0,
                                         "penetrating_frames": 0, "sphere_contact_episodes": 0})
    by_task = defaultdict(lambda: {"episodes": 0, "penetrating_episodes": 0, "minimum_z_m": float("inf")})
    for row in rows:
        group = by_collection[row["collection"]]
        group["episodes"] += 1; group["frames"] += row["frames_after_trim"]
        group["penetrating_frames"] += row["center_penetration_frames"]
        group["penetrating_episodes"] += int(row["center_penetration_frames"] > 0)
        group["sphere_contact_episodes"] += int(row["tcp_sphere_contact_frames"] > 0)
        task = by_task[row["task"]]
        task["episodes"] += 1
        task["penetrating_episodes"] += int(row["center_penetration_frames"] > 0)
        task["minimum_z_m"] = min(task["minimum_z_m"], row["target_min_z_m"])
    penetrating = [row for row in rows if row["center_penetration_frames"]]
    contacts = [row for row in rows if row["tcp_sphere_contact_frames"]]
    payload = {
        "schema_version": 1,
        "policy": {
            "edge_trim_s": 0.15, "table_top_z_m": TABLE_TOP_Z_M,
            "table_half_extents_xy_m": [TABLE_HALF_X_M, TABLE_HALF_Y_M],
            "task_anchor_policy": "lowest robot-independent per-task anchor preserving TCP clearance",
            "required_tcp_clearance_m": REQUIRED_TCP_CLEARANCE_M,
            "tcp_render_radius_m": TCP_RENDER_RADIUS_M,
            "placement": "target = common task anchor + captured position - captured first position",
        },
        "summary": {
            "episodes": len(rows), "frames": sum(row["frames_after_trim"] for row in rows),
            "center_penetrating_episodes": len(penetrating),
            "center_penetrating_frames": sum(row["center_penetration_frames"] for row in rows),
            "tcp_sphere_contact_episodes": len(contacts), "errors": len(errors),
            "global_minimum_target_z_m": min(row["target_min_z_m"] for row in rows),
        },
        "by_collection": dict(sorted(by_collection.items())),
        "by_task": dict(sorted(by_task.items())),
        "penetrating_episodes": sorted(penetrating, key=lambda row: row["target_min_z_m"]),
        "tcp_sphere_contact_episodes": sorted(contacts, key=lambda row: row["target_min_z_m"]),
        "errors": errors,
    }
    output = ROOT / "reports/single_arm/all_local_table_clearance_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(output)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
