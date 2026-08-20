"""Export a small deterministic task-stratified JSON sample without importing Torch."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from design_optimization.local_pose_sampling import relative_pose_sample
base = ROOT / "data/processed/local_pose_benchmark"
manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
rows = []
for episode in sorted(manifest["episodes"], key=lambda row: row["episode_id"]):
    if episode["split"] != "validation" or not episode.get("trajectory_edge_trim_eligible", True):
        continue
    data = np.load(base / episode["artifact"])
    try:
        position, relative_quaternion = relative_pose_sample(data, 10)
    except ValueError:
        continue
    rows.append({"task": episode["task"], "hand": episode["hand"], "episode_id": episode["episode_id"],
                 "edge_trim_seconds": 0.15,
                 "relative_position_m": position.tolist(),
                 "relative_quaternion_wxyz": relative_quaternion.tolist()})
output = ROOT / "data/processed/local_pose_benchmark/pilot_samples.json"
output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"tasks": len(rows), "frames": sum(len(x["relative_position_m"]) for x in rows)}))
