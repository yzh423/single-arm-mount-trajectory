"""Refresh collision/status fields and summarize the 8-05 strict validation caches."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.solve_strict_urdf_task_cache import build_model, collision_flags
from scripts.strict_urdf_model_audit import MODELS


TASKS = {"cap-left", "fold-towel-dual"}


def main() -> None:
    rows = []
    cache_root = ROOT / "videos/single_arm/strict_cache/local"
    for audit_path in sorted(cache_root.glob("*/*.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("task") not in TASKS:
            continue
        robot = audit["robot"]
        cache = np.load(audit_path.with_suffix(".npz"))
        model = build_model(robot, cache["base_xyz_m"], float(cache["tilt_deg"]),
                            float(cache["yaw_deg"]), float(cache["roll_deg"]))
        table, self_collision = collision_flags(model, MODELS[robot].joints, cache["q"])
        audit["table_collision_frames"] = int(table.sum())
        audit["self_collision_frames"] = int(self_collision.sum())
        audit["status"] = "pass" if (bool(np.asarray(cache["success"]).all())
                                         and bool(audit["joint_limits_respected"])
                                         and not table.any() and not self_collision.any()) else "fail"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        rows.append(audit)
    output = ROOT / "reports/single_arm/validation_8-05_two_tasks/strict_audit_summary.json"
    output.write_text(json.dumps({"jobs": len(rows), "passes": sum(r["status"] == "pass" for r in rows),
                                  "results": rows}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
