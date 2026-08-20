"""Exact native-length URDF mount rerank and 2x render for 13 arms x 2 tasks."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ("doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5", "big_yam",
          "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "nero", "openarm", "piperx", "willow")
TASKS = ("cap-left", "fold-towel-dual")


def main() -> None:
    rows = []
    for robot in ROBOTS:
        for task in TASKS:
            search = subprocess.run(
                [sys.executable, "scripts/search_strict_urdf_mount.py", "--domain", "local",
                 "--robot", robot, "--task", task, "--candidates", "32",
                 "--screen-frames", "32", "--pose-finalists", "6"],
                cwd=ROOT, capture_output=True, text=True,
            )
            render = subprocess.run(
                [sys.executable, "scripts/render_strict_single_arm_task.py", "--domain", "local",
                 "--robot", robot, "--task", task], cwd=ROOT, capture_output=True, text=True,
            )
            row = {"robot": robot, "task": task, "search_returncode": search.returncode,
                   "render_returncode": render.returncode, "search_stderr": search.stderr[-2000:],
                   "render_stderr": render.stderr[-2000:]}
            rows.append(row)
            print(robot, task, f"search={search.returncode}", f"render={render.returncode}", flush=True)
    output = ROOT / "reports/single_arm/validation_8-05_two_tasks/native_strict_rerank_matrix.json"
    output.write_text(json.dumps({"jobs": rows}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
