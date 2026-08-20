"""Run strict MuJoCo solve and render for the 8-05 two-task validation matrix."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBOTS = (
    "doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5", "big_yam",
    "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "nero", "openarm",
    "piperx", "willow",
)
TASKS = ("cap-left", "fold-towel-dual")


def main() -> None:
    rows = []
    for robot in ROBOTS:
        for task in TASKS:
            solve = subprocess.run(
                [sys.executable, "scripts/solve_strict_urdf_task_cache.py", "--domain", "local",
                 "--robot", robot, "--task", task], cwd=ROOT, capture_output=True, text=True,
            )
            render = None
            cache = ROOT / "videos/single_arm/strict_cache/local" / robot / f"{task}.npz"
            if cache.is_file():
                render = subprocess.run(
                    [sys.executable, "scripts/render_strict_single_arm_task.py", "--domain", "local",
                     "--robot", robot, "--task", task], cwd=ROOT, capture_output=True, text=True,
                )
            row = {
                "robot": robot, "task": task, "solve_returncode": solve.returncode,
                "render_returncode": None if render is None else render.returncode,
                "solve_stderr": solve.stderr[-2000:],
                "render_stderr": "" if render is None else render.stderr[-2000:],
            }
            rows.append(row)
            print(robot, task, f"solve={solve.returncode}",
                  f"render={row['render_returncode']}", flush=True)
    output = ROOT / "reports/single_arm/validation_8-05_two_tasks/strict_matrix.json"
    output.write_text(json.dumps({"robots": list(ROBOTS), "tasks": list(TASKS), "jobs": rows}, indent=2),
                      encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
