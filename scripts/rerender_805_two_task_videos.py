"""Re-render the 13 x 2 validated caches with visual-only robot geometry."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_805_two_task_strict_validation import ROBOTS, TASKS


def main() -> None:
    failures = []
    for robot in ROBOTS:
        for task in TASKS:
            completed = subprocess.run(
                [sys.executable, "scripts/render_strict_single_arm_task.py", "--domain", "local",
                 "--robot", robot, "--task", task], cwd=ROOT,
            )
            print(robot, task, completed.returncode, flush=True)
            if completed.returncode:
                failures.append((robot, task, completed.returncode))
    if failures:
        raise SystemExit(f"render failures: {failures}")


if __name__ == "__main__":
    main()
