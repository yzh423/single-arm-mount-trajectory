"""Run the formal 12-arm matrix over every purely single-hand Local task."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_twelve_arm_two_single_tasks as pipeline


TASKS = (
    "cap-left", "in-case-left", "open-box", "open-box-2", "open-box-3",
    "pick-from-high-left", "pick-right-left", "stick-battery", "toss-high",
    "tube", "tube-left-random", "tube-left-upright",
)
OUT = ROOT / "reports/single_arm/twelve_arm_all_single_tasks"


def main() -> None:
    pipeline.TASKS = TASKS
    pipeline.OUT = OUT
    pipeline.main()
    subprocess.run([sys.executable, "scripts/build_twelve_arm_all_single_task_outputs.py"],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
