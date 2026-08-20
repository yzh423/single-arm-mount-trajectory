"""Formal ten-arm study on cap-left and open-box-2.

Willow, Big YAM, and Nero are intentionally excluded.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_twelve_arm_two_single_tasks as pipeline


ROBOTS = (
    "doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5",
    "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "openarm", "piperx",
)
TASKS = ("cap-left", "open-box-2")
OUT = pipeline.ROOT / "reports/single_arm/ten_arm_two_single_tasks"


def main() -> None:
    pipeline.ROBOTS = ROBOTS
    pipeline.TASKS = TASKS
    pipeline.OUT = OUT
    pipeline.main()
    subprocess.run(
        [sys.executable, "scripts/build_ten_arm_two_single_task_outputs.py"],
        cwd=pipeline.ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
