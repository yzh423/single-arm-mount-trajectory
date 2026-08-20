"""Fresh ten-arm/two-task rerun with the current solver and search code."""
from __future__ import annotations

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
OUT = ROOT / "reports/single_arm/ten_arm_two_single_tasks_handbook_fixed_4096"


def main() -> None:
    pipeline.ROBOTS = ROBOTS
    pipeline.TASKS = TASKS
    pipeline.OUT = OUT
    pipeline.main()


if __name__ == "__main__":
    main()
