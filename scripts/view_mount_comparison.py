"""View desktop, 45-degree or 90-degree side-mounted benchmark scene."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_mount_variants import build
from doosan_teleop.four_robot_sim_app import initialize_benchmark_home


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount", choices=("desktop", "tilt45", "side90"), default="side90")
    args = parser.parse_args()
    scene = (
        ROOT / "models/four_robot_dual_arm_benchmark_scene.xml"
        if args.mount == "desktop"
        else build(args.mount)
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    initialize_benchmark_home(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.7]
        viewer.cam.distance = 5.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
