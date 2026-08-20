from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_xarm6_scene import DEFAULT_OUTPUT, build_scene
from doosan_teleop.sim_app import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            "qpservo_pvt",
            scene_builder=build_scene,
            scene_path=DEFAULT_OUTPUT,
            robot_label="xArm6",
            log_tag="xarm6",
        )
    )
