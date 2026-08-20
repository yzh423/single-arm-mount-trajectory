"""Interactive tuned MPC for the optimized-spacing dual UR5 cell."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_xarm_ur_realtime_scenes import UR5_OUTPUT, build_ur5_realtime_scene
from doosan_teleop.sim_app import main


def tuned_profile() -> dict[str, object]:
    profiles = json.loads(
        (ROOT / "configs" / "mpc_profiles.json").read_text(encoding="utf-8")
    )
    return dict(profiles["ur5"])


if __name__ == "__main__":
    print(
        "[ur5 realtime] base spacing=0.50 m, control=50 Hz, "
        "tuned profile=ur5, collision=enabled"
    )
    raise SystemExit(
        main(
            "mpc_pvt",
            scene_builder=build_ur5_realtime_scene,
            scene_path=UR5_OUTPUT,
            robot_label="UR5 optimized-spacing 50 Hz",
            log_tag="ur5_spacing050_50hz",
            config_overrides=tuned_profile(),
        )
    )
