"""Interactive tuned MPC for the optimized-spacing dual xArm6 cell."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_xarm_ur_realtime_scenes import XARM6_OUTPUT, build_xarm6_realtime_scene
from doosan_teleop.sim_app import main


def tuned_profile() -> dict[str, object]:
    profiles = json.loads(
        (ROOT / "configs" / "mpc_profiles.json").read_text(encoding="utf-8")
    )
    profile = dict(profiles["xarm6"])
    # xArm's detailed collision model makes the five-scale offline candidate
    # set miss a 20 ms budget. Keep the safe stop and full nominal candidates
    # for the low-latency interactive controller.
    profile["candidate_scales"] = (0.0, 1.0)
    return profile


if __name__ == "__main__":
    print(
        "[xarm6 realtime] base spacing=0.56 m, control=50 Hz, "
        "tuned profile=xarm6, collision=enabled"
    )
    raise SystemExit(
        main(
            "mpc_pvt",
            scene_builder=build_xarm6_realtime_scene,
            scene_path=XARM6_OUTPUT,
            robot_label="xArm6 optimized-spacing 50 Hz",
            log_tag="xarm6_spacing056_50hz",
            config_overrides=tuned_profile(),
        )
    )
