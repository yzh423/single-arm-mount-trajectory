"""Interactive tuned MPC for the optimized-spacing dual Doosan M0609 cell."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_doosan_realtime_scene import DEFAULT_OUTPUT, build_realtime_scene
from doosan_teleop.sim_app import main


def tuned_profile() -> dict[str, object]:
    profiles = json.loads(
        (ROOT / "configs" / "mpc_profiles.json").read_text(encoding="utf-8")
    )
    return dict(profiles["doosan"])


if __name__ == "__main__":
    print(
        "[doosan realtime] base spacing=0.84 m, control=50 Hz, "
        "tuned profile=doosan, collision=enabled"
    )
    raise SystemExit(
        main(
            "mpc_pvt",
            scene_builder=build_realtime_scene,
            scene_path=DEFAULT_OUTPUT,
            robot_label="M0609 optimized-spacing 50 Hz",
            log_tag="doosan_spacing084_50hz",
            config_overrides=tuned_profile(),
        )
    )
