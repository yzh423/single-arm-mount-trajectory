"""Headless target-step smoke test for the optimized 50 Hz Doosan MPC."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_doosan_realtime_scene import DEFAULT_OUTPUT, build_realtime_scene
from doosan_teleop.mpc_pvt import DualArmMPCPVT, MPCConfig


def main() -> int:
    scene = build_realtime_scene(DEFAULT_OUTPUT)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    profile = json.loads(
        (ROOT / "configs" / "mpc_profiles.json").read_text(encoding="utf-8")
    )["doosan"]
    controller = DualArmMPCPVT(
        model,
        data,
        MPCConfig(**profile),
        robot_kind="doosan",
    )
    controller.initialize_home()
    offsets = {
        "left": np.array([0.040, -0.030, 0.025]),
        "right": np.array([-0.035, -0.025, 0.020]),
    }
    for side, offset in offsets.items():
        data.mocap_pos[controller.arms[side].mocap_id] += offset

    control_ms = []
    for _ in range(150):
        started = time.perf_counter()
        controller.step()
        control_ms.append(1000.0 * (time.perf_counter() - started))

    errors_mm = {}
    for side, arm in controller.arms.items():
        errors_mm[side] = 1000.0 * float(
            np.linalg.norm(
                data.mocap_pos[arm.mocap_id] - data.site_xpos[arm.site_id]
            )
        )
    penetration_count = sum(contact.dist < 0.0 for contact in data.contact)
    result = {
        "final_position_error_mm": errors_mm,
        "control_mean_ms": float(np.mean(control_ms)),
        "control_p95_ms": float(np.percentile(control_ms, 95)),
        "control_max_ms": float(np.max(control_ms)),
        "penetration_count": int(penetration_count),
    }
    print(json.dumps(result, indent=2))
    if max(errors_mm.values()) > 5.0:
        raise RuntimeError(f"target-step error exceeds 5 mm: {errors_mm}")
    if result["control_p95_ms"] > 20.0:
        raise RuntimeError(f"50 Hz budget missed: {result['control_p95_ms']:.3f} ms")
    if penetration_count:
        raise RuntimeError(f"target-step test penetrated {penetration_count} contacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
