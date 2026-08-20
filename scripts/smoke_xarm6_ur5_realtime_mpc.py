"""Dynamic target-step and timing smoke test for xArm6/UR5 realtime MPC."""

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

from build_xarm_ur_realtime_scenes import (
    UR5_OUTPUT,
    XARM6_OUTPUT,
    build_ur5_realtime_scene,
    build_xarm6_realtime_scene,
)
from doosan_teleop.mpc_pvt import DualArmMPCPVT, MPCConfig


def run_robot(robot: str) -> dict[str, object]:
    profiles = json.loads(
        (ROOT / "configs" / "mpc_profiles.json").read_text(encoding="utf-8")
    )
    profile = dict(profiles[robot])
    if robot == "xarm6":
        profile["candidate_scales"] = (0.0, 1.0)
        scene = build_xarm6_realtime_scene(XARM6_OUTPUT)
    else:
        scene = build_ur5_realtime_scene(UR5_OUTPUT)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    controller = DualArmMPCPVT(
        model, data, MPCConfig(**profile), robot_kind=robot
    )
    controller.initialize_home()
    offsets = {
        "left": np.array([0.030, -0.020, 0.020]),
        "right": np.array([-0.030, -0.020, 0.020]),
    }
    for side, offset in offsets.items():
        data.mocap_pos[controller.arms[side].mocap_id] += offset

    control_ms = []
    for _ in range(150):
        started = time.perf_counter()
        controller.step()
        control_ms.append(1000.0 * (time.perf_counter() - started))
    errors = {
        side: 1000.0
        * float(
            np.linalg.norm(
                data.mocap_pos[arm.mocap_id] - data.site_xpos[arm.site_id]
            )
        )
        for side, arm in controller.arms.items()
    }
    unsafe_contacts = []
    for contact in data.contact:
        if contact.dist >= 0.0:
            continue
        body_pair = tuple(
            sorted(
                (
                    int(model.geom_bodyid[int(contact.geom1)]),
                    int(model.geom_bodyid[int(contact.geom2)]),
                )
            )
        )
        if body_pair in controller._allowed_body_pairs:
            continue
        unsafe_contacts.append(contact)
    result = {
        "final_position_error_mm": errors,
        "control_mean_ms": float(np.mean(control_ms)),
        "control_p95_ms": float(np.percentile(control_ms, 95)),
        "control_max_ms": float(np.max(control_ms)),
        "penetration_count": len(unsafe_contacts),
        "penetration_pairs": [
            (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                ),
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                ),
                1000.0 * float(contact.dist),
            )
            for contact in unsafe_contacts
        ],
    }
    print(robot, json.dumps(result), flush=True)
    if max(errors.values()) > 5.0:
        raise RuntimeError(f"{robot} target-step error exceeds 5 mm: {errors}")
    if result["control_p95_ms"] > 20.0:
        raise RuntimeError(
            f"{robot} misses 50 Hz budget: {result['control_p95_ms']:.3f} ms"
        )
    if result["penetration_count"]:
        raise RuntimeError(f"{robot} has penetrating contacts: {result}")
    return result


def main() -> int:
    result = {robot: run_robot(robot) for robot in ("xarm6", "ur5")}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
