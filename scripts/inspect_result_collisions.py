"""List the worst non-excluded contact pairs in an offline result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.four_robot_sim_app import SCENE, _names, robot_config
from doosan_teleop.mpc_pvt import DualArmMPCPVT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    result = np.load(args.result)
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    controller = DualArmMPCPVT(
        model,
        data,
        robot_config(args.robot),
        name_map=_names(args.robot),
        robot_kind=args.robot,
    )
    records = []
    for frame, timestamp in enumerate(result["time"]):
        data.qpos[:] = result["qpos"][frame]
        data.mocap_pos[:] = result["mocap_pos"][frame]
        data.mocap_quat[:] = result["mocap_quat"][frame]
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if not (controller._robot_geom[geom1] or controller._robot_geom[geom2]):
                continue
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            pair = tuple(sorted((body1, body2)))
            if body1 == body2 or pair in controller._allowed_body_pairs:
                continue
            records.append(
                (
                    float(contact.dist),
                    float(timestamp),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                )
            )
    print("distance_m,time_s,body1,body2,geom1,geom2")
    for record in sorted(records)[: args.limit]:
        print(
            f"{record[0]:.6f},{record[1]:.3f},"
            + ",".join(str(value) for value in record[2:])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
