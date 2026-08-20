"""Print kinematic diagnostics around failures in an offline result."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.four_robot_sim_app import SCENE, _names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--column", type=int, required=True)
    parser.add_argument("--every", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=float("inf"))
    args = parser.parse_args()

    result = np.load(args.result)
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    names = _names(args.robot)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        for joint_name in names[args.side]["joints"]
    ]
    qpos_ids = np.asarray([model.jnt_qposadr[joint] for joint in joint_ids])
    dof_ids = np.asarray([model.jnt_dofadr[joint] for joint in joint_ids])
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, names[args.side]["site"]
    )
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    next_time = args.start
    print("time_s,error_mm,sigma_min,condition,joint_deg")
    for frame, timestamp in enumerate(result["time"]):
        if timestamp < next_time or timestamp > args.end:
            continue
        data.qpos[:] = result["qpos"][frame]
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        jacobian = np.vstack((jac_pos[:, dof_ids], jac_rot[:, dof_ids]))
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        joints = np.degrees(data.qpos[qpos_ids])
        error_mm = 1000.0 * result["position_error"][frame, args.column]
        print(
            f"{timestamp:.3f},{error_mm:.3f},{singular_values[-1]:.6f},"
            f"{singular_values[0] / max(singular_values[-1], 1e-12):.1f},"
            + " ".join(f"{joint:.2f}" for joint in joints)
        )
        next_time += args.every
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
