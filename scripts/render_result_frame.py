"""Render one timestamp from an offline NPZ for visual collision inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.four_robot_sim_app import SCENE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookat", nargs=3, type=float, default=(1.35, -1.55, 0.65))
    parser.add_argument("--distance", type=float, default=1.6)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    result = np.load(args.result)
    frame = int(np.argmin(np.abs(result["time"] - args.time)))
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    data.qpos[:] = result["qpos"][frame]
    data.mocap_pos[:] = result["mocap_pos"][frame]
    data.mocap_quat[:] = result["mocap_quat"][frame]
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = args.lookat
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
