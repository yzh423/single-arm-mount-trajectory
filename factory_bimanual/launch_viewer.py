"""Launch an interactive MuJoCo viewer for a generated bimanual scene."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
