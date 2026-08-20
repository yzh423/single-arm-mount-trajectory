"""Scan dual-arm base spacing with a common world-frame task trajectory."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.four_robot_sim_app import (
    DatasetPlayback,
    SCENE,
    _align_targets,
    _names,
    robot_config,
)
from doosan_teleop.mpc_pvt import DualArmMPCPVT
from doosan_teleop.offline_consequence import _collision_free_start_pose


CENTERS = {"xarm6": 1.35, "ur5": -1.35, "doosan": 1.35}


def make_scene(
    source: Path,
    output: Path,
    robot: str,
    spacing: float,
    base_height: float | None = None,
) -> Path:
    tree = ET.parse(source)
    root = tree.getroot()
    center = CENTERS[robot]
    for side, sign in (("left", -1.0), ("right", 1.0)):
        body = root.find(f".//body[@name='{robot}_{side}_base']")
        if body is None:
            raise RuntimeError(f"Missing {robot}_{side}_base in {source}")
        position = [float(value) for value in body.get("pos", "").split()]
        position[0] = center + sign * spacing / 2.0
        if base_height is not None:
            position[2] = base_height
        body.set("pos", " ".join(f"{value:.9g}" for value in position))
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def evaluate(
    scene: Path,
    dataset: Path,
    robot: str,
    duration: float,
    control_hz: float,
    z_scale: float,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 1.0 / control_hz
    data = mujoco.MjData(model)
    controller = DualArmMPCPVT(
        model,
        data,
        robot_config(robot),
        name_map=_names(robot),
        robot_kind=robot,
    )
    controller.initialize_home()
    _align_targets(data, {robot: controller})
    playback = DatasetPlayback(
        dataset, data, {robot: controller}, 1.0, False, "canonical", z_scale
    )
    playback.apply_at(0.0)
    start_pose = _collision_free_start_pose(
        model, data, {robot: controller}
    )[robot]
    for _ in range(round(2.0 * control_hz)):
        controller.step()

    errors = {"left": [], "right": []}
    sigma = {"left": [], "right": []}
    collision_frames = 0
    steps = round(min(duration, float(playback.time[-1])) * control_hz) + 1
    for step in range(steps):
        playback.apply_at(step / control_hz)
        controller.step()
        collision_frames += controller._collision_penalty(data) > 0.0
        for side, arm in controller.arms.items():
            errors[side].append(float(arm.last_debug["pos_err"]))
            sigma[side].append(float(arm.last_debug["sigma_min"]))

    result: dict[str, float | dict] = {
        "start_pose": start_pose,
        "collision_frame_percent": 100.0 * collision_frames / steps,
        "arms": {},
    }
    for side in ("left", "right"):
        values = np.asarray(errors[side])
        result["arms"][side] = {
            "position_rmse_mm": float(1000.0 * np.sqrt(np.mean(values**2))),
            "position_p95_mm": float(1000.0 * np.percentile(values, 95)),
            "position_max_mm": float(1000.0 * np.max(values)),
            "minimum_sigma": float(min(sigma[side])),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=tuple(CENTERS), required=True)
    parser.add_argument("--spacing", nargs="+", type=float, required=True)
    parser.add_argument("--base-height", nargs="+", type=float)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "COFFAIL/benchmark/coffee_dual_active_82s.csv",
    )
    parser.add_argument("--duration", type=float, default=65.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    generated_scenes: list[Path] = []
    try:
        heights = args.base_height if args.base_height else [None]
        for spacing in args.spacing:
            for base_height in heights:
                print(
                    f"[installation] {args.robot} spacing={spacing:.3f} m "
                    f"base_z={base_height}",
                    flush=True,
                )
                handle = tempfile.NamedTemporaryFile(
                    prefix=f"scan_{args.robot}_",
                    suffix=".xml",
                    dir=args.scene.resolve().parent,
                    delete=False,
                )
                handle.close()
                generated_scene = Path(handle.name)
                generated_scenes.append(generated_scene)
                scene = make_scene(
                    args.scene.resolve(),
                    generated_scene,
                    args.robot,
                    spacing,
                    base_height,
                )
                metrics = evaluate(
                    scene,
                    args.dataset.resolve(),
                    args.robot,
                    args.duration,
                    args.control_hz,
                    args.z_scale,
                )
                result = {
                    "spacing_m": spacing,
                    "base_height_m": base_height,
                    **metrics,
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        for generated_scene in generated_scenes:
            generated_scene.unlink(missing_ok=True)

    document = {
        "robot": args.robot,
        "dataset": str(args.dataset.resolve()),
        "duration_s": args.duration,
        "control_hz": args.control_hz,
        "z_scale": args.z_scale,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
