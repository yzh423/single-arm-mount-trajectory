"""Scan Doosan dual-base spacing under canonical collision-aware MPC."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.four_robot_sim_app import DatasetPlayback, _align_targets, _names, robot_config
from doosan_teleop.mpc_pvt import DualArmMPCPVT


SOURCE = ROOT / "models/four_robot_dual_arm_benchmark_scene.xml"
DATASET = ROOT / "COFFAIL/benchmark/coffee_dual_active_set2_70s.csv"
OUTPUT = ROOT / "tuning/doosan_spacing_scan.json"


def scene_for(spacing: float) -> Path:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    center = 1.35
    for side, sign in (("left", -1.0), ("right", 1.0)):
        body = root.find(f".//body[@name='doosan_{side}_base']")
        pos = [float(v) for v in body.get("pos").split()]
        pos[0] = center + sign * spacing / 2
        body.set("pos", " ".join(str(v) for v in pos))
    path = ROOT / f"models/doosan_spacing_{spacing:.2f}_scene.xml"
    ET.indent(root, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def evaluate(spacing: float, duration: float = 20.0, rate: float = 50.0) -> dict:
    scene = scene_for(spacing)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 1.0 / rate
    data = mujoco.MjData(model)
    controller = DualArmMPCPVT(
        model,
        data,
        robot_config("doosan"),
        name_map=_names("doosan"),
        robot_kind="doosan",
    )
    controller.initialize_home()
    _align_targets(data, {"doosan": controller})
    playback = DatasetPlayback(DATASET, data, {"doosan": controller}, 1.0, False, "canonical")
    playback.apply_at(0.0)
    for _ in range(int(2.0 * rate)):
        controller.step()
    position = []
    collision = []
    sigma = []
    for step in range(int(duration * rate) + 1):
        playback.apply_at(step / rate)
        controller.step()
        collision.append(controller._collision_penalty(data))
        for arm in controller.arms.values():
            position.append(float(arm.last_debug["pos_err"]))
            sigma.append(float(arm.last_debug["sigma_min"]))
    p = np.asarray(position)
    return {
        "spacing_m": spacing,
        "position_rmse_m": float(np.sqrt(np.mean(p * p))),
        "position_p95_m": float(np.percentile(p, 95)),
        "position_max_m": float(np.max(p)),
        "minimum_sigma": float(min(sigma)),
        "collision_frame_percent": float(100 * np.mean(np.asarray(collision) > 0)),
    }


results = []
for spacing in (0.72, 0.78, 0.84, 0.90, 0.96, 1.02):
    print(f"[spacing] {spacing:.2f} m", flush=True)
    result = evaluate(spacing)
    results.append(result)
    print(result, flush=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
