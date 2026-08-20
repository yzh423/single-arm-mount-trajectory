"""Measure joint-axis angles and line-to-line distances from the MuJoCo scene."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.four_robot_sim_app import (
    SCENE,
    _names,
    initialize_benchmark_home,
)


def unsigned_axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first, second)))
    return math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))


def line_distance(
    point_a: np.ndarray,
    axis_a: np.ndarray,
    point_b: np.ndarray,
    axis_b: np.ndarray,
) -> float:
    cross = np.cross(axis_a, axis_b)
    norm = float(np.linalg.norm(cross))
    delta = point_b - point_a
    if norm < 1e-9:
        return float(np.linalg.norm(np.cross(delta, axis_a)))
    return abs(float(np.dot(delta, cross))) / norm


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    initialize_benchmark_home(model, data)
    mujoco.mj_forward(model, data)
    report = {}
    for robot in ("xarm6", "ur5", "doosan"):
        joint_names = _names(robot)["left"]["joints"]
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ]
        points = [data.xanchor[joint_id].copy() for joint_id in joint_ids]
        axes = [data.xaxis[joint_id].copy() for joint_id in joint_ids]
        pair_metrics = {}
        for first, second in ((1, 2), (2, 3), (3, 4), (4, 5), (3, 5)):
            key = f"J{first+1}_J{second+1}"
            pair_metrics[key] = {
                "axis_angle_deg_unsigned": unsigned_axis_angle_deg(
                    axes[first], axes[second]
                ),
                "axis_line_distance_mm": 1000.0
                * line_distance(
                    points[first],
                    axes[first],
                    points[second],
                    axes[second],
                ),
            }
        wrist_point_spread = max(
            line_distance(points[i], axes[i], points[j], axes[j])
            for i in (3, 4, 5)
            for j in (3, 4, 5)
            if i < j
        )
        report[robot] = {
            "pairs": pair_metrics,
            "maximum_pairwise_J4_J5_J6_line_distance_mm": (
                1000.0 * wrist_point_spread
            ),
        }
    output = ROOT / "tuning/axis_topology_metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
