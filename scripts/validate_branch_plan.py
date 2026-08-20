"""Validate joint continuity and interpolated-edge collision of a branch plan."""

from __future__ import annotations

import argparse
import json
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
    robot_config,
)
from doosan_teleop.mpc_pvt import DualArmMPCPVT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--edge-samples", type=int, default=21)
    args = parser.parse_args()

    plan = np.load(args.plan.resolve())
    robot = str(plan["robot"])
    q_path = plan["q"]
    time_path = plan["time"]
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    initialize_benchmark_home(model, data)
    audit = DualArmMPCPVT(
        model,
        data,
        robot_config(robot, collision=True, tuned=True),
        name_map=_names(robot),
        robot_kind=robot,
    )

    delta = (np.diff(q_path, axis=0) + np.pi) % (2.0 * np.pi) - np.pi
    bad_edges = []
    minimum_distance = float("inf")
    maximum_penetrating_pairs = 0.0
    for edge in range(len(q_path) - 1):
        for alpha in np.linspace(0.0, 1.0, args.edge_samples):
            q = (1.0 - alpha) * q_path[edge] + alpha * q_path[edge + 1]
            data.qpos[audit.arms["left"].qpos_ids] = q[:6]
            data.qpos[audit.arms["right"].qpos_ids] = q[6:]
            mujoco.mj_forward(model, data)
            collision = audit.collision_diagnostics(data)
            minimum_distance = min(
                minimum_distance, float(collision["minimum_distance_m"])
            )
            maximum_penetrating_pairs = max(
                maximum_penetrating_pairs,
                float(collision["penetrating_pairs"]),
            )
            if float(collision["penetrating_pairs"]) > 0.0:
                bad_edges.append(
                    {
                        "edge": edge,
                        "time_s": float(
                            (1.0 - alpha) * time_path[edge]
                            + alpha * time_path[edge + 1]
                        ),
                        "minimum_distance_m": float(
                            collision["minimum_distance_m"]
                        ),
                    }
                )
                break
    report = {
        "plan": str(args.plan.resolve()),
        "robot": robot,
        "waypoints": int(len(q_path)),
        "edge_samples": int(args.edge_samples),
        "maximum_joint_jump_deg": float(
            np.degrees(np.max(np.abs(delta), initial=0.0))
        ),
        "rms_joint_jump_deg": float(
            np.degrees(np.sqrt(np.mean(delta * delta)))
        ),
        "minimum_interpolated_distance_m": (
            minimum_distance if np.isfinite(minimum_distance) else None
        ),
        "penetrating_edges": int(len(bad_edges)),
        "maximum_penetrating_pairs": float(maximum_penetrating_pairs),
        "bad_edges": bad_edges,
    }
    output = args.plan.with_name(args.plan.stem + "_validation.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if bad_edges else 0


if __name__ == "__main__":
    raise SystemExit(main())
