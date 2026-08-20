"""Deterministic per-robot MPC parameter search on a trajectory prefix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.four_robot_sim_app import DatasetPlayback, SCENE, _align_targets, _names
from doosan_teleop.mpc_pvt import DualArmMPCPVT, MPCConfig


DATASET = ROOT / "COFFAIL/benchmark/coffee_dual_active_set2_70s.csv"
OUTPUT = ROOT / "tuning/mpc_search_results.json"


CANDIDATES = (
    {},
    {"task_tau": 0.050, "damping": 0.018, "singularity_cost": 0.002},
    {"task_tau": 0.060, "damping": 0.025, "singularity_cost": 0.008},
    {"task_tau": 0.045, "damping": 0.015, "singularity_cost": 0.012,
     "orientation_weight": 1.0},
    {"task_tau": 0.070, "damping": 0.030, "singularity_cost": 0.020,
     "orientation_weight": 0.8},
    {"task_tau": 0.040, "damping": 0.012, "singularity_cost": 0.004,
     "candidate_scales": (0.0, 0.25, 0.5, 0.8, 1.0)},
    {"task_tau": 0.055, "damping": 0.020, "singularity_cost": 0.010,
     "collision_cost": 250.0, "collision_margin": 0.012},
    {"task_tau": 0.065, "damping": 0.022, "singularity_cost": 0.015,
     "collision_cost": 500.0, "collision_margin": 0.020,
     "joint_limit_cost": 0.01},
    {"task_tau": 0.080, "damping": 0.025, "singularity_cost": 0.008,
     "collision_cost": 120.0, "collision_margin": 0.010,
     "orientation_weight": 0.7},
    {"task_tau": 0.100, "damping": 0.030, "singularity_cost": 0.015,
     "collision_cost": 80.0, "collision_margin": 0.008,
     "orientation_weight": 0.5, "candidate_scales": (0.0, 0.2, 0.4, 0.7, 1.0)},
    {"task_tau": 0.060, "damping": 0.018, "singularity_cost": 0.003,
     "collision_cost": 100.0, "collision_margin": 0.010,
     "orientation_weight": 1.2, "candidate_scales": (0.0, 0.15, 0.4, 0.75, 1.1)},
)


def evaluate(robot: str, changes: dict[str, object], duration: float, rate: float) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = 1.0 / rate
    data = mujoco.MjData(model)
    config = MPCConfig(**changes)
    controller = DualArmMPCPVT(
        model, data, config, name_map=_names(robot), robot_kind=robot
    )
    controller.initialize_home()
    controllers = {robot: controller}
    _align_targets(data, controllers)
    playback = DatasetPlayback(DATASET, data, controllers, 1.0, False, "canonical")
    playback.apply_at(0.0)
    for _ in range(int(rate)):
        controller.step()
    position, orientation, sigma, margins, collision = [], [], [], [], []
    started = time.perf_counter()
    steps = int(duration * rate) + 1
    for step in range(steps):
        playback.apply_at(step / rate)
        controller.step()
        for arm in controller.arms.values():
            position.append(float(arm.last_debug.get("pos_err", 0.0)))
            orientation.append(float(arm.last_debug.get("rot_err", 0.0)))
            sigma.append(float(arm.last_debug.get("sigma_min", 0.0)))
            q = data.qpos[arm.qpos_ids]
            margins.append(float(np.min(np.minimum(q - arm.q_min, arm.q_max - q))))
        collision.append(float(controller._collision_penalty(data)))
    p = np.asarray(position)
    o = np.asarray(orientation)
    c = np.asarray(collision)
    objective = (
        np.sqrt(np.mean(p * p))
        + 0.5 * np.percentile(p, 95)
        + 0.08 * np.sqrt(np.mean(o * o))
        + 0.02 * np.percentile(o, 95)
        + 0.0001 * np.mean(c > 0.0)
        + 0.25 * max(0.0, 0.01 - min(sigma))
        + 0.1 * max(0.0, 0.05 - min(margins))
    )
    return {
        "robot": robot,
        "changes": changes,
        "objective": float(objective),
        "position_rmse_m": float(np.sqrt(np.mean(p * p))),
        "position_p95_m": float(np.percentile(p, 95)),
        "position_max_m": float(np.max(p)),
        "orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(o * o)))),
        "orientation_p95_deg": float(np.degrees(np.percentile(o, 95))),
        "minimum_sigma": float(min(sigma)),
        "minimum_joint_margin_deg": float(np.degrees(min(margins))),
        "collision_frame_percent": float(100.0 * np.mean(c > 0.0)),
        "wall_time_s": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--robots",
        nargs="+",
        choices=("xarm6", "ur5", "doosan"),
        default=("xarm6", "ur5", "doosan"),
    )
    args = parser.parse_args()
    results = []
    for robot in args.robots:
        for index, candidate in enumerate(CANDIDATES):
            print(f"[tune] {robot} candidate {index + 1}/{len(CANDIDATES)}", flush=True)
            result = evaluate(robot, candidate, args.duration, args.rate)
            result["candidate"] = index
            results.append(result)
            print(
                f"  objective={result['objective']:.5f} "
                f"rmse={result['position_rmse_m'] * 1000:.2f}mm "
                f"p95={result['position_p95_m'] * 1000:.2f}mm",
                flush=True,
            )
    best = {
        robot: min(
            (result for result in results if result["robot"] == robot),
            key=lambda result: result["objective"],
        )
        for robot in args.robots
    }
    payload = {
        "dataset": str(DATASET),
        "duration_s": args.duration,
        "rate_hz": args.rate,
        "base_config": asdict(MPCConfig()),
        "best": best,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
