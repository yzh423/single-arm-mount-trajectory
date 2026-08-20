from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_scene import DEFAULT_OUTPUT as DOOSAN_SCENE, build_scene as build_doosan
from build_xarm6_scene import DEFAULT_OUTPUT as XARM6_SCENE, build_scene as build_xarm6
from doosan_teleop.mpc_pvt import DualArmMPCPVT, MPCConfig


DEFAULT_TRACE = (
    ROOT.parent
    / "Willow_VR_Teleop_Release"
    / "data"
    / "active_trace.json"
)
DEFAULT_OUTPUT = ROOT / "logs" / "willow_trace_benchmark"


def quat_mul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, first, second)
    return out


def quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
        dtype=np.float64,
    )


def quat_slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    delta = np.zeros(3)
    mujoco.mju_subQuat(delta, second, first)
    scaled = first.copy()
    mujoco.mju_quatIntegrate(scaled, delta, fraction)
    return scaled / np.linalg.norm(scaled)


def site_quaternion(data: mujoco.MjData, site_id: int) -> np.ndarray:
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    return quaternion


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload["samples"]
    times = np.asarray([float(sample["t"]) for sample in samples])
    times -= times[0]
    positions = np.asarray([sample["pos"] for sample in samples], dtype=np.float64)
    quaternions = np.asarray([sample["quat"] for sample in samples], dtype=np.float64)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return times, positions, quaternions, str(payload.get("arm", "right"))


def run_robot(
    label: str,
    scene_path: Path,
    scene_builder,
    trace_times: np.ndarray,
    trace_positions: np.ndarray,
    trace_quaternions: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    scene = scene_builder(scene_path) if not scene_path.exists() else scene_path
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    controller = DualArmMPCPVT(
        model,
        data,
        MPCConfig(collision_detection_enabled=True),
    )
    controller.initialize_home()
    arm = controller.arms["right"]

    robot_start_position = data.site_xpos[arm.site_id].copy()
    robot_start_quaternion = site_quaternion(data, arm.site_id)
    trace_start_position = trace_positions[0].copy()
    trace_start_quaternion_inv = quat_conjugate(trace_quaternions[0])

    dt = float(model.opt.timestep)
    evaluation_times = np.arange(0.0, trace_times[-1] + 0.5 * dt, dt)
    trace_index = 0
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    control_times: list[float] = []
    joint_margin_radians: list[float] = []
    collision_frames = 0
    rows: list[list[float]] = []

    started = time.perf_counter()
    for evaluation_time in evaluation_times:
        while (
            trace_index + 1 < len(trace_times) - 1
            and trace_times[trace_index + 1] < evaluation_time
        ):
            trace_index += 1
        next_index = min(trace_index + 1, len(trace_times) - 1)
        interval = max(trace_times[next_index] - trace_times[trace_index], 1e-12)
        fraction = float(
            np.clip(
                (evaluation_time - trace_times[trace_index]) / interval,
                0.0,
                1.0,
            )
        )
        trace_position = (
            (1.0 - fraction) * trace_positions[trace_index]
            + fraction * trace_positions[next_index]
        )
        trace_quaternion = quat_slerp(
            trace_quaternions[trace_index],
            trace_quaternions[next_index],
            fraction,
        )

        target_position = robot_start_position + (trace_position - trace_start_position)
        trace_delta_quaternion = quat_mul(
            trace_quaternion,
            trace_start_quaternion_inv,
        )
        target_quaternion = quat_mul(
            trace_delta_quaternion,
            robot_start_quaternion,
        )
        target_quaternion /= np.linalg.norm(target_quaternion)
        data.mocap_pos[arm.mocap_id] = target_position
        data.mocap_quat[arm.mocap_id] = target_quaternion

        control_started = time.perf_counter()
        controller.step(enabled_sides=("right",))
        control_times.append((time.perf_counter() - control_started) * 1000.0)

        actual_position = data.site_xpos[arm.site_id].copy()
        actual_quaternion = site_quaternion(data, arm.site_id)
        rotation_error = np.zeros(3)
        mujoco.mju_subQuat(rotation_error, target_quaternion, actual_quaternion)
        position_error = float(np.linalg.norm(target_position - actual_position))
        orientation_error = float(np.linalg.norm(rotation_error))
        position_errors.append(position_error)
        orientation_errors.append(orientation_error)

        q = data.qpos[arm.qpos_ids]
        margin = float(np.min(np.minimum(q - arm.q_min, arm.q_max - q)))
        joint_margin_radians.append(margin)
        collision = controller._collision_penalty(data) > 0.0
        collision_frames += int(collision)
        rows.append(
            [
                evaluation_time,
                *target_position,
                *actual_position,
                position_error,
                orientation_error,
                control_times[-1],
                margin,
                float(collision),
                *q,
            ]
        )

    elapsed = time.perf_counter() - started
    csv_path = output_dir / f"{label}_willow_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "target_x",
                "target_y",
                "target_z",
                "actual_x",
                "actual_y",
                "actual_z",
                "position_error_m",
                "orientation_error_rad",
                "control_ms",
                "minimum_joint_margin_rad",
                "collision",
                *[f"q{i}" for i in range(1, 7)],
            ]
        )
        writer.writerows(rows)

    return {
        "robot": label,
        "scene": str(scene),
        "trace_duration_s": float(trace_times[-1]),
        "evaluation_samples": len(evaluation_times),
        "wall_time_s": elapsed,
        "position_error_mm": {
            "mean": float(np.mean(position_errors) * 1000.0),
            "rms": float(np.sqrt(np.mean(np.square(position_errors))) * 1000.0),
            "p95": percentile(position_errors, 95.0) * 1000.0,
            "max": float(np.max(position_errors) * 1000.0),
        },
        "orientation_error_deg": {
            "mean": float(np.rad2deg(np.mean(orientation_errors))),
            "rms": float(np.rad2deg(np.sqrt(np.mean(np.square(orientation_errors))))),
            "p95": float(np.rad2deg(percentile(orientation_errors, 95.0))),
            "max": float(np.rad2deg(np.max(orientation_errors))),
        },
        "control_ms": {
            "mean": float(np.mean(control_times)),
            "p95": percentile(control_times, 95.0),
            "max": float(np.max(control_times)),
            "overrun_frames": int(np.count_nonzero(np.asarray(control_times) > dt * 1000.0)),
        },
        "minimum_joint_margin_deg": float(np.rad2deg(np.min(joint_margin_radians))),
        "joint_limit_frames": int(np.count_nonzero(np.asarray(joint_margin_radians) < 1e-3)),
        "collision_frames": collision_frames,
        "collision_fraction": collision_frames / len(evaluation_times),
        "csv": str(csv_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay one Willow TCP trace on Doosan M0609 and xArm6 MPC-PVT."
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    trace_path = args.trace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    times, positions, quaternions, arm = load_trace(trace_path)
    if arm != "right":
        raise ValueError(f"This benchmark currently expects a right-arm trace, got {arm!r}")

    results = []
    for label, scene, builder in (
        ("doosan_m0609", DOOSAN_SCENE, build_doosan),
        ("xarm6", XARM6_SCENE, build_xarm6),
    ):
        print(f"[benchmark] running {label} ...", flush=True)
        result = run_robot(
            label,
            scene,
            builder,
            times,
            positions,
            quaternions,
            output_dir,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    summary = {
        "format": "willow_trace_cross_robot_benchmark_v1",
        "trace": str(trace_path),
        "mapping": (
            "first trace pose aligned to each right-arm startup TCP; "
            "subsequent world-frame translation and orientation deltas preserved"
        ),
        "controller": "mpc_pvt",
        "collision_detection": True,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[benchmark] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
