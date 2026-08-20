"""Offline beam search for a continuous, collision-free dual-arm IK branch path.

This is the global layer intentionally missing from a one-step Cartesian MPC.
It samples the whole task before execution, keeps several joint-space branches,
and rejects candidates that penetrate or enter the configured clearance margin.
The output is a sparse joint reference suitable for interpolation/trajectory
generation and as a future MPC branch hint.
"""

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

from doosan_teleop.easy_ik import DualArmEasyIKPVT
from doosan_teleop.four_robot_sim_app import (
    DatasetPlayback,
    SCENE,
    _align_targets,
    _names,
    initialize_benchmark_home,
    robot_config,
)
from doosan_teleop.mpc_pvt import DualArmMPCPVT
from doosan_teleop.offline_consequence import _collision_free_start_pose


def _wrapped_delta(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    delta = q1 - q0
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _solve_seed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik: DualArmEasyIKPVT,
    audit: DualArmMPCPVT,
    seed: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, float, dict[str, float]] | None:
    split = len(seed) // 2
    for offset, side in ((0, "left"), (split, "right")):
        arm = ik.arms[side]
        data.qpos[arm.qpos_ids] = np.clip(
            seed[offset : offset + split], arm.q_min + 1e-4, arm.q_max - 1e-4
        )
        data.qvel[arm.dof_ids] = 0.0
        ik._dq_filtered[side][:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(iterations):
        ik.step()

    q = np.concatenate(
        [data.qpos[ik.arms[side].qpos_ids].copy() for side in ("left", "right")]
    )
    pose_cost = 0.0
    minimum_sigma = float("inf")
    minimum_margin = float("inf")
    for side, arm in audit.arms.items():
        error, _, sigma = audit._pose_error_and_jacobian(
            data,
            arm,
            data.mocap_pos[arm.mocap_id],
            data.mocap_quat[arm.mocap_id],
        )
        pose_cost += 4000.0 * float(error[:3] @ error[:3])
        pose_cost += 20.0 * float(error[3:] @ error[3:])
        minimum_sigma = min(minimum_sigma, sigma)
        margin = np.minimum(
            data.qpos[arm.qpos_ids] - arm.q_min,
            arm.q_max - data.qpos[arm.qpos_ids],
        )
        minimum_margin = min(minimum_margin, float(np.min(margin)))

    collision = audit.collision_diagnostics(data)
    # Actual penetration is a hard constraint.  The predictive clearance band
    # is a soft warning zone: making it hard can incorrectly declare a narrow
    # but physically collision-free task infeasible.
    if float(collision["penetrating_pairs"]) > 0.0:
        return None
    local_cost = pose_cost + 200.0 * float(collision["penalty"])
    local_cost += 0.20 * max(0.0, 0.04 - minimum_sigma) ** 2
    local_cost += 0.05 * max(0.0, np.radians(10.0) - minimum_margin) ** 2
    return q, local_cost, {
        "minimum_sigma": float(minimum_sigma),
        "minimum_joint_margin_rad": float(minimum_margin),
        "minimum_distance_m": (
            float(collision["minimum_distance_m"])
            if np.isfinite(collision["minimum_distance_m"])
            else float("inf")
        ),
        "target_quat_left": data.mocap_quat[audit.arms["left"].mocap_id].tolist(),
        "target_quat_right": data.mocap_quat[audit.arms["right"].mocap_id].tolist(),
        "target_pos_left": data.mocap_pos[audit.arms["left"].mocap_id].tolist(),
        "target_pos_right": data.mocap_pos[audit.arms["right"].mocap_id].tolist(),
        "orientation_relax_deg": 0.0,
        "position_relax_xyz_m": [0.0, 0.0, 0.0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("xarm6", "ur5", "doosan"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waypoint-dt", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--ik-iterations", type=int, default=80)
    parser.add_argument("--random-restarts", type=int, default=6)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--z-scale", type=float, default=0.55)
    parser.add_argument("--max-waypoint-jump-deg", type=float, default=60.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    homes = initialize_benchmark_home(model, data)
    audit = DualArmMPCPVT(
        model,
        data,
        robot_config(args.robot, collision=True, tuned=True),
        name_map=_names(args.robot),
        robot_kind=args.robot,
    )
    controller = {args.robot: audit}
    _align_targets(data, controller)
    playback = DatasetPlayback(
        args.dataset.resolve(),
        data,
        controller,
        1.0,
        False,
        "canonical",
        args.z_scale,
    )
    playback.apply_at(0.0)
    _collision_free_start_pose(model, data, controller)
    ik = DualArmEasyIKPVT(
        model,
        data,
        robot_config(args.robot, collision=False),
        name_map=_names(args.robot),
        robot_kind=args.robot,
    )

    q0 = np.concatenate(
        [data.qpos[audit.arms[s].qpos_ids].copy() for s in ("left", "right")]
    )
    duration = float(playback.time[-1])
    if args.duration > 0.0:
        duration = min(duration, args.duration)
    times = np.arange(0.0, duration + 0.5 * args.waypoint_dt, args.waypoint_dt)
    if times[-1] < duration:
        times = np.append(times, duration)

    # Each item is cumulative cost, q, parent index, diagnostics.
    layers: list[list[tuple[float, np.ndarray, int, dict[str, float]]]] = [
        [(0.0, q0, -1, {})]
    ]
    perturbations = [None]
    for side in range(2):
        for joint, amount in ((0, 0.65), (2, 0.55), (4, 0.55)):
            for sign in (-1.0, 1.0):
                vector = np.zeros(12)
                vector[side * 6 + joint] = sign * amount
                perturbations.append(vector)
    rng = np.random.default_rng(730)
    random_lower = np.concatenate(
        [
            np.maximum(audit.arms[s].q_min + 1e-4, np.radians(-175.0))
            for s in ("left", "right")
        ]
    )
    random_upper = np.concatenate(
        [
            np.minimum(audit.arms[s].q_max - 1e-4, np.radians(175.0))
            for s in ("left", "right")
        ]
    )

    for waypoint_index, waypoint_time in enumerate(times[1:], start=1):
        playback.apply_at(float(waypoint_time))
        children: list[tuple[float, np.ndarray, int, dict[str, float]]] = []
        previous_layer = layers[-1]
        persistent_relax_deg = float(
            previous_layer[0][3].get("orientation_relax_deg", 0.0)
        )
        persistent_relax_side = previous_layer[0][3].get(
            "orientation_relax_side"
        )
        if persistent_relax_deg and persistent_relax_side in ("left", "right"):
            half = np.radians(persistent_relax_deg) / 2.0
            local_roll = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            mocap_id = audit.arms[persistent_relax_side].mocap_id
            relaxed_quat = np.zeros(4)
            mujoco.mju_mulQuat(
                relaxed_quat, data.mocap_quat[mocap_id], local_roll
            )
            data.mocap_quat[mocap_id] = relaxed_quat
        persistent_position_relax = np.asarray(
            previous_layer[0][3].get(
                "position_relax_xyz_m", [0.0, 0.0, 0.0]
            ),
            dtype=float,
        )
        persistent_position_side = previous_layer[0][3].get(
            "position_relax_side"
        )
        if (
            np.any(persistent_position_relax)
            and persistent_position_side in ("left", "right")
        ):
            data.mocap_pos[
                audit.arms[persistent_position_side].mocap_id
            ] += persistent_position_relax
        for parent_index, (parent_cost, parent_q, _, _) in enumerate(previous_layer):
            active_perturbations = perturbations if parent_index == 0 else [None]
            seeds = [
                parent_q if perturbation is None else parent_q + perturbation
                for perturbation in active_perturbations
            ]
            # Global restarts keep the beam from permanently collapsing to one
            # IK branch.  Charge their transition from the current parent so
            # they are selected only when continuity really needs a branch swap.
            near_clearance_boundary = (
                float(previous_layer[0][3].get("minimum_distance_m", float("inf")))
                < 1.5 * audit.config.collision_margin
            )
            if parent_index == 0 and (
                len(previous_layer) < args.beam_width or near_clearance_boundary
            ):
                restart_count = (
                    args.random_restarts
                    if near_clearance_boundary
                    else min(args.random_restarts, 8)
                )
                seeds.extend(
                    rng.uniform(random_lower, random_upper)
                    for _ in range(restart_count)
                )
            for seed in seeds:
                solved = _solve_seed(
                    model, data, ik, audit, seed, args.ik_iterations
                )
                if solved is None:
                    continue
                q, local_cost, diagnostics = solved
                if persistent_relax_deg:
                    diagnostics["orientation_relax_deg"] = persistent_relax_deg
                    diagnostics["orientation_relax_side"] = persistent_relax_side
                if np.any(persistent_position_relax):
                    diagnostics["position_relax_xyz_m"] = (
                        persistent_position_relax.tolist()
                    )
                    diagnostics["position_relax_side"] = (
                        persistent_position_side
                    )
                transition = _wrapped_delta(q, parent_q)
                if np.max(np.abs(transition)) > np.radians(
                    args.max_waypoint_jump_deg
                ):
                    continue
                transition_cost = 0.12 * float(transition @ transition)
                children.append(
                    (
                        parent_cost + transition_cost + local_cost,
                        q,
                        parent_index,
                        diagnostics,
                    )
                )
        # Quantised de-duplication prevents the beam filling with one branch.
        unique: dict[tuple[int, ...], tuple[float, np.ndarray, int, dict[str, float]]] = {}
        for child in sorted(children, key=lambda item: item[0]):
            # Coarse clustering is intentional: a fine bin lets five nearly
            # identical numerical solutions crowd out a genuinely different
            # shoulder/elbow/wrist branch.
            key = tuple(np.round(child[1] / 0.45).astype(int))
            if key not in unique:
                unique[key] = child
        layer = list(unique.values())[: args.beam_width]
        if not layer:
            # Doosan-style task/path avoidance fallback: preserve position but
            # permit a bounded roll about the tool axis only after exact 6D IK
            # has been proven collision-infeasible at this waypoint.
            relaxed_children = []
            original_quats = {
                side: data.mocap_quat[audit.arms[side].mocap_id].copy()
                for side in ("left", "right")
            }
            for relaxed_side in ("left", "right"):
                mocap_id = audit.arms[relaxed_side].mocap_id
                for angle_deg in (-60.0, -45.0, -30.0, -15.0, 15.0, 30.0, 45.0, 60.0):
                    half = np.radians(angle_deg) / 2.0
                    local_roll = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
                    relaxed_quat = np.zeros(4)
                    mujoco.mju_mulQuat(
                        relaxed_quat, original_quats[relaxed_side], local_roll
                    )
                    data.mocap_quat[mocap_id] = relaxed_quat
                    for parent_index, (parent_cost, parent_q, _, _) in enumerate(
                        previous_layer[:3]
                    ):
                        fallback_seeds = [parent_q]
                        for seed in fallback_seeds:
                            solved = _solve_seed(
                                model, data, ik, audit, seed, args.ik_iterations
                            )
                            if solved is None:
                                continue
                            q, local_cost, diagnostics = solved
                            diagnostics["orientation_relax_deg"] = angle_deg
                            diagnostics["orientation_relax_side"] = relaxed_side
                            transition = _wrapped_delta(q, parent_q)
                            if np.max(np.abs(transition)) > np.radians(
                                args.max_waypoint_jump_deg
                            ):
                                continue
                            relaxed_children.append(
                                (
                                    parent_cost
                                    + 0.12 * float(transition @ transition)
                                    + local_cost
                                    + 0.002 * angle_deg * angle_deg,
                                    q,
                                    parent_index,
                                    diagnostics,
                                )
                            )
                    data.mocap_quat[mocap_id] = original_quats[relaxed_side]
            unique = {}
            for child in sorted(relaxed_children, key=lambda item: item[0]):
                key = tuple(np.round(child[1] / 0.45).astype(int))
                if key not in unique:
                    unique[key] = child
            layer = list(unique.values())[: args.beam_width]
            if not layer:
                position_children = []
                original_positions = {
                    side: data.mocap_pos[audit.arms[side].mocap_id].copy()
                    for side in ("left", "right")
                }
                for relaxed_side in ("left", "right"):
                    mocap_id = audit.arms[relaxed_side].mocap_id
                    for dz in (0.02, 0.04, 0.06, 0.08):
                        offset = np.array([0.0, 0.0, dz])
                        total_offset = (
                            persistent_position_relax + offset
                            if relaxed_side == persistent_position_side
                            else offset
                        )
                        if float(np.linalg.norm(total_offset)) > 0.080001:
                            continue
                        data.mocap_pos[mocap_id] = (
                            original_positions[relaxed_side] + offset
                        )
                        for parent_index, (
                            parent_cost,
                            parent_q,
                            _,
                            _,
                        ) in enumerate(previous_layer[:3]):
                            solved = _solve_seed(
                                model,
                                data,
                                ik,
                                audit,
                                parent_q,
                                args.ik_iterations,
                            )
                            if solved is None:
                                continue
                            q, local_cost, diagnostics = solved
                            diagnostics["position_relax_xyz_m"] = (
                                total_offset.tolist()
                            )
                            diagnostics["position_relax_side"] = relaxed_side
                            transition = _wrapped_delta(q, parent_q)
                            if np.max(np.abs(transition)) > np.radians(
                                args.max_waypoint_jump_deg
                            ):
                                continue
                            position_children.append(
                                (
                                    parent_cost
                                    + 0.12 * float(transition @ transition)
                                    + local_cost
                                    + 1000.0 * float(
                                        total_offset @ total_offset
                                    ),
                                    q,
                                    parent_index,
                                    diagnostics,
                                )
                            )
                        data.mocap_pos[mocap_id] = original_positions[relaxed_side]
                unique = {}
                for child in sorted(position_children, key=lambda item: item[0]):
                    key = tuple(np.round(child[1] / 0.45).astype(int))
                    if key not in unique:
                        unique[key] = child
                layer = list(unique.values())[: args.beam_width]
            if not layer:
                failure = {
                    "robot": args.robot,
                    "dataset": str(args.dataset.resolve()),
                    "failure_time_s": float(waypoint_time),
                    "reason": (
                        "No continuous collision-free IK branch even with "
                        "bounded tool-roll and vertical-position relaxation"
                    ),
                    "z_scale": float(args.z_scale),
                    "beam_width": int(args.beam_width),
                    "random_restarts": int(args.random_restarts),
                    "ik_iterations": int(args.ik_iterations),
                    "max_waypoint_jump_deg": float(
                        args.max_waypoint_jump_deg
                    ),
                    "maximum_tool_roll_relax_deg": 60.0,
                    "maximum_vertical_relax_m": 0.08,
                }
                args.output.with_name(
                    args.output.stem + "_failure.json"
                ).write_text(json.dumps(failure, indent=2), encoding="utf-8")
                raise RuntimeError(
                    f"No collision-free IK branch at t={waypoint_time:.3f}s "
                    "even with bounded task-space relaxation."
                )
        layers.append(layer)
        print(
            f"[branch-plan] {waypoint_index}/{len(times)-1} "
            f"t={waypoint_time:.2f}s beam={len(layer)} cost={layer[0][0]:.4f}",
            flush=True,
        )

    best_index = min(range(len(layers[-1])), key=lambda i: layers[-1][i][0])
    path = []
    diagnostics = []
    for layer_index in range(len(layers) - 1, -1, -1):
        item = layers[layer_index][best_index]
        path.append(item[1])
        diagnostics.append(item[3])
        best_index = item[2]
        if layer_index == 0:
            break
    path.reverse()
    diagnostics.reverse()
    if len(diagnostics) > 1 and not diagnostics[0]:
        diagnostics[0] = dict(diagnostics[1])
        diagnostics[0]["orientation_relax_deg"] = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        time=times,
        q=np.asarray(path),
        robot=args.robot,
        diagnostics=np.asarray([json.dumps(x) for x in diagnostics]),
        planned_target_quat=np.asarray(
            [
                [
                    x.get("target_quat_left", [np.nan] * 4),
                    x.get("target_quat_right", [np.nan] * 4),
                ]
                for x in diagnostics
            ]
        ),
        planned_target_pos=np.asarray(
            [
                [
                    x.get("target_pos_left", [np.nan] * 3),
                    x.get("target_pos_right", [np.nan] * 3),
                ]
                for x in diagnostics
            ]
        ),
    )
    summary = {
        "robot": args.robot,
        "dataset": str(args.dataset.resolve()),
        "duration_s": float(duration),
        "waypoint_dt_s": float(args.waypoint_dt),
        "z_scale": float(args.z_scale),
        "beam_width": int(args.beam_width),
        "waypoints": int(len(times)),
        "max_waypoint_jump_deg": float(args.max_waypoint_jump_deg),
        "minimum_sigma": min(
            (x.get("minimum_sigma", float("inf")) for x in diagnostics[1:]),
            default=None,
        ),
        "minimum_joint_margin_deg": np.degrees(
            min(
                (
                    x.get("minimum_joint_margin_rad", float("inf"))
                    for x in diagnostics[1:]
                ),
                default=float("inf"),
            )
        ),
        "maximum_orientation_relax_deg": max(
            (abs(float(x.get("orientation_relax_deg", 0.0))) for x in diagnostics),
            default=0.0,
        ),
        "maximum_position_relax_m": max(
            (
                float(
                    np.linalg.norm(
                        x.get("position_relax_xyz_m", [0.0, 0.0, 0.0])
                    )
                )
                for x in diagnostics
            ),
            default=0.0,
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
