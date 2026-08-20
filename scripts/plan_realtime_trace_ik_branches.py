"""Plan continuous M0609 shoulder/elbow/wrist branches for a recorded trace."""

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
from doosan_teleop.mpc_pvt import DualArmMPCPVT, MPCConfig
from scripts.plan_collision_free_ik_branches import _solve_seed, _wrapped_delta


def _nlerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    result = (1.0 - amount) * q0 + amount * q1
    return result / max(float(np.linalg.norm(result)), 1e-12)


def _filtered_targets(
    samples: list[dict], tau_position: float, tau_rotation: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray([float(sample["t"]) for sample in samples])
    positions = np.asarray([sample["mocap_pos"] for sample in samples], dtype=float)
    quaternions = np.asarray([sample["mocap_quat"] for sample in samples], dtype=float)
    filtered_position = positions.copy()
    filtered_quaternion = quaternions.copy()
    for index in range(1, len(time)):
        dt = max(time[index] - time[index - 1], 1e-4)
        ap = 1.0 - np.exp(-dt / max(tau_position, 1e-6))
        ar = 1.0 - np.exp(-dt / max(tau_rotation, 1e-6))
        filtered_position[index] = (
            filtered_position[index - 1]
            + ap * (positions[index] - filtered_position[index - 1])
        )
        for side in range(quaternions.shape[1]):
            filtered_quaternion[index, side] = _nlerp(
                filtered_quaternion[index - 1, side],
                quaternions[index, side],
                ar,
            )
    return time, filtered_position, filtered_quaternion


def _analytic_wrist_flip(seed: np.ndarray, side: int) -> np.ndarray:
    result = seed.copy()
    offset = side * 6
    result[offset + 3] += np.pi
    result[offset + 4] *= -1.0
    result[offset + 5] += np.pi
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waypoint-dt", type=float, default=0.5)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--ik-iterations", type=int, default=55)
    parser.add_argument("--position-tau", type=float, default=0.05)
    parser.add_argument("--rotation-tau", type=float, default=0.06)
    parser.add_argument("--max-jump-deg", type=float, default=55.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--reject-penetrating-ik",
        action="store_true",
        help=(
            "Reject geometric IK seeds that penetrate. By default collision is "
            "left to the downstream MPC so an infeasible exact target does not "
            "collapse every branch."
        ),
    )
    parser.add_argument(
        "--right-only",
        action="store_true",
        help="Keep the left arm at home and plan only the recorded right target.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=5.0,
        help="Ignore startup alignment transient (trace time, seconds).",
    )
    args = parser.parse_args()

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    samples = payload["samples"]
    scene = Path(payload.get("metadata", {}).get(
        "scene", ROOT / "models/dual_m0609_2f85_spacing084_50hz_scene.xml"
    ))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    audit = DualArmMPCPVT(
        model,
        data,
        MPCConfig(collision_detection_enabled=args.reject_penetrating_ik),
        robot_kind="doosan",
    )
    ik = DualArmEasyIKPVT(
        model,
        data,
        MPCConfig(collision_detection_enabled=False),
        robot_kind="doosan",
    )
    time, position, quaternion = _filtered_targets(
        samples, args.position_tau, args.rotation_tau
    )
    if args.right_only:
        audit.initialize_home()
        mujoco.mj_forward(model, data)
        left_arm = audit.arms["left"]
        left_position = data.site_xpos[left_arm.site_id].copy()
        left_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(
            left_quaternion, data.site_xmat[left_arm.site_id]
        )
        position[:, 0, :] = left_position
        quaternion[:, 0, :] = left_quaternion
    start_time = max(float(time[0]), args.start)
    duration = time[-1] if args.duration <= 0.0 else min(
        time[-1], start_time + args.duration
    )
    waypoint_time = np.arange(
        start_time, duration + 0.5 * args.waypoint_dt, args.waypoint_dt
    )
    waypoint_indices = np.searchsorted(time, waypoint_time, side="left")
    waypoint_indices = np.minimum(waypoint_indices, len(time) - 1)

    start_sample = int(waypoint_indices[0])
    if args.right_only:
        data.mocap_pos[:] = position[start_sample]
        data.mocap_quat[:] = quaternion[start_sample]
        for _ in range(180):
            ik.step(enabled_sides=("right",))
        q0 = np.concatenate([
            data.qpos[audit.arms[side].qpos_ids].copy()
            for side in ("left", "right")
        ])
    else:
        q0 = np.concatenate([
            np.asarray(samples[start_sample]["arms"][side]["q"], dtype=float)
            for side in ("left", "right")
        ])
    layers: list[list[tuple[float, np.ndarray, int, dict]]] = [[(0.0, q0, -1, {})]]
    maximum_jump = np.radians(args.max_jump_deg)
    perturbations = []
    for side in range(2):
        for joint, amount in ((0, 0.8), (1, 0.65), (2, 0.8), (4, 0.55)):
            for sign in (-1.0, 1.0):
                delta = np.zeros(12)
                delta[side * 6 + joint] = sign * amount
                perturbations.append(delta)

    for layer_index, sample_index in enumerate(waypoint_indices[1:], start=1):
        data.mocap_pos[:] = position[sample_index]
        data.mocap_quat[:] = quaternion[sample_index]
        children = []
        for parent_index, (parent_cost, parent_q, _, _) in enumerate(layers[-1]):
            seeds = [parent_q]
            # Keep all three meaningful families alive: local continuation,
            # exact spherical-wrist alternate decomposition, and shoulder/elbow
            # numerical basins.  The latter are necessary when both wrist
            # decompositions cross J5=0 for the same arm posture.
            flip_sides = (1,) if args.right_only else range(2)
            seeds.extend(
                _analytic_wrist_flip(parent_q, side)
                for side in flip_sides
            )
            previous_sigma = float(
                layers[-1][0][3].get("minimum_sigma", 0.0)
            )
            if parent_index == 0 and (
                previous_sigma < 0.065 or layer_index % 12 == 0
            ):
                seeds.extend(parent_q + delta for delta in perturbations)
            for seed in seeds:
                solved = _solve_seed(model, data, ik, audit, seed, args.ik_iterations)
                if solved is None:
                    continue
                q, local_cost, diagnostics = solved
                transition = _wrapped_delta(q, parent_q)
                if float(np.max(np.abs(transition))) > maximum_jump:
                    continue
                sigma = float(diagnostics["minimum_sigma"])
                # Strongly prefer a branch that remains outside the wrist
                # warning zone; local pose accuracy still dominates once safe.
                singular_cost = 20.0 * max(0.0, 0.04 - sigma) ** 2
                transition_cost = 0.20 * float(transition @ transition)
                children.append((
                    parent_cost + local_cost + transition_cost + singular_cost,
                    q,
                    parent_index,
                    diagnostics,
                ))
        unique = {}
        for child in sorted(children, key=lambda item: item[0]):
            key = tuple(np.round(child[1] / 0.35).astype(int))
            if key not in unique:
                unique[key] = child
        layer = list(unique.values())[: args.beam_width]
        if not layer:
            raise RuntimeError(
                f"no continuous collision-free branch at t={time[sample_index]:.3f}s"
            )
        layers.append(layer)
        if layer_index % 10 == 0 or layer_index == len(waypoint_indices) - 1:
            print(
                f"[branch] {layer_index}/{len(waypoint_indices)-1} "
                f"t={time[sample_index]:.2f}s branches={len(layer)} "
                f"sigma={layer[0][3]['minimum_sigma']:.5f}"
            )

    selected = []
    index = 0
    for layer_index in range(len(layers) - 1, -1, -1):
        selected.append(layers[layer_index][index])
        index = selected[-1][2]
    selected.reverse()
    q_plan = np.asarray([item[1] for item in selected])
    sigma_plan = np.asarray([
        item[3].get("minimum_sigma", np.nan) for item in selected
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        robot="doosan",
        time=time[waypoint_indices],
        q=q_plan,
        minimum_sigma=sigma_plan,
        target_position=position[waypoint_indices],
        target_quaternion=quaternion[waypoint_indices],
    )
    report = {
        "trace": str(args.trace.resolve()),
        "scene": str(scene.resolve()),
        "waypoints": int(len(q_plan)),
        "waypoint_dt": float(args.waypoint_dt),
        "beam_width": int(args.beam_width),
        "minimum_planned_sigma": float(np.nanmin(sigma_plan[1:])),
        "maximum_wrapped_joint_jump_deg": float(np.degrees(np.max(np.abs(
            [_wrapped_delta(q_plan[i], q_plan[i - 1]) for i in range(1, len(q_plan))]
        )))),
        "target_filter_tau_s": [args.position_tau, args.rotation_tau],
        "ik_seed_penetration_rejection": bool(args.reject_penetrating_ik),
        "execution_collision_handling": "downstream MPC",
        "right_only": bool(args.right_only),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
