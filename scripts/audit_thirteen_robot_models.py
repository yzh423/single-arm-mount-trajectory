"""Audit official native-length chains and their GPU/MuJoCo FK parity."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import mujoco


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from design_optimization.robot_registry import load_robot_registry
from design_optimization.urdf_chain import (
    NativeSerialChain,
    fk_flange,
    load_native_chain,
    sampled_maximum_reach,
)
from design_optimization.robot_contract import PANDA_LOCKED_J3_RANGE_RAD
from scripts.strict_urdf_model_audit import MODELS, ModelEntry, load_native_spec


def canonical_chain_from_entry(
    robot_id: str, entry: ModelEntry
) -> tuple[NativeSerialChain, mujoco.MjModel, int]:
    """Extract the search POE chain from the exact model used by MuJoCo."""
    # The Panda control and locked-J3 ablation share identical native geometry;
    # only the joint range differs.
    unlocked_entry = replace(entry, locked_joint_ranges={}) if entry.locked_joint_ranges else entry
    spec = load_native_spec(unlocked_entry)
    flange_parent = spec.body(entry.flange_parent or entry.tcp_parent)
    tcp_parent = spec.body(entry.tcp_parent)
    if flange_parent is None or tcp_parent is None:
        raise ValueError(f"{robot_id}: missing flange/TCP parent")
    flange_parent.add_site(name="canonical_flange", pos=[0.0, 0.0, 0.0], size=[0.001, 0.0, 0.0])
    for joint_name, joint_range in (entry.locked_joint_ranges or {}).items():
        joint = spec.joint(joint_name)
        joint.range[:] = joint_range
        joint.limited = True
    tcp_parent.add_site(name="canonical_tcp", pos=entry.tool_translation_m,
                        quat=entry.tool_quaternion_wxyz, size=[0.001, 0.0, 0.0])
    model = spec.compile()
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in entry.joints]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError(f"{robot_id}: canonical model is missing an active joint")
    addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    data.qpos[addresses] = 0.0
    mujoco.mj_forward(model, data)
    root = int(model.jnt_bodyid[joint_ids[0]])
    while int(model.body_parentid[root]) != 0:
        root = int(model.body_parentid[root])
    base_rotation = data.xmat[root].reshape(3, 3).copy()
    base_position = data.xpos[root].copy()
    points = np.asarray([base_rotation.T @ (data.xanchor[j] - base_position) for j in joint_ids])
    axes = np.asarray([base_rotation.T @ data.xaxis[j] for j in joint_ids])
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "canonical_flange")
    home = np.eye(4)
    home[:3, :3] = base_rotation.T @ data.site_xmat[site_id].reshape(3, 3)
    home[:3, 3] = base_rotation.T @ (data.site_xpos[site_id] - base_position)
    q_min, q_max = [], []
    for index, joint_id in enumerate(joint_ids):
        bounds = model.jnt_range[joint_id] if model.jnt_limited[joint_id] else (-np.pi, np.pi)
        if robot_id == "franka_panda_locked_j3" and index == 2:
            bounds = PANDA_LOCKED_J3_RANGE_RAD
        q_min.append(float(bounds[0]))
        q_max.append(float(bounds[1]))
    deltas = np.vstack((np.diff(points, axis=0), home[:3, 3] - points[-1]))
    chain = NativeSerialChain(robot_id, entry.joints, axes, points, home,
                              np.asarray(q_min), np.asarray(q_max), deltas)
    return chain, model, site_id


def _fk_equivalence(chain: NativeSerialChain, model: mujoco.MjModel, site_id: int,
                    *, sample_count: int, seed: int = 20260808) -> tuple[float, float]:
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in chain.joint_names]
    addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    root = int(model.jnt_bodyid[joint_ids[0]])
    while int(model.body_parentid[root]) != 0:
        root = int(model.body_parentid[root])
    data = mujoco.MjData(model)
    data.qpos[addresses] = 0.0
    mujoco.mj_forward(model, data)
    base_rotation = data.xmat[root].reshape(3, 3).copy()
    base_position = data.xpos[root].copy()
    rng = np.random.default_rng(seed)
    maximum_position = 0.0
    maximum_orientation = 0.0
    for q in rng.uniform(chain.q_min, chain.q_max, size=(sample_count, chain.dof)):
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        expected_rotation = base_rotation.T @ data.site_xmat[site_id].reshape(3, 3)
        expected_position = base_rotation.T @ (data.site_xpos[site_id] - base_position)
        actual = fk_flange(chain, q)
        maximum_position = max(maximum_position, float(np.linalg.norm(actual[:3, 3] - expected_position)))
        relative = actual[:3, :3].T @ expected_rotation
        angle = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        maximum_orientation = max(maximum_orientation, angle)
    return maximum_position, maximum_orientation


def audit_registry(registry_path: str | Path, *, sample_count: int = 1000) -> dict:
    # registry_path remains accepted for CLI compatibility; canonical MODELS is
    # the sole geometry authority for both search and MuJoCo rendering.
    _ = registry_path
    robots: dict[str, dict] = {}
    failures: list[str] = []
    for robot_id, entry in MODELS.items():
        try:
            native, model, site_id = canonical_chain_from_entry(robot_id, entry)
            native_reach = sampled_maximum_reach(native, sample_count=2048, seed=20260814)
            position_error, orientation_error = _fk_equivalence(
                native, model, site_id, sample_count=sample_count
            )
            if position_error > 1e-4 or np.degrees(orientation_error) > 0.01:
                failures.append(f"{robot_id}: FK mismatch {position_error} m, {np.degrees(orientation_error)} deg")
            data = mujoco.MjData(model)
            joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                         for name in entry.joints]
            data.qpos[[int(model.jnt_qposadr[j]) for j in joint_ids]] = 0.0
            mujoco.mj_forward(model, data)
            tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "canonical_tcp")
            flange_rotation = data.site_xmat[site_id].reshape(3, 3)
            flange_tcp_translation = flange_rotation.T @ (
                data.site_xpos[tcp_site] - data.site_xpos[site_id])
            robots[robot_id] = {
                "model_format": entry.path.suffix.lower().lstrip("."),
                "source_model": str(entry.path),
                "native_joint_count": native.dof,
                "active_dof": native.dof,
                "joint_names": list(native.joint_names),
                "q_min_rad": native.q_min.tolist(),
                "q_max_rad": native.q_max.tolist(),
                "axes": native.axes.tolist(),
                "flange_home_rotation": native.home[:3, :3].tolist(),
                "first_joint_origin_m": native.points[0].tolist(),
                "source_deltas_m": native.deltas.tolist(),
                "native_deltas_m": native.deltas.tolist(),
                "native_sampled_max_reach_m": native_reach,
                "geometry_policy": "official_vendor_native_dimensions",
                "flange_tcp_translation_m": np.round(flange_tcp_translation, 15).tolist(),
                "tcp_authority": entry.tcp_authority,
                "joint_limit_authority": entry.joint_limit_authority,
                "fk_gate_samples": sample_count,
                "fk_position_error_max_m": position_error,
                "fk_orientation_error_max_deg": float(np.degrees(orientation_error)),
                "locked_joints_rad": ({"joint3": list(PANDA_LOCKED_J3_RANGE_RAD)}
                                      if robot_id == "franka_panda_locked_j3" else {}),
            }
        except Exception as error:
            failures.append(f"{robot_id}: {type(error).__name__}: {error}")
    return {
        "status": "pass" if not failures and len(robots) == len(MODELS) else "fail",
        "geometry_policy": "official_vendor_native_dimensions",
        "sample_count": sample_count,
        "failures": failures,
        "robots": robots,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "configs/robot_registry_13.yaml"
    )
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/single_arm/model_audit.json",
    )
    args = parser.parse_args()
    payload = audit_registry(args.registry, sample_count=args.sample_count)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failures": payload["failures"]}, indent=2))
    print(output)
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
