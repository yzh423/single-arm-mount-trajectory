"""Certify gripper-base clearance infeasibility for fixed raw TCP targets.

The two gripper-base meshes are rigid relative to their respective TCPs.
Consequently their target-pose gap is independent of the robot mounts and IK
branches. A conservative rigid-point displacement bound covers every pose
within the declared strict TCP tolerances.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def paired_clearance_gain_bound_m(
        left_radius_m, right_radius_m, *, position_tolerance_m,
        orientation_tolerance_rad):
    """Maximum increase in pair gap under both allowed TCP pose errors."""
    if (min(left_radius_m, right_radius_m, position_tolerance_m,
            orientation_tolerance_rad) < 0
            or orientation_tolerance_rad > np.pi):
        raise ValueError("radii and pose tolerances must be physical")
    angular_displacement = 2.0 * np.sin(
        orientation_tolerance_rad / 2.0)
    return float(2.0 * position_tolerance_m
                 + (left_radius_m + right_radius_m)
                 * angular_displacement)


def _target_rotation(quaternion_wxyz):
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("target quaternion must be finite and nonzero")
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion / norm)
    return matrix.reshape(3, 3)


def _gripper_geoms(model, data, side):
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
    if site_id < 0:
        raise ValueError(f"missing {side} TCP site")
    tcp_rotation = data.site_xmat[site_id].reshape(3, 3)
    tcp_position = data.site_xpos[site_id]
    geometries = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if (name.startswith(f"{side}_gripper_base")
                and "collision" in name):
            local_position = tcp_rotation.T @ (
                data.geom_xpos[geom_id] - tcp_position)
            local_rotation = (tcp_rotation.T
                              @ data.geom_xmat[geom_id].reshape(3, 3))
            geometries[geom_id] = (local_position, local_rotation)
    if not geometries:
        raise ValueError(f"missing {side} gripper-base collision geometry")
    radius = max(float(np.linalg.norm(position)
                       + model.geom_rbound[geom_id])
                 for geom_id, (position, _) in geometries.items())
    return geometries, radius


def audit_target_clearance(
        scene_path, trajectory_path, *, clearance_margin_m=.015,
        position_tolerance_m=.001,
        orientation_tolerance_rad=np.deg2rad(.5)):
    """Return a conservative, mount-independent infeasibility certificate."""
    model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))
    archive = np.load(trajectory_path)
    times = np.asarray(archive["source_time_s"], dtype=float)
    qpos = np.asarray(archive["qpos"], dtype=float)
    if (len(times) == 0 or qpos.shape != (len(times), model.nq)
            or bool(np.asarray(archive["retiming_applied"]).item())
            or not np.array_equal(times, archive["fixed_time_s"])):
        raise ValueError("archive is not a fixed-source-time joint path")
    targets = {}
    for side in ("left", "right"):
        position = np.asarray(archive[f"{side}_target_position_m"],
                              dtype=float)
        quaternion = np.asarray(
            archive[f"{side}_target_quaternion_wxyz"], dtype=float)
        if (position.shape != (len(times), 3)
                or quaternion.shape != (len(times), 4)
                or not np.all(np.isfinite(position))):
            raise ValueError(f"invalid {side} target track")
        targets[side] = (position, quaternion)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    grippers = {}
    radii = {}
    for side in ("left", "right"):
        grippers[side], radii[side] = _gripper_geoms(model, data, side)
    gain_bound = paired_clearance_gain_bound_m(
        radii["left"], radii["right"],
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad)
    gaps = np.empty(len(times), dtype=float)
    segment = np.empty(6, dtype=float)
    for row in range(len(times)):
        for side in ("left", "right"):
            target_position, target_quaternion = targets[side]
            rotation = _target_rotation(target_quaternion[row])
            for geom_id, (local_position, local_rotation) in (
                    grippers[side].items()):
                data.geom_xpos[geom_id] = (
                    target_position[row] + rotation @ local_position)
                data.geom_xmat[geom_id] = (
                    rotation @ local_rotation).reshape(9)
        gaps[row] = min(float(mujoco.mj_geomDistance(
            model, data, left, right, 10.0, segment))
            for left in grippers["left"] for right in grippers["right"])

    impossible = gaps + gain_bound < clearance_margin_m
    accepted = np.asarray(archive["both_accept"], dtype=bool)
    if accepted.shape != impossible.shape:
        raise ValueError("paired acceptance track has the wrong length")
    indices = np.flatnonzero(impossible).astype(int)
    return {
        "schema": "piperx-v4-rigid-target-clearance-certificate-v1",
        "event_count": len(times),
        "clearance_margin_m": float(clearance_margin_m),
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_rad": float(orientation_tolerance_rad),
        "gripper_radius_from_tcp_m": radii,
        "maximum_tolerance_clearance_gain_m": gain_bound,
        "minimum_exact_target_gap_m": float(np.min(gaps)),
        "exact_target_gap_m": gaps.tolist(),
        "certified_infeasible_indices": indices.tolist(),
        "certified_infeasible_count": len(indices),
        "accepted_infeasible_overlap_count": int(np.count_nonzero(
            accepted & impossible)),
        "acceptance_upper_bound": (len(times) - len(indices)) / len(times),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_target_clearance(args.scene, args.trajectory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "certified_infeasible_count": result["certified_infeasible_count"],
        "acceptance_upper_bound": result["acceptance_upper_bound"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
