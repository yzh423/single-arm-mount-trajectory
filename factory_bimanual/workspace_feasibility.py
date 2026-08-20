"""Rigid-transform-invariant necessary workspace checks before experiments.

The geometric chain bound is deliberately conservative: every serial-link offset
is added by the triangle inequality.  Consequently an ``impossible`` result is a
proof, while sampled reach can only support ``unknown`` (never a false proof of
feasibility).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, distance


@dataclass(frozen=True)
class TrajectoryGeometry:
    point_count: int
    diameter_m: float
    enclosing_center_m: tuple[float, float, float]
    enclosing_radius_m: float


def trajectory_geometry(points: np.ndarray) -> TrajectoryGeometry:
    """Return exact diameter and a deterministic enclosing-ball approximation."""
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("points must be a non-empty finite Nx3 array")
    if len(xyz) == 1:
        center, radius, diameter = xyz[0].copy(), 0.0, 0.0
    else:
        try:
            hull_points = xyz[ConvexHull(xyz, qhull_options="QJ").vertices]
        except Exception:
            hull_points = xyz
        condensed = distance.pdist(hull_points)
        diameter = float(condensed.max(initial=0.0))

        # Ritter ball followed by a final numerical enclosure correction.
        seed = hull_points[0]
        a = hull_points[np.argmax(np.linalg.norm(hull_points - seed, axis=1))]
        b = hull_points[np.argmax(np.linalg.norm(hull_points - a, axis=1))]
        center = (a + b) / 2.0
        radius = float(np.linalg.norm(b - a) / 2.0)
        for point in hull_points:
            d = float(np.linalg.norm(point - center))
            if d > radius:
                new_radius = (radius + d) / 2.0
                center += ((new_radius - radius) / d) * (point - center)
                radius = new_radius
        radius = float(np.linalg.norm(xyz - center, axis=1).max(initial=0.0))
    return TrajectoryGeometry(len(xyz), diameter, tuple(map(float, center)), radius)


def _root_and_chain(model: mujoco.MjModel, site_id: int) -> tuple[int, list[int]]:
    chain: list[int] = []
    body = int(model.site_bodyid[site_id])
    while body:
        chain.append(body)
        body = int(model.body_parentid[body])
    return chain[-1], chain


def _kinematic_chain_bound(model: mujoco.MjModel, chain: list[int], site_id: int) -> float:
    """Bound TCP motion about the fixed mount, excluding world placement."""
    movable = [body for body in chain if int(model.body_jntnum[body]) > 0]
    if not movable:
        return float(np.linalg.norm(model.site_pos[site_id]))
    outermost_movable = movable[-1]
    included = chain[:chain.index(outermost_movable) + 1]
    return (sum(float(np.linalg.norm(model.body_pos[body])) for body in included)
            + float(np.linalg.norm(model.site_pos[site_id])))


def _reach_audit(scene: Path, side: str, sample_count: int) -> dict:
    model = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
    if site_id < 0:
        raise ValueError(f"scene lacks {side}_tcp")
    root, chain = _root_and_chain(model, site_id)
    geometric_bound = _kinematic_chain_bound(model, chain, site_id)

    joint_ids = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if name.startswith(f"{side}_") and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
            joint_ids.append(joint_id)
    maximum = 0.0
    count = max(1, int(sample_count))
    for sample in range(count):
        for axis, joint_id in enumerate(joint_ids):
            lo, hi = map(float, model.jnt_range[joint_id]) if model.jnt_limited[joint_id] else (-np.pi, np.pi)
            phase = ((sample + 1) * (axis * 2 + 3) * 0.6180339887498949) % 1.0
            data.qpos[int(model.jnt_qposadr[joint_id])] = lo + phase * (hi - lo)
        mujoco.mj_forward(model, data)
        maximum = max(maximum, float(np.linalg.norm(data.site_xpos[site_id] - data.xpos[root])))
    return {
        "empirical_max_radius_m": maximum,
        "reachable_radius_upper_bound_m": geometric_bound,
        "upper_bound_method": "serial_chain_triangle_inequality",
        "deterministic_joint_samples": count,
    }


def audit_workspace_feasibility(
    task_name: str,
    trajectories: Mapping[str, np.ndarray],
    scenes: Mapping[str, Path],
    *,
    sample_count: int = 256,
    require_targets_above_mount: bool = False,
) -> dict:
    """Audit each robot/hand without cropping, scaling, or choosing registration."""
    geometries = {side: trajectory_geometry(points) for side, points in trajectories.items()}
    results = []
    for robot in sorted(scenes):
        for side in sorted(geometries):
            geometry = geometries[side]
            reach = _reach_audit(scenes[robot], side, sample_count)
            z_span = float(np.ptp(np.asarray(trajectories[side], dtype=float)[:, 2]))
            diameter_impossible = geometry.diameter_m > 2.0 * reach["reachable_radius_upper_bound_m"] + 1e-12
            vertical_impossible = (require_targets_above_mount and
                z_span > reach["reachable_radius_upper_bound_m"] + 1e-12)
            impossible = diameter_impossible or vertical_impossible
            proof = None
            if diameter_impossible:
                proof = {
                    "inequality": "trajectory_diameter_m > 2 * reachable_radius_upper_bound_m",
                    "reason": "No translation or rotation can place both diameter endpoints inside the robot reach ball.",
                }
            elif vertical_impossible:
                proof = {
                    "inequality": "trajectory_z_span_m > reachable_radius_upper_bound_m",
                    "reason": "With an upright table mount and every target at or above the mount plane, the highest target must be at least the full Z span from the mount and exceeds the serial-chain reach bound.",
                }
            results.append({
                "task": str(task_name), "robot": robot, "side": side,
                "status": "impossible" if impossible else "unknown",
                "trajectory_point_count": geometry.point_count,
                "trajectory_diameter_m": geometry.diameter_m,
                "trajectory_z_span_m": z_span,
                "minimum_enclosing_ball_approx_radius_m": geometry.enclosing_radius_m,
                **reach, "proof": proof,
            })
    return {
        "schema_version": 1,
        "task": str(task_name),
        "constraints": ({"full_trajectory": True, "crop": False, "scale": False, "shared_rigid_transform": True}
                        | ({"targets_at_or_above_mount": True} if require_targets_above_mount else {})),
        "interpretation": "impossible is proven; unknown requires registration and IK; empirical samples are not a feasibility proof",
        "results": results,
    }
