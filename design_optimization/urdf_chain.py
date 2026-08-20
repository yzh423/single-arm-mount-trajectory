"""Native-DOF serial-chain extraction and reach measurement."""

from __future__ import annotations

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np

from .robot_registry import RobotSpec


@dataclass(frozen=True)
class NativeSerialChain:
    name: str
    joint_names: tuple[str, ...]
    axes: np.ndarray
    points: np.ndarray
    home: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    deltas: np.ndarray

    @property
    def dof(self) -> int:
        return len(self.joint_names)


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"expected finite 3-vector, got {text!r}")
    return values


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = _vector(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = _vector(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    transform = np.eye(4)
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = xyz
    return transform


def load_urdf_chain(spec: RobotSpec) -> NativeSerialChain:
    if spec.model_format != "urdf":
        raise ValueError(f"{spec.robot_id}: URDF chain requested for {spec.model_format}")
    root = ET.parse(spec.source_model).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    by_child = {
        joint.find("child").get("link"): joint
        for joint in joints.values()
        if joint.find("child") is not None
    }
    path: list[ET.Element] = []
    link = spec.flange_link
    visited: set[str] = set()
    while link != spec.base_link:
        if link in visited:
            raise ValueError(f"{spec.robot_id}: cycle in URDF chain at {link}")
        visited.add(link)
        joint = by_child.get(link)
        if joint is None or joint.find("parent") is None:
            raise ValueError(
                f"{spec.robot_id}: no chain from {spec.base_link} to {spec.flange_link}"
            )
        path.append(joint)
        link = joint.find("parent").get("link")
    path.reverse()
    active_set = set(spec.active_joints)
    path_active = [joint.get("name") for joint in path if joint.get("name") in active_set]
    if tuple(path_active) != spec.active_joints:
        raise ValueError(
            f"{spec.robot_id}: configured active joints do not match base-to-flange order"
        )

    transform = np.eye(4)
    axes: list[np.ndarray] = []
    points: list[np.ndarray] = []
    q_min: list[float] = []
    q_max: list[float] = []
    for joint in path:
        transform = transform @ _origin_transform(joint)
        name = joint.get("name")
        if name not in active_set:
            if joint.get("type") != "fixed":
                raise ValueError(f"{spec.robot_id}: unregistered moving joint {name}")
            continue
        joint_type = joint.get("type")
        if joint_type not in {"revolute", "continuous"}:
            raise ValueError(f"{spec.robot_id}: unsupported active joint type {joint_type}")
        axis_element = joint.find("axis")
        local_axis = _vector(
            axis_element.get("xyz") if axis_element is not None else None,
            (1.0, 0.0, 0.0),
        )
        local_axis /= np.linalg.norm(local_axis)
        axes.append(transform[:3, :3] @ local_axis)
        points.append(transform[:3, 3].copy())
        if name in spec.locked_joints_rad:
            lower, upper = spec.locked_joints_rad[name]
        elif joint_type == "continuous":
            lower, upper = -math.pi, math.pi
        else:
            limit = joint.find("limit")
            if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                raise ValueError(f"{spec.robot_id}: missing position limits for {name}")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"{spec.robot_id}: invalid position limits for {name}")
        q_min.append(lower)
        q_max.append(upper)

    point_array = np.asarray(points, dtype=float)
    home = transform.copy()
    deltas = np.vstack((np.diff(point_array, axis=0), home[:3, 3] - point_array[-1]))
    return NativeSerialChain(
        spec.robot_id,
        spec.active_joints,
        np.asarray(axes, dtype=float),
        point_array,
        home,
        np.asarray(q_min, dtype=float),
        np.asarray(q_max, dtype=float),
        deltas,
    )


def load_mjcf_chain(spec: RobotSpec) -> NativeSerialChain:
    if spec.model_format != "mjcf":
        raise ValueError(f"{spec.robot_id}: MJCF chain requested for {spec.model_format}")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(spec.source_model))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.base_link
    )
    if base_id < 0:
        raise ValueError(f"{spec.robot_id}: missing base body {spec.base_link}")
    base_rotation = data.xmat[base_id].reshape(3, 3).copy()
    base_position = data.xpos[base_id].copy()
    points: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    q_min: list[float] = []
    q_max: list[float] = []
    for name in spec.active_joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"{spec.robot_id}: missing joint {name}")
        points.append(base_rotation.T @ (data.xanchor[joint_id] - base_position))
        axes.append(base_rotation.T @ data.xaxis[joint_id])
        if name in spec.locked_joints_rad:
            lower, upper = spec.locked_joints_rad[name]
        else:
            lower, upper = model.jnt_range[joint_id]
        q_min.append(float(lower))
        q_max.append(float(upper))
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, spec.flange_link
    )
    if site_id < 0:
        raise ValueError(f"{spec.robot_id}: missing flange site {spec.flange_link}")
    home = np.eye(4)
    home[:3, :3] = base_rotation.T @ data.site_xmat[site_id].reshape(3, 3)
    home[:3, 3] = base_rotation.T @ (data.site_xpos[site_id] - base_position)
    strip = np.eye(4)
    strip[2, 3] = -spec.source_tool_length_m
    home = home @ strip
    point_array = np.asarray(points)
    deltas = np.vstack((np.diff(point_array, axis=0), home[:3, 3] - point_array[-1]))
    return NativeSerialChain(
        spec.robot_id,
        spec.active_joints,
        np.asarray(axes),
        point_array,
        home,
        np.asarray(q_min),
        np.asarray(q_max),
        deltas,
    )


def load_native_chain(spec: RobotSpec) -> NativeSerialChain:
    if spec.model_format == "urdf":
        return load_urdf_chain(spec)
    if spec.model_format == "mjcf":
        return load_mjcf_chain(spec)
    raise ValueError(f"{spec.robot_id}: unsupported model format {spec.model_format}")


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    identity = np.eye(3)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def fk_flange(chain: NativeSerialChain, q: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    for axis, point, angle in zip(chain.axes, chain.points, np.asarray(q, dtype=float)):
        rotation = _axis_angle(axis, float(angle))
        joint = np.eye(4)
        joint[:3, :3] = rotation
        joint[:3, 3] = point - rotation @ point
        transform = transform @ joint
    return transform @ chain.home


def _joint_samples(chain: NativeSerialChain, sample_count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    samples = generator.uniform(chain.q_min, chain.q_max, size=(sample_count, chain.dof))
    center = ((chain.q_min + chain.q_max) / 2.0)[None]
    return np.vstack((center, samples))


def sampled_maximum_reach(
    chain: NativeSerialChain, *, sample_count: int = 4096, seed: int = 20260814
) -> float:
    return max(
        float(np.linalg.norm(fk_flange(chain, q)[:3, 3]))
        for q in _joint_samples(chain, sample_count, seed)
    )
