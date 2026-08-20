"""Large-crossing gate for fixed-base bimanual MuJoCo mount screening."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class MountTopologyConfig:
    structural_crossing_limit_m: float = .030
    gripper_overlap_limit_m: float = .080
    transition_steps: int = 5

    def __post_init__(self):
        if (self.structural_crossing_limit_m < 0
                or self.gripper_overlap_limit_m < 0
                or self.transition_steps < 2):
            raise ValueError("invalid mount-topology configuration")


@dataclass(frozen=True)
class MountTopologyReport:
    maximum_structural_crossing_m: float
    structural_crossing_count: int
    gripper_overlap_m: float
    gripper_overlap_count: int
    valid: bool


def evaluate_mount_topology_positions(
        *, left_base, right_base, left_structural, right_structural,
        left_tcp, right_tcp,
        config: MountTopologyConfig = MountTopologyConfig(),
) -> MountTopologyReport:
    """Classify left/right order in the fixed base-pair coordinate frame."""
    left_base = np.asarray(left_base, float)
    right_base = np.asarray(right_base, float)
    left_structural = np.asarray(left_structural, float)
    right_structural = np.asarray(right_structural, float)
    left_tcp = np.asarray(left_tcp, float)
    right_tcp = np.asarray(right_tcp, float)
    if (left_base.shape != (3,) or right_base.shape != (3,)
            or left_tcp.shape != (3,) or right_tcp.shape != (3,)
            or left_structural.ndim != 2
            or left_structural.shape[1:] != (3,)
            or right_structural.shape != left_structural.shape):
        raise ValueError("mount topology positions have invalid shapes")
    axis = right_base - left_base
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("left/right base positions must be distinct")
    axis /= norm
    structural_depths = np.maximum(
        0., (left_structural - right_structural) @ axis)
    maximum = float(np.max(structural_depths, initial=0.))
    structural_count = int(np.count_nonzero(
        structural_depths > config.structural_crossing_limit_m + 1e-12))
    gripper_overlap = max(0., float((left_tcp - right_tcp) @ axis))
    gripper_count = int(gripper_overlap > 1e-12)
    valid = (structural_count == 0
             and gripper_overlap <= config.gripper_overlap_limit_m + 1e-12)
    return MountTopologyReport(
        maximum, structural_count, gripper_overlap, gripper_count, valid)


class MuJoCoMountTopologyChecker:
    """Evaluate structural/TCP ordering for paired joint states and edges."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 name_map: dict[str, dict[str, Any]], *,
                 config: MountTopologyConfig = MountTopologyConfig()):
        self.model = model
        self.data = mujoco.MjData(model)
        self.base_qpos = np.asarray(data.qpos).copy()
        self.config = config
        self.qids = {}
        for side in ("left", "right"):
            joint_ids = [mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in name_map[side]["joints"]]
            if any(index < 0 for index in joint_ids):
                raise ValueError(f"missing {side} topology joint")
            self.qids[side] = np.asarray(
                [model.jnt_qposadr[index] for index in joint_ids], int)
        self.body_ids = {}
        for side in ("left", "right"):
            for anchor in ("base_mount", "link3", "link5"):
                name = f"{side}_{anchor}"
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    raise ValueError(f"missing topology body: {name}")
                self.body_ids[(side, anchor)] = body_id
        self.site_ids = {}
        for side in ("left", "right"):
            name = name_map[side].get("site", f"{side}_tcp")
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id < 0:
                raise ValueError(f"missing topology site: {name}")
            self.site_ids[side] = site_id

    def state(self, left_q, right_q) -> MountTopologyReport:
        self.data.qpos[:] = self.base_qpos
        self.data.qpos[self.qids["left"]] = left_q
        self.data.qpos[self.qids["right"]] = right_q
        mujoco.mj_forward(self.model, self.data)
        structural = {}
        for side in ("left", "right"):
            structural[side] = np.asarray([
                self.data.xpos[self.body_ids[(side, "link3")]],
                self.data.xpos[self.body_ids[(side, "link5")]],
            ])
        return evaluate_mount_topology_positions(
            left_base=self.data.xpos[self.body_ids[("left", "base_mount")]],
            right_base=self.data.xpos[self.body_ids[("right", "base_mount")]],
            left_structural=structural["left"],
            right_structural=structural["right"],
            left_tcp=self.data.site_xpos[self.site_ids["left"]],
            right_tcp=self.data.site_xpos[self.site_ids["right"]],
            config=self.config)

    def transition(self, previous, current) -> MountTopologyReport:
        reports = []
        for alpha in np.linspace(0., 1., self.config.transition_steps):
            reports.append(self.state(
                (1-alpha)*np.asarray(previous[0])+alpha*np.asarray(current[0]),
                (1-alpha)*np.asarray(previous[1])+alpha*np.asarray(current[1])))
        maximum = max(item.maximum_structural_crossing_m for item in reports)
        structural_count = sum(not item.valid and
                               item.structural_crossing_count > 0
                               for item in reports)
        gripper_overlap = max(item.gripper_overlap_m for item in reports)
        gripper_count = sum(item.gripper_overlap_count > 0 for item in reports)
        valid = all(item.valid for item in reports)
        return MountTopologyReport(
            maximum, int(structural_count), gripper_overlap,
            int(gripper_count), valid)
