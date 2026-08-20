"""Native-geometry measurements for imported MuJoCo robot specs."""
from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np


def active_chain_length_m(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
    tool_offset_m: float,
) -> float:
    body_ids: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint not found: {name}")
        body_id = int(model.jnt_bodyid[joint_id])
        if not body_ids or body_ids[-1] != body_id:
            body_ids.append(body_id)
    return float(sum(np.linalg.norm(model.body_pos[body_id]) for body_id in body_ids) + tool_offset_m)


def sampled_maximum_tcp_reach_m(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
    site_name: str,
    *,
    sample_count: int = 2048,
) -> float:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"site not found: {site_name}")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError("one or more active joints are missing")
    root = int(model.jnt_bodyid[joint_ids[0]])
    while int(model.body_parentid[root]) != 0:
        root = int(model.body_parentid[root])
    q_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    limits = np.asarray([
        model.jnt_range[joint_id] if model.jnt_limited[joint_id] else (-np.pi, np.pi)
        for joint_id in joint_ids
    ], dtype=float)
    rng = np.random.default_rng(20260814)
    configurations = [0.5 * (limits[:, 0] + limits[:, 1])]
    configurations.extend(rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(sample_count - 1))
    data = mujoco.MjData(model)
    maximum = 0.0
    for q in configurations:
        data.qpos[q_addresses] = q
        mujoco.mj_forward(model, data)
        maximum = max(maximum, float(np.linalg.norm(data.site_xpos[site_id] - data.xpos[root])))
    return maximum
