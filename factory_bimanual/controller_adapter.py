"""Narrow scene-to-controller adapter for isolated factory experiments."""

from __future__ import annotations

import numpy as np

from .native_dof_easyik import NativeDofDualArmEasyIKPVT
from .native_dof_mpc import (
    MPCConfig,
    NativeDofDualArmMPCPVT,
    UnsupportedModelControllerContract,
)


def _validate(contract, name_map):
    expected = contract.dof_per_arm
    for side in ("left", "right"):
        found = len(name_map.get(side, {}).get("joints", ()))
        if found != expected:
            raise UnsupportedModelControllerContract(
                f"{side} expected {expected} joints but found {found}"
            )


def make_easyik(model, data, contract, config=None, *, name_map, velocity_limits=None):
    _validate(contract, name_map)
    return NativeDofDualArmEasyIKPVT(
        model, data, config, name_map=name_map, robot_kind=contract.name,
        velocity_limits=velocity_limits,
    )


def make_mpc(model, data, contract, config=None, *, name_map, velocity_limits=None):
    _validate(contract, name_map)
    return NativeDofDualArmMPCPVT(
        model, data, config, name_map=name_map, robot_kind=contract.name,
        velocity_limits=velocity_limits,
    )


def set_bimanual_targets(data, controller, left_pose, right_pose):
    for side, pose in (("left", left_pose), ("right", right_pose)):
        position, quaternion = (np.asarray(value, dtype=np.float64) for value in pose)
        if position.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ValueError(f"invalid {side} target pose")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise ValueError(f"invalid {side} target quaternion")
        arm = controller.arms[side]
        data.mocap_pos[arm.mocap_id] = position
        data.mocap_quat[arm.mocap_id] = quaternion / norm


__all__ = ["MPCConfig", "UnsupportedModelControllerContract", "make_easyik", "make_mpc", "set_bimanual_targets"]

