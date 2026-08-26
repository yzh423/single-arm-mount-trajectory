"""Deterministic base orientations for the Piper X mount comparison."""
from __future__ import annotations

import mujoco
import numpy as np


MOUNT_MODES = ("upright_table", "horizontal_wall", "inverted")
SUPPORTED_MOUNT_MODES = (*MOUNT_MODES, "horizontal_forward")
FACTORY_BATCH_MOUNT_MODES = (
    "upright_table", "horizontal_forward", "inverted")


def _matrix_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=float).reshape(9))
    if quat[0] < 0:
        quat *= -1
    return tuple(float(value) for value in quat)


def _rotation_about_local_z(yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(yaw_deg))
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def mount_quaternions(mode, xy, target_center, yaw_deg):
    """Return world-from-base quaternions; local +Z is the mounting axis."""
    if mode not in SUPPORTED_MOUNT_MODES:
        raise ValueError(f"unsupported mount orientation: {mode!r}")
    target_center = np.asarray(target_center, dtype=float)
    if target_center.shape != (3,):
        raise ValueError("target_center must have shape (3,)")
    result = {}
    common_forward = None
    if mode == "horizontal_forward":
        base_midpoint = np.mean(
            [np.asarray(xy[side], dtype=float) for side in ("left", "right")],
            axis=0)
        axis = target_center[:2] - base_midpoint
        norm = np.linalg.norm(axis)
        if norm <= 1e-9:
            raise ValueError("forward mount midpoint cannot coincide with target center")
        common_forward = np.asarray([axis[0] / norm, axis[1] / norm, 0.])
    for side in ("left", "right"):
        if mode == "upright_table":
            frame = np.eye(3)
        elif mode == "inverted":
            frame = np.diag([1., -1., -1.])
        elif mode == "horizontal_wall":
            # A wall-mounted pair extends from each base toward the task.
            axis = target_center[:2] - np.asarray(xy[side], dtype=float)
            norm = np.linalg.norm(axis)
            if norm <= 1e-9:
                raise ValueError("wall mount cannot coincide with target XY center")
            z_axis = np.asarray([axis[0] / norm, axis[1] / norm, 0.])
            y_axis = np.asarray([0., 0., 1.])
            x_axis = np.cross(y_axis, z_axis)
            frame = np.column_stack((x_axis, y_axis, z_axis))
        else:
            z_axis = common_forward
            y_axis = np.asarray([0., 0., 1.])
            x_axis = np.cross(y_axis, z_axis)
            frame = np.column_stack((x_axis, y_axis, z_axis))
        result[side] = _matrix_quaternion(
            frame @ _rotation_about_local_z(yaw_deg[side]))
    return result
