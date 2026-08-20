"""Bounded SE(3) conditioning used by the complete-follow planner."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ConditioningAudit:
    window: int
    polyorder: int
    maximum_position_deviation_m: float
    maximum_orientation_deviation_rad: float


@dataclass(frozen=True)
class ConditionedTrajectory:
    task: object
    audit: ConditioningAudit


def _bounded_position(values, *, window, polyorder, maximum_deviation_m):
    position = np.asarray(values, dtype=float)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position trajectory must have shape (frames, 3)")
    filtered = savgol_filter(
        position, window, polyorder, axis=0, mode="interp")
    delta = filtered - position
    norm = np.linalg.norm(delta, axis=1)
    scale = np.minimum(
        1.0, maximum_deviation_m / np.maximum(norm, 1e-15))
    return position + delta * scale[:, None]


def _bounded_quaternion(values, *, window, polyorder,
                        maximum_deviation_rad):
    quaternion = np.asarray(values, dtype=float)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion trajectory must have shape (frames, 4)")
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(~np.isfinite(quaternion)) or np.any(norm < 1e-12):
        raise ValueError("quaternion trajectory must be finite and nonzero")
    canonical = quaternion / norm[:, None]
    for row in range(1, len(canonical)):
        if float(canonical[row - 1] @ canonical[row]) < 0.0:
            canonical[row] *= -1.0
    filtered = savgol_filter(
        canonical, window, polyorder, axis=0, mode="interp")
    filtered /= np.linalg.norm(filtered, axis=1)[:, None]
    original_rotation = Rotation.from_quat(canonical[:, [1, 2, 3, 0]])
    filtered_rotation = Rotation.from_quat(filtered[:, [1, 2, 3, 0]])
    residual = (original_rotation.inv() * filtered_rotation).as_rotvec()
    residual_norm = np.linalg.norm(residual, axis=1)
    scale = np.minimum(
        1.0, maximum_deviation_rad / np.maximum(residual_norm, 1e-15))
    bounded = original_rotation * Rotation.from_rotvec(
        residual * scale[:, None])
    xyzw = bounded.as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def bounded_savgol_se3(
    task,
    *,
    sides=("left", "right"),
    window=9,
    polyorder=3,
    maximum_position_deviation_m=0.005,
    maximum_orientation_deviation_rad=np.deg2rad(1.0),
):
    """Apply report algorithm three while bounding deviation from raw poses."""
    window = int(window)
    polyorder = int(polyorder)
    if window < 3 or window % 2 != 1 or polyorder < 0 or polyorder >= window:
        raise ValueError("Savitzky-Golay window must be odd and exceed polyorder")
    if (not np.isfinite(maximum_position_deviation_m) or
            maximum_position_deviation_m < 0.0 or
            not np.isfinite(maximum_orientation_deviation_rad) or
            maximum_orientation_deviation_rad < 0.0):
        raise ValueError("SE(3) deviation bounds must be finite and nonnegative")
    values = task.__dict__.copy()
    maximum_position = 0.0
    maximum_orientation = 0.0
    for side in sides:
        position_name = f"{side}_position_m"
        quaternion_name = f"{side}_quaternion_wxyz"
        position = np.asarray(getattr(task, position_name), dtype=float)
        quaternion = np.asarray(getattr(task, quaternion_name), dtype=float)
        if len(position) < window or len(quaternion) != len(position):
            raise ValueError("trajectory must contain at least one full window")
        smooth_position = _bounded_position(
            position, window=window, polyorder=polyorder,
            maximum_deviation_m=maximum_position_deviation_m)
        smooth_quaternion = _bounded_quaternion(
            quaternion, window=window, polyorder=polyorder,
            maximum_deviation_rad=maximum_orientation_deviation_rad)
        values[position_name] = smooth_position
        values[quaternion_name] = smooth_quaternion
        maximum_position = max(
            maximum_position,
            float(np.max(np.linalg.norm(smooth_position-position, axis=1))),
        )
        original = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
        smooth = Rotation.from_quat(smooth_quaternion[:, [1, 2, 3, 0]])
        maximum_orientation = max(
            maximum_orientation,
            float(np.max(np.linalg.norm(
                (original.inv() * smooth).as_rotvec(), axis=1))),
        )
    return ConditionedTrajectory(
        task=SimpleNamespace(**values),
        audit=ConditioningAudit(
            window=window,
            polyorder=polyorder,
            maximum_position_deviation_m=maximum_position,
            maximum_orientation_deviation_rad=maximum_orientation,
        ),
    )


__all__ = [
    "ConditionedTrajectory",
    "ConditioningAudit",
    "bounded_savgol_se3",
]
