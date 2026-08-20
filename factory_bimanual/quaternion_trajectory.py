"""Timestamp-preserving quaternion trajectory reconstruction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuaternionReconstruction:
    quaternion_wxyz: np.ndarray
    changed: np.ndarray
    change_rad: np.ndarray


def quaternion_poses_equal(first: np.ndarray, second: np.ndarray,
                           *, atol_rad: float = 1e-10) -> bool:
    """Return whether two WXYZ quaternions encode the same rotation."""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != (4,) or right.shape != (4,):
        raise ValueError("quaternions must have shape (4,)")
    left_norm = np.linalg.norm(left); right_norm = np.linalg.norm(right)
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("quaternions must be nonzero")
    return _angular_distance(left / left_norm, right / right_norm) <= atol_rad


def _angular_distance(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first, second)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    second = second.copy()
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = (1.0 - fraction) * first + fraction * second
    else:
        angle = np.arccos(dot)
        result = (np.sin((1.0 - fraction) * angle) * first
                  + np.sin(fraction * angle) * second) / np.sin(angle)
    return result / np.linalg.norm(result)


def reconstruct_held_quaternions(
    time_s: np.ndarray,
    quaternion_wxyz: np.ndarray,
    *,
    stationary_rad: float = 1e-9,
    motion_rad: float = 1e-6,
) -> QuaternionReconstruction:
    """Place held intermediate samples back onto the next quaternion edge."""
    time = np.asarray(time_s, dtype=float)
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if time.ndim != 1 or quaternion.shape != (len(time), 4) or len(time) < 2:
        raise ValueError("timestamps and quaternions have inconsistent shapes")
    if np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(~np.isfinite(quaternion)) or np.any(norms <= 0.0):
        raise ValueError("quaternions must be finite and nonzero")
    if not 0.0 <= stationary_rad < motion_rad:
        raise ValueError("quaternion motion thresholds are invalid")
    normalized = quaternion / norms[:, None]
    for row in range(1, len(normalized)):
        if np.dot(normalized[row - 1], normalized[row]) < 0.0:
            normalized[row] *= -1.0
    output = normalized.copy()
    edge = np.asarray([_angular_distance(a, b)
                       for a, b in zip(normalized[:-1], normalized[1:])])
    changed = np.zeros(len(time), dtype=bool)
    change = np.zeros(len(time), dtype=float)
    for row in range(1, len(time) - 1):
        if edge[row - 1] <= stationary_rad and edge[row] >= motion_rad:
            fraction = float((time[row] - time[row - 1])
                             / (time[row + 1] - time[row - 1]))
            output[row] = _slerp(normalized[row - 1], normalized[row + 1], fraction)
            changed[row] = True
            change[row] = _angular_distance(normalized[row], output[row])
    return QuaternionReconstruction(output, changed, change)
