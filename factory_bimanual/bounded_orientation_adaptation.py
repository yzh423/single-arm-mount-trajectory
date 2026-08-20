"""Bounded, auditable orientation alternatives for PiperX TCP targets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class OrientationAdaptationConfig:
    tool_soft_limit_deg: float = 15.0
    tool_hard_limit_deg: float = 30.0
    swing_soft_limit_deg: float = 7.5
    swing_hard_limit_deg: float = 15.0
    tool_step_deg: float = 15.0
    swing_step_deg: float = 7.5

    def __post_init__(self):
        values = np.asarray([
            self.tool_soft_limit_deg, self.tool_hard_limit_deg,
            self.swing_soft_limit_deg, self.swing_hard_limit_deg,
            self.tool_step_deg, self.swing_step_deg,
        ], dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("orientation adaptation limits must be finite")
        if np.any(values <= 0.0):
            raise ValueError("orientation adaptation limits and steps must be positive")
        if self.tool_soft_limit_deg > self.tool_hard_limit_deg:
            raise ValueError("tool soft limit must not exceed its hard limit")
        if self.swing_soft_limit_deg > self.swing_hard_limit_deg:
            raise ValueError("swing soft limit must not exceed its hard limit")


@dataclass(frozen=True)
class OrientationCandidate:
    quaternion_wxyz: np.ndarray
    offset_rotvec_rad: np.ndarray
    level: str
    source: str
    tool_axis_offset_deg: float
    swing_offset_deg: float


def _normalize_quaternion_wxyz(quaternion):
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or np.any(~np.isfinite(value)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("quaternion must be nonzero")
    return value / norm


def _wxyz_to_rotation(quaternion):
    value = _normalize_quaternion_wxyz(quaternion)
    return Rotation.from_quat([value[1], value[2], value[3], value[0]])


def _rotation_to_wxyz(rotation):
    x, y, z, w = rotation.as_quat()
    value = np.asarray([w, x, y, z])
    if value[0] < 0.0:
        value *= -1.0
    return value / np.linalg.norm(value)


def orientation_offset_angles(offset_rotvec_rad):
    """Return signed local tool-axis twist and swing magnitude in degrees."""
    value = np.asarray(offset_rotvec_rad, dtype=float)
    if value.shape != (3,) or np.any(~np.isfinite(value)):
        raise ValueError("orientation offset must contain three finite values")
    return float(np.rad2deg(value[2])), float(np.rad2deg(np.linalg.norm(value[:2])))


def _clip_offset(offset, config):
    value = np.asarray(offset, dtype=float).copy()
    value[2] = np.clip(
        value[2], -np.deg2rad(config.tool_hard_limit_deg),
        np.deg2rad(config.tool_hard_limit_deg))
    swing = float(np.linalg.norm(value[:2]))
    maximum = np.deg2rad(config.swing_hard_limit_deg)
    if swing > maximum:
        value[:2] *= maximum / swing
    return value


def orientation_adaptation_candidates(
        original_quaternion_wxyz, *, previous_offset_rotvec_rad=None,
        config=OrientationAdaptationConfig()):
    """Generate deterministic local-frame alternatives in increasing freedom."""
    original = _wxyz_to_rotation(original_quaternion_wxyz)
    offsets = [(np.zeros(3), "original", "original")]
    tool_values = sorted({
        -config.tool_hard_limit_deg, -config.tool_soft_limit_deg,
        config.tool_soft_limit_deg, config.tool_hard_limit_deg,
    })
    for tool_deg in tool_values:
        offsets.append((np.deg2rad([0.0, 0.0, tool_deg]),
                        "tool_axis", "grid"))
    swing_values = sorted({config.swing_soft_limit_deg,
                           config.swing_hard_limit_deg})
    for swing_deg in swing_values:
        for axis in ((1.0, 0.0), (-1.0, 0.0),
                     (0.0, 1.0), (0.0, -1.0)):
            for tool_deg in (0.0, *tool_values):
                offsets.append((np.deg2rad([
                    axis[0] * swing_deg, axis[1] * swing_deg, tool_deg]),
                    "full_pose", "grid"))
    if previous_offset_rotvec_rad is not None:
        clipped = _clip_offset(previous_offset_rotvec_rad, config)
        tool_deg, swing_deg = orientation_offset_angles(clipped)
        level = "tool_axis" if swing_deg <= 1e-10 else "full_pose"
        offsets.append((clipped, level, "continuation"))

    result = []
    seen = set()
    for offset, level, source in offsets:
        key = tuple(np.round(offset, 12))
        if key in seen:
            continue
        seen.add(key)
        adapted = original * Rotation.from_rotvec(offset)
        tool_deg, swing_deg = orientation_offset_angles(offset)
        result.append(OrientationCandidate(
            quaternion_wxyz=_rotation_to_wxyz(adapted),
            offset_rotvec_rad=np.asarray(offset, dtype=float),
            level=level,
            source=source,
            tool_axis_offset_deg=tool_deg,
            swing_offset_deg=swing_deg,
        ))
    return tuple(result)


__all__ = [
    "OrientationAdaptationConfig",
    "OrientationCandidate",
    "orientation_adaptation_candidates",
    "orientation_offset_angles",
]
