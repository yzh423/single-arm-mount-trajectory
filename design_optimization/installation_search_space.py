"""Physical installation bounds shared by all three benchmark domains."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class InstallationSearchSpace:
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    incumbent: np.ndarray


def first_version_bounds(geometry_dimensions: int = 7) -> InstallationSearchSpace:
    """Return pre-clipped legal translation and three-axis installation bounds."""
    geometry_names = tuple(f"geometry_{index + 1}" for index in range(geometry_dimensions))
    names = geometry_names + ("base_x_m", "base_y_m", "base_z_m", "tilt_pitch_deg", "yaw_deg", "roll_deg")
    # Rz(yaw) @ Ry(pitch/tilt) @ Rx(roll) spans the full installation SO(3).
    lower = np.r_[np.full(geometry_dimensions, -0.9), (-0.50, -0.55, 0.02, -90.0, -180.0, -180.0)]
    upper = np.r_[np.full(geometry_dimensions, 0.9), (0.50, 0.25, 0.65, 90.0, 180.0, 180.0)]
    incumbent = np.r_[np.zeros(geometry_dimensions), (0.0, -0.16, 0.18, 0.0, 0.0, 0.0)]
    return InstallationSearchSpace(names, lower, upper, incumbent)


def mount_rotation_matrix(*, tilt_pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch/tilt) @ Rx(roll)."""
    pitch, yaw, roll = np.deg2rad((tilt_pitch_deg, yaw_deg, roll_deg))
    cp, sp, cy, sy, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    return rz @ ry @ rx


def tabletop_mount_feasible(base_xyz, rotation: np.ndarray, *,
                            table_half_extent=(0.95, 0.72),
                            minimum_upward_normal: float = 0.10) -> bool:
    """Whether an oriented base can be supported by a straight table pedestal."""
    base = np.asarray(base_xyz, dtype=float)
    normal = np.asarray(rotation, dtype=float)[:, 2]
    if base.shape != (3,) or base[2] <= 0.0 or normal[2] <= minimum_upward_normal:
        return False
    # X/Y is the vertical column/table footprint. Orientation belongs only to
    # the top adapter and must not translate that footprint.
    return bool(abs(base[0]) <= table_half_extent[0] and abs(base[1]) <= table_half_extent[1])


def mount_transform_from_record(record: dict[str, object]) -> np.ndarray:
    """Reconstruct the complete saved XYZ/pitch/yaw/roll installation."""
    transform = np.eye(4)
    transform[:3, :3] = mount_rotation_matrix(
        tilt_pitch_deg=float(record["tilt_deg"]),
        yaw_deg=float(record.get("yaw_deg", 0.0)),
        roll_deg=float(record.get("roll_deg", 0.0)),
    )
    transform[:3, 3] = np.asarray(record["base_xyz_m"], dtype=float)
    return transform


def torch_mount_rotation_matrices(tilt_pitch_deg: torch.Tensor, yaw_deg: torch.Tensor,
                                  roll_deg: torch.Tensor) -> torch.Tensor:
    """Batched torch equivalent of :func:`mount_rotation_matrix`."""
    pitch, yaw, roll = (torch.deg2rad(value) for value in (tilt_pitch_deg, yaw_deg, roll_deg))
    cp, sp, cy, sy, cr, sr = torch.cos(pitch), torch.sin(pitch), torch.cos(yaw), torch.sin(yaw), torch.cos(roll), torch.sin(roll)
    zeros, ones = torch.zeros_like(pitch), torch.ones_like(pitch)
    rz = torch.stack((cy,-sy,zeros, sy,cy,zeros, zeros,zeros,ones), dim=-1).reshape(-1,3,3)
    ry = torch.stack((cp,zeros,sp, zeros,ones,zeros, -sp,zeros,cp), dim=-1).reshape(-1,3,3)
    rx = torch.stack((ones,zeros,zeros, zeros,cr,-sr, zeros,sr,cr), dim=-1).reshape(-1,3,3)
    return rz @ ry @ rx
