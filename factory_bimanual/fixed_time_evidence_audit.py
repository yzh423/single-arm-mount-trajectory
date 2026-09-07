"""Evidence helpers for fixed-time bimanual tracking audits.

The functions in this module are deliberately solver-independent.  They keep
source-time kinematics, raw-target fidelity, and solver-target conditioning as
separate quantities so a report cannot accidentally mix their meanings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskspaceSegmentRates:
    interval_s: np.ndarray
    linear_speed_m_s: np.ndarray
    angular_speed_rad_s: np.ndarray


@dataclass(frozen=True)
class PoseTrackingErrors:
    position_error_m: np.ndarray
    orientation_error_rad: np.ndarray


@dataclass(frozen=True)
class TargetTrackComparison:
    maximum_position_deviation_m: float
    maximum_orientation_deviation_rad: float
    position_budget_exceeded: bool
    orientation_budget_exceeded: bool


def _array(value: object, *, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))})")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _unit_quaternions(value: object, *, name: str) -> np.ndarray:
    quaternion = _array(value, shape_tail=(4,), name=name)
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm <= np.finfo(float).eps):
        raise ValueError(f"{name} contains a zero quaternion")
    return quaternion / norm[:, None]


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Shortest SO(3) angular distance for WXYZ quaternion rows."""

    dot = np.sum(first * second, axis=1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), -1.0, 1.0))


def taskspace_segment_rates(
    time_s: object,
    position_m: object,
    quaternion_wxyz: object,
) -> TaskspaceSegmentRates:
    """Compute segment rates using the supplied timestamps without retiming."""

    time = np.asarray(time_s, dtype=float)
    position = _array(position_m, shape_tail=(3,), name="position_m")
    quaternion = _unit_quaternions(quaternion_wxyz, name="quaternion_wxyz")
    if time.ndim != 1 or len(time) != len(position) or len(time) != len(quaternion):
        raise ValueError("time and pose tracks must have the same length")
    if len(time) < 2 or not np.all(np.isfinite(time)):
        raise ValueError("time must contain at least two finite samples")
    interval = np.diff(time)
    if np.any(interval <= 0.0):
        raise ValueError("time must be strictly increasing")
    return TaskspaceSegmentRates(
        interval_s=interval,
        linear_speed_m_s=np.linalg.norm(np.diff(position, axis=0), axis=1) / interval,
        angular_speed_rad_s=_quaternion_distance(quaternion[1:], quaternion[:-1]) / interval,
    )


def pose_tracking_errors(
    actual_pose_xyz_wxyz: object,
    target_position_m: object,
    target_quaternion_wxyz: object,
) -> PoseTrackingErrors:
    """Measure actual TCP pose error against an explicitly supplied target."""

    actual = _array(actual_pose_xyz_wxyz, shape_tail=(7,), name="actual_pose_xyz_wxyz")
    target_position = _array(target_position_m, shape_tail=(3,), name="target_position_m")
    target_quaternion = _unit_quaternions(
        target_quaternion_wxyz, name="target_quaternion_wxyz")
    if len(actual) != len(target_position) or len(actual) != len(target_quaternion):
        raise ValueError("actual and target pose tracks must have the same length")
    actual_quaternion = _unit_quaternions(actual[:, 3:7], name="actual quaternion")
    return PoseTrackingErrors(
        position_error_m=np.linalg.norm(actual[:, :3] - target_position, axis=1),
        orientation_error_rad=_quaternion_distance(actual_quaternion, target_quaternion),
    )


def strict_pair_accept(
    left_position_error_m: object,
    right_position_error_m: object,
    left_orientation_error_rad: object,
    right_orientation_error_rad: object,
    *,
    source_valid: object,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> np.ndarray:
    """Return the strict, simultaneous two-hand acceptance mask."""

    arrays = [np.asarray(value) for value in (
        left_position_error_m,
        right_position_error_m,
        left_orientation_error_rad,
        right_orientation_error_rad,
        source_valid,
    )]
    if any(value.ndim != 1 for value in arrays) or len({len(value) for value in arrays}) != 1:
        raise ValueError("all error and validity arrays must be one-dimensional and equally sized")
    if position_tolerance_m < 0.0 or orientation_tolerance_rad < 0.0:
        raise ValueError("tolerances must be non-negative")
    left_position, right_position, left_orientation, right_orientation, valid = arrays
    return (
        valid.astype(bool)
        & (left_position <= position_tolerance_m)
        & (right_position <= position_tolerance_m)
        & (left_orientation <= orientation_tolerance_rad)
        & (right_orientation <= orientation_tolerance_rad)
    )


def compare_target_tracks(
    raw_position_m: object,
    raw_quaternion_wxyz: object,
    conditioned_position_m: object,
    conditioned_quaternion_wxyz: object,
    *,
    position_budget_m: float,
    orientation_budget_rad: float,
) -> TargetTrackComparison:
    """Quantify how far the solver target departs from the raw mapped target."""

    raw_position = _array(raw_position_m, shape_tail=(3,), name="raw_position_m")
    conditioned_position = _array(
        conditioned_position_m, shape_tail=(3,), name="conditioned_position_m")
    raw_quaternion = _unit_quaternions(raw_quaternion_wxyz, name="raw_quaternion_wxyz")
    conditioned_quaternion = _unit_quaternions(
        conditioned_quaternion_wxyz, name="conditioned_quaternion_wxyz")
    lengths = {len(raw_position), len(conditioned_position),
               len(raw_quaternion), len(conditioned_quaternion)}
    if len(lengths) != 1:
        raise ValueError("raw and conditioned target tracks must have the same length")
    position_deviation = np.linalg.norm(conditioned_position - raw_position, axis=1)
    orientation_deviation = _quaternion_distance(conditioned_quaternion, raw_quaternion)
    maximum_position = float(np.max(position_deviation, initial=0.0))
    maximum_orientation = float(np.max(orientation_deviation, initial=0.0))
    return TargetTrackComparison(
        maximum_position_deviation_m=maximum_position,
        maximum_orientation_deviation_rad=maximum_orientation,
        position_budget_exceeded=maximum_position > position_budget_m,
        orientation_budget_exceeded=maximum_orientation > orientation_budget_rad,
    )
