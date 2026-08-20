"""Fixed-timestamp joint-path projection with hard derivative limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class FixedTimeRefinementResult:
    q: np.ndarray
    time_s: np.ndarray
    success: bool
    maximum_velocity_rad_s: float
    maximum_acceleration_rad_s2: float
    maximum_jerk_rad_s3: float
    maximum_reference_deviation_rad: float


def _derivatives(q: np.ndarray, time_s: np.ndarray):
    dt = np.diff(time_s)
    velocity = np.diff(q, axis=0) / dt[:, None]
    acceleration_dt = 0.5 * (dt[1:] + dt[:-1])
    acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
    jerk = np.diff(acceleration, axis=0) / dt[1:-1, None]
    return velocity, acceleration, jerk


def _limit(value, dof: int, name: str) -> np.ndarray:
    result = np.broadcast_to(np.asarray(value, dtype=float), (dof,)).copy()
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    return result


def refine_fixed_time_joint_path(
    q_reference: np.ndarray,
    time_s: np.ndarray,
    *,
    velocity_limit_rad_s,
    acceleration_limit_rad_s2,
    jerk_limit_rad_s3,
    lower_rad,
    upper_rad,
) -> FixedTimeRefinementResult:
    """Project a reference path into fixed-time velocity/acceleration/jerk bounds.

    The timestamps are immutable.  Each joint is an independent sparse LP that
    minimizes L1 deviation from the supplied IK branch while enforcing all
    derivative and physical joint bounds globally.
    """
    reference = np.asarray(q_reference, dtype=float)
    times = np.asarray(time_s, dtype=float)
    if reference.ndim != 2 or times.shape != (len(reference),):
        raise ValueError("q_reference and time_s have incompatible shapes")
    if len(reference) < 4 or np.any(~np.isfinite(reference)):
        raise ValueError("at least four finite trajectory knots are required")
    dt = np.diff(times)
    if np.any(~np.isfinite(times)) or np.any(dt <= 0.0):
        raise ValueError("time_s must be finite and strictly increasing")
    count, dof = reference.shape
    velocity_limit = _limit(velocity_limit_rad_s, dof, "velocity_limit_rad_s")
    acceleration_limit = _limit(
        acceleration_limit_rad_s2, dof, "acceleration_limit_rad_s2")
    jerk_limit = _limit(jerk_limit_rad_s3, dof, "jerk_limit_rad_s3")
    lower = np.broadcast_to(np.asarray(lower_rad, dtype=float), (dof,)).copy()
    upper = np.broadcast_to(np.asarray(upper_rad, dtype=float), (dof,)).copy()
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("joint limits must be finite and ordered")

    projected = np.empty_like(reference)
    objective = np.r_[np.zeros(count), np.ones(count)]
    for joint in range(dof):
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        rhs: list[float] = []

        def inequality(coefficients: dict[int, float], bound: float) -> None:
            row = len(rhs)
            for column, value in coefficients.items():
                rows.append(row); cols.append(column); values.append(value)
            rhs.append(float(bound))

        # |q-reference| <= slack.
        for knot in range(count):
            inequality({knot: 1.0, count + knot: -1.0}, reference[knot, joint])
            inequality({knot: -1.0, count + knot: -1.0}, -reference[knot, joint])

        # |dq/dt| <= velocity limit.
        for edge, edge_dt in enumerate(dt):
            coefficients = {edge: -1.0 / edge_dt, edge + 1: 1.0 / edge_dt}
            inequality(coefficients, velocity_limit[joint])
            inequality({key: -value for key, value in coefficients.items()},
                       velocity_limit[joint])

        # Acceleration is the difference of adjacent interval velocities.
        acceleration_rows = []
        acceleration_dt = 0.5 * (dt[1:] + dt[:-1])
        for index, scale_dt in enumerate(acceleration_dt):
            coefficients = {
                index: 1.0 / (dt[index] * scale_dt),
                index + 1: -(1.0 / dt[index] + 1.0 / dt[index + 1]) / scale_dt,
                index + 2: 1.0 / (dt[index + 1] * scale_dt),
            }
            acceleration_rows.append(coefficients)
            inequality(coefficients, acceleration_limit[joint])
            inequality({key: -value for key, value in coefficients.items()},
                       acceleration_limit[joint])

        # Jerk is the difference of adjacent accelerations at source knots.
        for index in range(len(acceleration_rows) - 1):
            coefficients: dict[int, float] = {}
            for key, value in acceleration_rows[index + 1].items():
                coefficients[key] = coefficients.get(key, 0.0) + value / dt[index + 1]
            for key, value in acceleration_rows[index].items():
                coefficients[key] = coefficients.get(key, 0.0) - value / dt[index + 1]
            inequality(coefficients, jerk_limit[joint])
            inequality({key: -value for key, value in coefficients.items()},
                       jerk_limit[joint])

        matrix = coo_matrix((values, (rows, cols)), shape=(len(rhs), 2 * count)).tocsr()
        bounds = [(lower[joint], upper[joint])] * count + [(0.0, None)] * count
        solution = linprog(objective, A_ub=matrix, b_ub=np.asarray(rhs),
                           bounds=bounds, method="highs")
        if not solution.success:
            raise RuntimeError(f"fixed-time joint refinement failed for joint {joint}: "
                               f"{solution.message}")
        projected[:, joint] = solution.x[:count]

    velocity, acceleration, jerk = _derivatives(projected, times)
    return FixedTimeRefinementResult(
        q=projected, time_s=times.copy(), success=True,
        maximum_velocity_rad_s=float(np.max(np.abs(velocity))),
        maximum_acceleration_rad_s2=float(np.max(np.abs(acceleration))),
        maximum_jerk_rad_s3=float(np.max(np.abs(jerk))),
        maximum_reference_deviation_rad=float(np.max(np.abs(projected - reference))),
    )


__all__ = ["FixedTimeRefinementResult", "refine_fixed_time_joint_path"]
