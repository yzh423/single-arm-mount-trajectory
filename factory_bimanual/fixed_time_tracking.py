"""Local joint tracking primitives for immutable source timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundedJointStep:
    q: np.ndarray
    velocity_rad_s: np.ndarray
    limited: bool


def fixed_time_derivatives(q, time_s, *, initial_velocity_rad_s=0.0):
    """Return frame-aligned interval velocity and acceleration commands."""
    position = np.asarray(q, dtype=float)
    times = np.asarray(time_s, dtype=float)
    if position.ndim != 2 or times.shape != (len(position),):
        raise ValueError("q and time_s have incompatible shapes")
    if len(position) < 2 or np.any(~np.isfinite(position)):
        raise ValueError("at least two finite joint rows are required")
    dt = np.diff(times)
    if np.any(~np.isfinite(times)) or np.any(dt <= 0.0):
        raise ValueError("time_s must be finite and strictly increasing")
    initial = np.broadcast_to(
        np.asarray(initial_velocity_rad_s, dtype=float),
        (position.shape[1],)).copy()
    velocity = np.empty_like(position)
    velocity[0] = initial
    velocity[1:] = np.diff(position, axis=0) / dt[:, None]
    acceleration = np.zeros_like(position)
    acceleration[1] = (velocity[1] - velocity[0]) / dt[0]
    if len(position) > 2:
        acceleration_dt = 0.5 * (dt[1:] + dt[:-1])
        acceleration[2:] = (
            np.diff(velocity[1:], axis=0) / acceleration_dt[:, None])
    return velocity, acceleration


def _positive_limit(value, shape, name):
    result = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    return result


def bounded_joint_step(
        *, previous_q, desired_q, previous_velocity_rad_s,
        dt_s, previous_dt_s, velocity_limit_rad_s,
        acceleration_limit_rad_s2, lower_rad=None,
        upper_rad=None) -> BoundedJointStep:
    """Move closest to ``desired_q`` under fixed-time v/a limits.

    Acceleration is the change between adjacent interval velocities divided
    by the mean duration of those intervals.  The timestamps themselves are
    never changed.
    """
    previous = np.asarray(previous_q, dtype=float)
    desired = np.asarray(desired_q, dtype=float)
    old_velocity = np.asarray(previous_velocity_rad_s, dtype=float)
    if (previous.shape != desired.shape
            or previous.shape != old_velocity.shape):
        raise ValueError("joint position and velocity shapes must match")
    if not all(np.all(np.isfinite(value))
               for value in (previous, desired, old_velocity)):
        raise ValueError("joint position and velocity must be finite")
    dt = float(dt_s)
    previous_dt = float(previous_dt_s)
    if (not np.isfinite(dt) or dt <= 0.0
            or not np.isfinite(previous_dt) or previous_dt <= 0.0):
        raise ValueError("source intervals must be positive and finite")
    velocity_limit = _positive_limit(
        velocity_limit_rad_s, previous.shape, "velocity_limit_rad_s")
    acceleration_limit = _positive_limit(
        acceleration_limit_rad_s2, previous.shape,
        "acceleration_limit_rad_s2")
    acceleration_delta = acceleration_limit * 0.5 * (previous_dt + dt)
    lower = np.maximum(-velocity_limit, old_velocity - acceleration_delta)
    upper = np.minimum(velocity_limit, old_velocity + acceleration_delta)
    if (lower_rad is None) != (upper_rad is None):
        raise ValueError("lower_rad and upper_rad must be supplied together")
    if lower_rad is not None:
        joint_lower = np.broadcast_to(
            np.asarray(lower_rad, dtype=float), previous.shape)
        joint_upper = np.broadcast_to(
            np.asarray(upper_rad, dtype=float), previous.shape)
        if (np.any(~np.isfinite(joint_lower))
                or np.any(~np.isfinite(joint_upper))
                or np.any(joint_lower >= joint_upper)):
            raise ValueError("joint limits must be finite and ordered")
        lower = np.maximum(lower, (joint_lower - previous) / dt)
        upper = np.minimum(upper, (joint_upper - previous) / dt)
        # Preserve a continuously feasible braking envelope near hard stops.
        # Without this cap a greedy tracker can arrive at a joint limit with
        # nonzero velocity, making the next immutable-time step impossible.
        # Reserve the distance travelled during this interval before applying
        # the usual continuous stopping-distance bound.  The positive root of
        # v*dt + v**2/(2*a) <= distance is conservative for the next command
        # interval and avoids arriving near a hard stop with no braking step.
        positive_braking = np.maximum(
            0.0,
            -acceleration_limit * dt + np.sqrt(np.maximum(
                0.0,
                (acceleration_limit * dt) ** 2
                + 2.0 * acceleration_limit * (joint_upper - previous))),
        )
        negative_braking = np.maximum(
            0.0,
            -acceleration_limit * dt + np.sqrt(np.maximum(
                0.0,
                (acceleration_limit * dt) ** 2
                + 2.0 * acceleration_limit * (previous - joint_lower))),
        )
        lower = np.maximum(lower, -negative_braking)
        upper = np.minimum(upper, positive_braking)
    if np.any(lower > upper + 1e-12):
        raise ValueError(
            "no dynamically feasible joint step inside joint limits")
    desired_velocity = (desired - previous) / dt
    velocity = np.clip(desired_velocity, lower, upper)
    q = previous + velocity * dt
    return BoundedJointStep(
        q=q,
        velocity_rad_s=velocity,
        limited=not np.allclose(q, desired, rtol=0.0, atol=1e-12),
    )


__all__ = [
    "BoundedJointStep", "bounded_joint_step", "fixed_time_derivatives",
]
