"""Pure branch-continuity and rescue-v3.1 scheduling primitives."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StrictGate:
    """The non-negotiable pose and branch-continuity ACCEPT gate."""

    position_tolerance_m: float = 0.001
    orientation_tolerance_rad: float = np.deg2rad(0.5)
    branch_guard_rad: float = 0.30

    def __post_init__(self) -> None:
        values = (
            self.position_tolerance_m,
            self.orientation_tolerance_rad,
            self.branch_guard_rad,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("strict gate limits must be positive and finite")

    def accepts(
        self,
        position_error_m: float,
        orientation_error_rad: float,
        joint_delta_rad: np.ndarray,
    ) -> bool:
        delta = np.asarray(joint_delta_rad, dtype=float)
        if delta.ndim != 1:
            raise ValueError("joint delta must be one-dimensional")
        if (
            not np.isfinite(position_error_m)
            or not np.isfinite(orientation_error_rad)
            or np.any(~np.isfinite(delta))
        ):
            return False
        return bool(
            position_error_m <= self.position_tolerance_m
            and orientation_error_rad <= self.orientation_tolerance_rad
            and np.all(np.abs(delta) <= self.branch_guard_rad)
        )

    def pose_accepts(
        self,
        position_error_m: float,
        orientation_error_rad: float,
    ) -> bool:
        return self.accepts(
            position_error_m,
            orientation_error_rad,
            np.zeros(1, dtype=float),
        )


def shortest_joint_delta(
    new: np.ndarray,
    old: np.ndarray,
    periodic: np.ndarray,
) -> np.ndarray:
    """Return ``new-old`` while wrapping only declared periodic joints."""

    new = np.asarray(new, dtype=float)
    old = np.asarray(old, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    if new.ndim != 1 or new.shape != old.shape or periodic.shape != new.shape:
        raise ValueError("joint values and periodic mask must have matching 1-D shapes")
    delta = new - old
    delta = delta.copy()
    delta[periodic] = (delta[periodic] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def wrist_branch_signature(
    q: np.ndarray,
    *,
    deadband_rad: float = 0.10,
) -> tuple[int, int]:
    """Coarsely identify the PiperX J4/J5 basin without zero chatter."""

    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or np.any(~np.isfinite(q)):
        raise ValueError("PiperX wrist signature requires six finite joints")
    if not np.isfinite(deadband_rad) or deadband_rad < 0:
        raise ValueError("wrist deadband must be finite and non-negative")

    def classify(value: float) -> int:
        if abs(value) <= deadband_rad:
            return 0
        return 1 if value > 0 else -1

    return classify(float(q[3])), classify(float(q[4]))


def trapezoidal_transition_time(
    delta_rad: np.ndarray,
    *,
    velocity_rad_s: float = 1.0,
    acceleration_rad_s2: float = 4.0,
    settle_s: float = 0.1,
    decision_s: float = 0.015,
) -> float:
    """Time required by the slowest joint under a symmetric trapezoid."""

    delta = np.asarray(delta_rad, dtype=float)
    if delta.ndim != 1 or np.any(~np.isfinite(delta)):
        raise ValueError("joint delta must be a finite one-dimensional array")
    parameters = (velocity_rad_s, acceleration_rad_s2, settle_s, decision_s)
    if any(not np.isfinite(value) or value < 0 for value in parameters):
        raise ValueError("execution timing parameters must be finite and non-negative")
    if velocity_rad_s <= 0 or acceleration_rad_s2 <= 0:
        raise ValueError("velocity and acceleration must be positive")
    distance = float(np.max(np.abs(delta), initial=0.0))
    switch_distance = velocity_rad_s**2 / acceleration_rad_s2
    if distance <= switch_distance:
        motion_s = 2.0 * np.sqrt(distance / acceleration_rad_s2)
    else:
        motion_s = (
            2.0 * velocity_rad_s / acceleration_rad_s2
            + (distance - switch_distance) / velocity_rad_s
        )
    return float(motion_s + settle_s + decision_s)


def minimum_jerk_transition(
    q0: np.ndarray,
    q1: np.ndarray,
    frames: int,
) -> np.ndarray:
    """Return a quintic smoothstep whose last row is exactly ``q1``."""

    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if q0.ndim != 1 or q1.shape != q0.shape:
        raise ValueError("transition endpoints must have matching one-dimensional shapes")
    if np.any(~np.isfinite(q0)) or np.any(~np.isfinite(q1)):
        raise ValueError("transition endpoints must be finite")
    frames = int(frames)
    if frames < 2:
        raise ValueError("minimum-jerk transition needs at least two frames")
    unit = np.linspace(0.0, 1.0, frames)
    blend = 10.0 * unit**3 - 15.0 * unit**4 + 6.0 * unit**5
    result = q0[None, :] + blend[:, None] * (q1 - q0)[None, :]
    result[0] = q0
    result[-1] = q1
    return result
