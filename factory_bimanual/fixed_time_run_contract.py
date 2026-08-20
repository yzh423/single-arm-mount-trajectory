"""Contracts shared by source-timestamp PiperX production runs."""
from __future__ import annotations

import numpy as np


def source_time_schedule(source_time_s):
    """Return an immutable relative source timeline and its row intervals."""
    source = np.asarray(source_time_s, dtype=float)
    if (source.ndim != 1 or len(source) < 2 or
            np.any(~np.isfinite(source)) or np.any(np.diff(source) <= 0.0)):
        raise ValueError("source_time_s must contain at least two increasing finite values")
    relative = source - source[0]
    delta = np.diff(source)
    intervals = np.r_[delta[0], delta]
    return relative, intervals


def validate_synchronized_mount(mount, minimum_separation_m=0.60):
    """Validate one upright shared-height paired mount and return separation."""
    if not np.isfinite(minimum_separation_m) or minimum_separation_m <= 0.0:
        raise ValueError("minimum separation must be positive and finite")
    try:
        left = np.asarray(mount["xy"]["left"], dtype=float)
        right = np.asarray(mount["xy"]["right"], dtype=float)
        yaw = np.asarray([mount["yaw"][side] for side in ("left", "right")],
                         dtype=float)
        shared_z = float(mount["shared_base_z_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("mount must define paired xy/yaw and shared base height") from exc
    if (left.shape != (2,) or right.shape != (2,) or
            np.any(~np.isfinite(np.r_[left, right, yaw, shared_z]))):
        raise ValueError("mount coordinates must be finite")
    for field in ("roll", "pitch"):
        values = mount.get(field, {"left": 0.0, "right": 0.0})
        if any(abs(float(values.get(side, 0.0))) > 1e-12
               for side in ("left", "right")):
            raise ValueError("synchronized mounts must remain upright")
    separation = float(np.linalg.norm(right - left))
    if separation + 1e-12 < minimum_separation_m:
        raise ValueError(
            f"base separation {separation:.6f} m is below "
            f"{minimum_separation_m:.6f} m")
    return separation


__all__ = ["source_time_schedule", "validate_synchronized_mount"]
