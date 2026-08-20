"""Deterministic joint-path retiming for fair dynamic comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetimedPath:
    time_s: torch.Tensor
    segment_dt_s: torch.Tensor
    speed_scale: torch.Tensor


def retime_joint_path(q: torch.Tensor, nominal_time_s: torch.Tensor,
                      velocity_limits_rad_s: torch.Tensor,
                      *, minimum_dt_s: float = 1e-4) -> RetimedPath:
    """Stretch time only, preserving every configuration and path branch.

    ``q`` is ``[T, J]`` and limits are scalar or ``[J]``.  Each segment gets
    the longer of its recorded duration and the exact duration required by
    the most restrictive joint.  This cleanly separates kinematic path
    quality from whether the human demonstration was recorded too quickly.
    """
    if q.ndim != 2 or nominal_time_s.ndim != 1 or len(q) != len(nominal_time_s):
        raise ValueError("expected q [T,J] and matching nominal_time_s [T]")
    if len(q) < 2:
        empty = nominal_time_s.new_empty((0,))
        return RetimedPath(nominal_time_s.clone(), empty, empty)
    limits = torch.as_tensor(velocity_limits_rad_s, device=q.device, dtype=q.dtype)
    if limits.ndim == 0:
        limits = limits.expand(q.shape[-1])
    if limits.shape != (q.shape[-1],) or torch.any(limits <= 0):
        raise ValueError("velocity limits must be positive scalar or [J]")
    recorded_dt = torch.diff(nominal_time_s).to(device=q.device, dtype=q.dtype)
    recorded_dt = recorded_dt.clamp_min(minimum_dt_s)
    required_dt = (torch.diff(q, dim=0).abs() / limits).amax(dim=-1)
    segment_dt = torch.maximum(recorded_dt, required_dt.clamp_min(minimum_dt_s))
    time = torch.cat((nominal_time_s[:1].to(q),
                      nominal_time_s[:1].to(q) + torch.cumsum(segment_dt, dim=0)))
    return RetimedPath(time, segment_dt, recorded_dt / segment_dt)
