from __future__ import annotations

from dataclasses import dataclass
import torch

from .kinematics import joint_world_positions
from .topology import DesignBatch


DEFAULT_LINK_RADII_M = (0.070, 0.065, 0.058, 0.052, 0.047, 0.042)


def _link_radii(radii, dof: int, *, dtype, device) -> torch.Tensor:
    values = torch.as_tensor(radii, dtype=dtype, device=device).flatten()
    if values.numel() < dof:
        values = torch.cat((values, values[-1].expand(dof - values.numel())))
    return values[:dof]


@dataclass
class CollisionMetrics:
    self_clearance_m: torch.Tensor
    dual_clearance_m: torch.Tensor | None
    table_clearance_m: torch.Tensor
    penalty: torch.Tensor


def segment_segment_distance(p0, p1, q0, q1, eps: float = 1e-12) -> torch.Tensor:
    """Exact batched closest distance between finite 3-D line segments."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a = (u * u).sum(-1); b = (u * v).sum(-1); c = (v * v).sum(-1)
    d = (u * w).sum(-1); e = (v * w).sum(-1)
    denominator = a * c - b * b
    parallel = denominator.abs() < eps
    s = torch.where(parallel, torch.zeros_like(denominator), (b * e - c * d) / denominator.clamp_min(eps))
    s = s.clamp(0.0, 1.0)
    t = ((b * s + e) / c.clamp_min(eps)).clamp(0.0, 1.0)
    # Reproject s after t clipping; this handles closest endpoint cases.
    s = ((b * t - d) / a.clamp_min(eps)).clamp(0.0, 1.0)
    closest = w + s[..., None] * u - t[..., None] * v
    return torch.linalg.vector_norm(closest, dim=-1)


def _segments(designs: DesignBatch, q: torch.Tensor):
    points = joint_world_positions(designs, q)
    return points[..., :-1, :], points[..., 1:, :]


def _active_segments(designs: DesignBatch, minimum_length_m: float = 0.015) -> torch.Tensor:
    return torch.linalg.vector_norm(designs.deltas, dim=-1) >= minimum_length_m


def self_segment_pair_distances(designs: DesignBatch, q: torch.Tensor, allowed_pairs=None):
    """Centerline distances for every topology-valid non-adjacent link pair."""
    starts, ends = _segments(designs, q); active = _active_segments(designs)
    # Pairs listed here are fixed topology intersections/near-intersections,
    # analogous to adjacent-link exclusions in the source MJCF.  They are not
    # configuration-dependent self collisions.  UR's (2, 4) capsule pair has
    # a constant -2.5 mm overlap across both arms; (2, 5) remains active.
    allowed = ({"doosan": {(3, 5)}, "xarm6": set(), "ur5": {(2, 4)},
                "kinova": {(2, 4), (3, 5)}}.get(designs.template.name, set())
               if allowed_pairs is None else set(map(tuple, allowed_pairs)))
    values, pairs = [], []
    for first in range(designs.template.dof):
        for second in range(first + 2, designs.template.dof):
            if (first, second) in allowed: continue
            distance = segment_segment_distance(starts[..., first, :], ends[..., first, :],
                                                starts[..., second, :], ends[..., second, :])
            active_between = active[:, first + 1:second].any(dim=-1)
            valid = active[:, first] & active[:, second] & active_between
            valid = valid.view(valid.shape[0], *([1] * (distance.ndim - 1)))
            values.append(torch.where(valid, distance, torch.full_like(distance, torch.inf)))
            pairs.append((first, second))
    return torch.stack(values, dim=-1), tuple(pairs)


def self_capsule_clearance(designs: DesignBatch, q: torch.Tensor,
                           radii=DEFAULT_LINK_RADII_M, allowed_pairs=None) -> torch.Tensor:
    distances, pairs = self_segment_pair_distances(designs, q, allowed_pairs=allowed_pairs)
    radii = _link_radii(radii, designs.template.dof, dtype=q.dtype, device=q.device)
    thresholds = torch.stack([radii[first] + radii[second] for first, second in pairs])
    return (distances - thresholds).amin(dim=-1)


def dual_capsule_clearance(left_designs: DesignBatch, left_q: torch.Tensor,
                           right_designs: DesignBatch, right_q: torch.Tensor,
                           left_base_xyz, right_base_xyz,
                           radii=DEFAULT_LINK_RADII_M) -> torch.Tensor:
    left_start, left_end = _segments(left_designs, left_q)
    right_start, right_end = _segments(right_designs, right_q)
    left_active, right_active = _active_segments(left_designs), _active_segments(right_designs)
    def transform(points, base, count):
        base = torch.as_tensor(base, dtype=points.dtype, device=points.device)
        if base.shape[-2:] == (4, 4):
            if base.ndim == 2: base = base[None].expand(count, 4, 4)
            rotated = torch.einsum("dij,d...j->d...i", base[:, :3, :3], points)
            translation = base[:, :3, 3].view(count, *([1] * (points.ndim - 2)), 3)
            return rotated + translation
        if base.ndim == 1: base = base[None].expand(count, 3)
        translation = base.view(count, *([1] * (points.ndim - 2)), 3)
        return points + translation
    left_start = transform(left_start, left_base_xyz, left_designs.count)
    left_end = transform(left_end, left_base_xyz, left_designs.count)
    right_start = transform(right_start, right_base_xyz, right_designs.count)
    right_end = transform(right_end, right_base_xyz, right_designs.count)
    left_radii = _link_radii(radii, left_designs.template.dof,
                             dtype=left_q.dtype, device=left_q.device)
    right_radii = _link_radii(radii, right_designs.template.dof,
                              dtype=right_q.dtype, device=right_q.device)
    values = []
    for first in range(left_designs.template.dof):
        for second in range(right_designs.template.dof):
            distance = segment_segment_distance(left_start[..., first, :], left_end[..., first, :],
                                                right_start[..., second, :], right_end[..., second, :])
            valid = left_active[:, first] & right_active[:, second]
            valid = valid.view(valid.shape[0], *([1] * (distance.ndim - 1)))
            distance = torch.where(valid, distance,
                                   torch.full_like(distance, torch.inf))
            values.append(distance - left_radii[first] - right_radii[second])
    return torch.stack(values, dim=-1).amin(dim=-1)


def table_capsule_clearance(designs: DesignBatch, q: torch.Tensor, base,
                            table_height_m: float = 0.0,
                            radii=DEFAULT_LINK_RADII_M) -> torch.Tensor:
    starts, ends = _segments(designs, q)
    active = _active_segments(designs).clone()
    # Link 0 emerges from the mounting flange and is allowed to intersect the
    # support plane; treating it as an arm/table collision makes every valid
    # flat or side-mounted installation infeasible by construction.
    active[:, 0] = False
    base = torch.as_tensor(base, dtype=q.dtype, device=q.device)
    if base.shape[-2:] == (4, 4):
        if base.ndim == 2: base = base[None].expand(designs.count, 4, 4)
        translation = base[:, :3, 3].view(designs.count, *([1] * (starts.ndim - 2)), 3)
        starts = torch.einsum("dij,d...j->d...i", base[:, :3, :3], starts) + translation
        ends = torch.einsum("dij,d...j->d...i", base[:, :3, :3], ends) + translation
    else:
        base_z = base[..., 2] if base.ndim > 0 else base
        if base_z.ndim == 0: base_z = base_z.expand(designs.count)
        starts = starts.clone(); ends = ends.clone()
        starts[..., 2] += base_z[:, None, None]; ends[..., 2] += base_z[:, None, None]
    radii = _link_radii(radii, designs.template.dof, dtype=q.dtype, device=q.device)
    lowest = torch.minimum(starts[..., 2], ends[..., 2]) - radii
    active_view = active.view(active.shape[0], *([1] * (lowest.ndim - 2)), active.shape[1])
    lowest = torch.where(active_view, lowest, torch.full_like(lowest, torch.inf))
    return lowest.amin(dim=-1) - table_height_m


def collision_penalty(clearance: torch.Tensor, margin_m: float = 0.015) -> torch.Tensor:
    return torch.relu(margin_m - clearance).square()
