from __future__ import annotations

import torch

from .topology import DesignBatch, TopologyTemplate


def _skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(v.shape[:-1] + (3, 3))


def axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation with broadcastable ``axis [...,3]`` and ``angle [...]``."""
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    K = _skew(axis)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(K.shape)
    s = torch.sin(angle)[..., None, None]
    c = torch.cos(angle)[..., None, None]
    return eye + s * K + (1.0 - c) * (K @ K)


def build_designs(
    template: TopologyTemplate,
    raw_scales: torch.Tensor,
    calibration_q: torch.Tensor,
) -> DesignBatch:
    """Replicate the audited native robot geometry for mount screening.

    The two legacy arguments remain in the call signature while old result
    readers are migrated, but neither is allowed to alter morphology. Candidate
    diversity belongs to the installation search, not to robot link scaling.
    """
    if raw_scales.ndim != 2 or raw_scales.shape[1] != template.dof:
        raise ValueError(f"raw_scales must have shape [design, {template.dof}]")
    if calibration_q.shape[-1] != template.dof:
        raise ValueError(f"calibration_q must end in {template.dof} joints")
    count = raw_scales.shape[0]
    deltas = template.deltas.to(
        device=raw_scales.device, dtype=raw_scales.dtype
    ).unsqueeze(0).expand(count, -1, -1).clone()
    native = torch.ones(count, device=raw_scales.device, dtype=raw_scales.dtype)
    return _assemble(template, deltas, native)


def _assemble(template: TopologyTemplate, deltas: torch.Tensor, scale: torch.Tensor) -> DesignBatch:
    design_count = deltas.shape[0]
    origin = (torch.zeros(3, dtype=deltas.dtype, device=deltas.device)
              if template.first_joint_origin_m is None else template.first_joint_origin_m)
    origin = origin[None] * scale[:, None]
    points = torch.cat((origin[:, None], origin[:, None] + torch.cumsum(deltas[:, :-1], dim=1)), dim=1)
    home = torch.eye(4, dtype=deltas.dtype, device=deltas.device).repeat(design_count, 1, 1)
    home[:, :3, :3] = template.home_rotation
    home[:, :3, 3] = origin + deltas.sum(dim=1)
    return DesignBatch(template, deltas, points, home, scale)


def assemble_from_deltas(template: TopologyTemplate, deltas: torch.Tensor) -> DesignBatch:
    """Construct a candidate from already verified topology deltas."""
    if deltas.ndim == 2:
        deltas = deltas[None]
    return _assemble(template, deltas, torch.ones(deltas.shape[0], dtype=deltas.dtype,
                                                  device=deltas.device))


def fk_flange(designs: DesignBatch, q: torch.Tensor) -> torch.Tensor:
    """Batched product-of-exponentials FK.

    Args:
        designs: D candidate designs.
        q: ``[D,N,dof]`` or ``[N,dof]`` (broadcast to all designs).
    Returns:
        Flange transforms ``[D,N,4,4]``.
    """
    if q.ndim == 2:
        q = q.unsqueeze(0).expand(designs.count, -1, -1)
    dof = designs.template.dof
    if q.ndim != 3 or q.shape[0] != designs.count or q.shape[-1] != dof:
        raise ValueError(f"q must have shape [N,{dof}] or [design,N,{dof}]")
    D, N = q.shape[:2]
    T = torch.eye(4, dtype=q.dtype, device=q.device).repeat(D, N, 1, 1)
    axes = designs.template.axes
    for index in range(dof):
        w = axes[index].view(1, 1, 3).expand(D, N, 3)
        p = designs.points[:, index].unsqueeze(1).expand(D, N, 3)
        R = axis_angle_matrix(w, q[..., index])
        translation = p - (R @ p.unsqueeze(-1)).squeeze(-1)
        A = torch.eye(4, dtype=q.dtype, device=q.device).repeat(D, N, 1, 1)
        A[..., :3, :3] = R
        A[..., :3, 3] = translation
        T = T @ A
    return T @ designs.home[:, None]


def fk_tcp(designs: DesignBatch, q: torch.Tensor) -> torch.Tensor:
    flange = fk_flange(designs, q)
    if designs.template.tool_transform is None:
        tool = torch.eye(4, dtype=flange.dtype, device=flange.device)
        tool[2, 3] = designs.template.tool_length_m
    else:
        tool = designs.template.tool_transform.to(dtype=flange.dtype, device=flange.device)
    return flange @ tool


def geometric_jacobian(designs: DesignBatch, q: torch.Tensor, *, tcp: bool = True) -> torch.Tensor:
    """World-aligned geometric Jacobian, shape ``[D,N,6,dof]``."""
    if q.ndim == 2:
        q = q.unsqueeze(0).expand(designs.count, -1, -1)
    end = (fk_tcp(designs, q) if tcp else fk_flange(designs, q))[..., :3, 3]
    D, N = q.shape[:2]
    T = torch.eye(4, dtype=q.dtype, device=q.device).repeat(D, N, 1, 1)
    columns = []
    for index in range(designs.template.dof):
        local_axis = designs.template.axes[index].view(1, 1, 3).expand(D, N, 3)
        local_point = designs.points[:, index].unsqueeze(1).expand(D, N, 3)
        world_axis = (T[..., :3, :3] @ local_axis.unsqueeze(-1)).squeeze(-1)
        world_point = (T[..., :3, :3] @ local_point.unsqueeze(-1)).squeeze(-1) + T[..., :3, 3]
        columns.append(torch.cat((torch.linalg.cross(world_axis, end - world_point), world_axis), dim=-1))
        R = axis_angle_matrix(local_axis, q[..., index])
        translation = local_point - (R @ local_point.unsqueeze(-1)).squeeze(-1)
        A = torch.eye(4, dtype=q.dtype, device=q.device).repeat(D, N, 1, 1)
        A[..., :3, :3] = R; A[..., :3, 3] = translation
        T = T @ A
    return torch.stack(columns, dim=-1)


def joint_world_positions(designs: DesignBatch, q: torch.Tensor, *, include_flange: bool = True) -> torch.Tensor:
    """World positions of all joint-axis reference points and flange.

    Returns ``[D,N,dof+1,3]`` when ``include_flange`` is true, otherwise
    ``[D,N,dof,3]``. These points define the centerline capsules used by the
    differentiable collision proxy.
    """
    if q.ndim == 2:
        q = q.unsqueeze(0).expand(designs.count, -1, -1)
    dof = designs.template.dof
    if q.ndim < 3 or q.shape[0] != designs.count or q.shape[-1] != dof:
        raise ValueError(f"q must have shape [N,{dof}] or [design,...,{dof}]")
    batch_shape = q.shape[1:-1]
    q = q.reshape(designs.count, -1, dof)
    D, N = q.shape[:2]
    transform = torch.eye(4, dtype=q.dtype, device=q.device).repeat(D, N, 1, 1)
    positions = []
    for index in range(dof):
        local_axis = designs.template.axes[index].view(1, 1, 3).expand(D, N, 3)
        local_point = designs.points[:, index].unsqueeze(1).expand(D, N, 3)
        world_point = (transform[..., :3, :3] @ local_point.unsqueeze(-1)).squeeze(-1) + transform[..., :3, 3]
        positions.append(world_point)
        rotation = axis_angle_matrix(local_axis, q[..., index])
        translation = local_point - (rotation @ local_point.unsqueeze(-1)).squeeze(-1)
        joint = torch.eye(4, dtype=q.dtype, device=q.device).repeat(D, N, 1, 1)
        joint[..., :3, :3] = rotation; joint[..., :3, 3] = translation
        transform = transform @ joint
    if include_flange:
        flange = fk_flange(designs, q)[..., :3, 3]
        positions.append(flange)
    return torch.stack(positions, dim=-2).reshape(D, *batch_shape, len(positions), 3)


def deterministic_joint_samples(template: TopologyTemplate, count: int, seed: int = 750) -> torch.Tensor:
    generator = torch.Generator(device=template.axes.device)
    generator.manual_seed(seed)
    u = torch.rand((count, template.dof), generator=generator, device=template.axes.device, dtype=template.axes.dtype)
    return template.q_min + u * (template.q_max - template.q_min)
