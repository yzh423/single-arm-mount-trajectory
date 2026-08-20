from __future__ import annotations

from collections.abc import Callable

from dataclasses import dataclass
import torch

from .kinematics import fk_tcp, geometric_jacobian
from .topology import DesignBatch


@dataclass
class IKResult:
    q: torch.Tensor
    position_error_m: torch.Tensor
    orientation_error_rad: torch.Tensor
    sigma_min: torch.Tensor
    success: torch.Tensor


@dataclass
class SelectedIKPath:
    q: torch.Tensor
    branch_index: torch.Tensor
    position_error_m: torch.Tensor
    orientation_error_rad: torch.Tensor
    sigma_min: torch.Tensor
    success: torch.Tensor
    total_cost: torch.Tensor


def reverse_ik_time(result: IKResult) -> IKResult:
    """Restore chronological order after solving a trajectory backwards."""
    fields = ("q", "position_error_m", "orientation_error_rad", "sigma_min", "success")
    return IKResult(*(torch.flip(getattr(result, field), dims=(1,)) for field in fields))


def concatenate_ik_branches(*results: IKResult) -> IKResult:
    """Concatenate compatible IK candidate sets along their branch axis."""
    if not results:
        raise ValueError("at least one IKResult is required")
    reference = results[0].q.shape[:2]
    if any(result.q.shape[:2] != reference for result in results[1:]):
        raise ValueError("IK results must have matching design/time dimensions")
    fields = ("q", "position_error_m", "orientation_error_rad", "sigma_min", "success")
    return IKResult(*(torch.cat([getattr(result, field) for result in results], dim=2)
                      for field in fields))


def rotation_log(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cosine)
    vee = torch.stack((rotation[..., 2, 1] - rotation[..., 1, 2],
                       rotation[..., 0, 2] - rotation[..., 2, 0],
                       rotation[..., 1, 0] - rotation[..., 0, 1]), dim=-1)
    scale = angle / (2.0 * torch.sin(angle).clamp_min(1e-7))
    small = angle < 1e-4
    return torch.where(small[..., None], 0.5 * vee, scale[..., None] * vee)


def pose_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    position = target[..., :3, 3] - current[..., :3, 3]
    rotation = target[..., :3, :3] @ current[..., :3, :3].transpose(-1, -2)
    return torch.cat((position, rotation_log(rotation)), dim=-1)


def adaptive_damping(
    sigma_min: torch.Tensor,
    *,
    minimum: float = 0.005,
    maximum: float = 0.08,
    safe_sigma: float = 0.10,
) -> torch.Tensor:
    """Smoothly increase DLS damping as the Jacobian approaches singularity."""
    proximity = (1.0 - sigma_min / safe_sigma).clamp(0.0, 1.0)
    return minimum + (maximum - minimum) * proximity.square()


def joint_limit_centering_velocity(
    q: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    activation: float = 0.70,
) -> torch.Tensor:
    """Return a smooth center-seeking velocity active near joint limits."""
    center = 0.5 * (lower + upper)
    half_span = (0.5 * (upper - lower)).clamp_min(1e-6)
    normalized = (q - center) / half_span
    magnitude = torch.relu(normalized.abs() - activation) / max(1.0 - activation, 1e-6)
    return -normalized.sign() * magnitude.square()


def solve_multistart(
    designs: DesignBatch,
    targets: torch.Tensor,
    seeds: torch.Tensor,
    *,
    iterations: int = 60,
    damping: float = 0.025,
    step_limit: float = 0.18,
    position_tolerance_m: float = 0.0025,
    orientation_tolerance_rad: float = 0.0261799388,
    compute_singular_values: bool = True,
    adaptive_damping_enabled: bool = True,
    damping_minimum: float = 0.005,
    damping_maximum: float = 0.08,
    damping_safe_sigma: float = 0.10,
    joint_centering_weight: float = 0.02,
) -> IKResult:
    """GPU DLS IK for D designs, T targets and K independent seeds.

    ``targets`` is [T,4,4], and ``seeds`` is [K,dof] or [D,T,K,dof].
    The returned tensors retain all K branches; callers can select branches by
    feasibility, continuity, manipulability or collision clearance.
    """
    D, dof = designs.count, designs.template.dof
    if targets.ndim == 3:
        T = targets.shape[0]
        target = targets.view(1, T, 1, 4, 4).expand(D, T, -1, -1, -1)
    elif targets.ndim == 4 and targets.shape[0] == D:
        T = targets.shape[1]
        target = targets[:, :, None]
    else:
        raise ValueError("targets must have shape [T,4,4] or [D,T,4,4]")
    if seeds.ndim == 2:
        K = seeds.shape[0]
        q = seeds.view(1, 1, K, dof).expand(D, T, K, dof).clone()
    else:
        q = seeds.clone(); K = q.shape[2]
    target = target.expand(D, T, K, 4, 4)
    lo = designs.template.q_min.view(1, 1, 1, dof)
    hi = designs.template.q_max.view(1, 1, 1, dof)
    q = q.clamp(lo, hi)
    identity = torch.eye(6, dtype=q.dtype, device=q.device)
    for _ in range(iterations):
        flat_q = q.reshape(D, T * K, dof)
        current = fk_tcp(designs, flat_q).reshape(D, T, K, 4, 4)
        error = pose_error(current, target)
        jacobian = geometric_jacobian(designs, flat_q).reshape(D, T, K, 6, dof)
        if adaptive_damping_enabled:
            sigma = torch.linalg.svdvals(jacobian)[..., -1]
            damping_now = adaptive_damping(
                sigma,
                minimum=damping_minimum,
                maximum=damping_maximum,
                safe_sigma=damping_safe_sigma,
            )
        else:
            damping_now = torch.full_like(error[..., 0], damping)
        system = (jacobian @ jacobian.transpose(-1, -2) +
                  damping_now.square()[..., None, None] * identity)
        damped_inverse = jacobian.transpose(-1, -2) @ torch.linalg.solve(system, identity)
        dq = (damped_inverse @ error.unsqueeze(-1)).squeeze(-1)
        if joint_centering_weight > 0.0:
            centering = joint_limit_centering_velocity(q, lo, hi)
            joint_identity = torch.eye(dof, dtype=q.dtype, device=q.device)
            nullspace = joint_identity - damped_inverse @ jacobian
            dq = dq + joint_centering_weight * (nullspace @ centering.unsqueeze(-1)).squeeze(-1)
        norm = torch.linalg.vector_norm(dq, dim=-1, keepdim=True).clamp_min(1e-12)
        dq = dq * torch.clamp(step_limit / norm, max=1.0)
        active = ((torch.linalg.vector_norm(error[..., :3], dim=-1) > position_tolerance_m) |
                  (torch.linalg.vector_norm(error[..., 3:], dim=-1) > orientation_tolerance_rad))
        q = torch.where(active[..., None], (q + dq).clamp(lo, hi), q)
    flat_q = q.reshape(D, T * K, dof)
    current = fk_tcp(designs, flat_q).reshape(D, T, K, 4, 4)
    error = pose_error(current, target)
    jacobian = geometric_jacobian(designs, flat_q).reshape(D, T, K, 6, dof)
    singular = (torch.linalg.svdvals(jacobian)[..., -1] if compute_singular_values
                else torch.ones_like(error[..., 0]))
    pe = torch.linalg.vector_norm(error[..., :3], dim=-1)
    oe = torch.linalg.vector_norm(error[..., 3:], dim=-1)
    return IKResult(q, pe, oe, singular, (pe <= position_tolerance_m) & (oe <= orientation_tolerance_rad))


def deterministic_seeds(template, count: int, seed: int = 20260806) -> torch.Tensor:
    generator = torch.Generator(device=template.axes.device); generator.manual_seed(seed)
    u = torch.rand((count, template.dof), generator=generator, device=template.axes.device, dtype=template.axes.dtype)
    seeds = template.q_min + u * (template.q_max - template.q_min)
    seeds[0] = 0.5 * (template.q_min + template.q_max)
    return seeds


def solve_trajectory_multistart(
    designs: DesignBatch,
    targets: torch.Tensor,
    *,
    seed_count: int = 16,
    initial_iterations: int = 100,
    tracking_iterations: int = 35,
    restart_interval: int = 15,
    restart_fraction: float = 0.25,
    maximum_joint_delta_rad: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    seed: int = 20260806,
) -> IKResult:
    """Propagate IK branches through a trajectory with periodic rescue seeds.

    This is materially different from solving every frame from the same global
    seeds: each branch is warm-started from its own previous configuration,
    giving the downstream beam search temporally coherent candidates.
    """
    if targets.ndim != 3:
        raise ValueError("trajectory targets must have shape [T,4,4]")
    global_seeds = deterministic_seeds(designs.template, seed_count, seed)
    pieces = []
    previous_q, previous_cost = None, None
    for time_index in range(targets.shape[0]):
        if previous_q is None:
            seeds = global_seeds
            iterations = initial_iterations
        else:
            seeds = previous_q.clone()
            iterations = tracking_iterations
            if restart_interval > 0 and time_index % restart_interval == 0:
                replace_count = max(1, int(seed_count * restart_fraction))
                worst = torch.topk(previous_cost, replace_count, largest=True).indices
                seeds[worst] = global_seeds[:replace_count]
        result = solve_multistart(designs, targets[time_index:time_index + 1], seeds,
                                  iterations=iterations)
        if previous_q is not None:
            q_now = result.q[0, 0]
            offsets = torch.tensor((-2 * torch.pi, 0.0, 2 * torch.pi),
                                   dtype=q_now.dtype, device=q_now.device)
            candidates = q_now[..., None] + offsets
            lo = designs.template.q_min[None, :, None]
            hi = designs.template.q_max[None, :, None]
            valid = (candidates >= lo) & (candidates <= hi)
            distance = torch.where(valid, (candidates - previous_q[..., None]).abs(),
                                   torch.full_like(candidates, torch.inf))
            choice = distance.argmin(dim=-1, keepdim=True)
            unwrapped = candidates.gather(-1, choice).squeeze(-1)
            result.q[0, 0] = unwrapped
            if maximum_joint_delta_rad is not None:
                bounded = previous_q + (unwrapped - previous_q).clamp(
                    -maximum_joint_delta_rad, maximum_joint_delta_rad)
                result.q[0, 0] = bounded
                current = fk_tcp(designs, bounded)[0]
                error = pose_error(current, targets[time_index].expand_as(current))
                jacobian = geometric_jacobian(designs, bounded)[0]
                pe = torch.linalg.vector_norm(error[..., :3], dim=-1)
                oe = torch.linalg.vector_norm(error[..., 3:], dim=-1)
                result.position_error_m[0, 0] = pe
                result.orientation_error_rad[0, 0] = oe
                result.sigma_min[0, 0] = torch.linalg.svdvals(jacobian)[..., -1]
                result.success[0, 0] = ((pe <= .0025) & (oe <= .0261799388))
        pieces.append(result)
        previous_q = result.q[0, 0].detach()
        previous_cost = (400 * result.position_error_m[0, 0] +
                         8 * result.orientation_error_rad[0, 0]).detach()
        if progress_callback is not None:
            progress_callback(time_index + 1, targets.shape[0])

    def concatenate(field: str):
        return torch.cat([getattr(piece, field) for piece in pieces], dim=1)

    return IKResult(concatenate("q"), concatenate("position_error_m"),
                    concatenate("orientation_error_rad"), concatenate("sigma_min"),
                    concatenate("success"))


def select_continuous_branches(
    result: IKResult,
    designs: DesignBatch,
    *,
    position_weight: float = 400.0,
    orientation_weight: float = 8.0,
    continuity_weight: float = 8.0,
    singularity_weight: float = 0.015,
    joint_margin_weight: float = 0.02,
    failure_cost: float = 50.0,
    maximum_joint_step_rad: float | None = None,
) -> SelectedIKPath:
    """Dynamic-programming selection of one smooth IK branch per timestep.

    Unlike choosing the lowest instantaneous pose error, this retains an arm
    posture through wrist/elbow branch crossings and penalizes joint limits and
    low Jacobian singular values. Inputs have shape ``[D,T,K,...]``.
    """
    q = result.q
    D, T, K = q.shape[:3]
    span = (designs.template.q_max - designs.template.q_min).clamp_min(1e-6)
    center = 0.5 * (designs.template.q_min + designs.template.q_max)
    dof = designs.template.dof
    normalized = (q - center.view(1, 1, 1, dof)) / (0.5 * span).view(1, 1, 1, dof)
    limit_cost = torch.relu(normalized.abs() - 0.72).square().sum(dim=-1)
    node = (position_weight * result.position_error_m +
            orientation_weight * result.orientation_error_rad +
            singularity_weight / result.sigma_min.clamp_min(1e-4) +
            joint_margin_weight * limit_cost +
            (~result.success).to(q.dtype) * failure_cost)
    accumulated = node[:, 0]
    parents = []
    joint_scale = span.view(1, 1, 1, dof)
    for time_index in range(1, T):
        delta = (q[:, time_index, :, None, :] - q[:, time_index - 1, None, :, :]) / joint_scale
        transition = continuity_weight * delta.square().sum(dim=-1)
        if maximum_joint_step_rad is not None:
            absolute_step = (
                q[:, time_index, :, None, :] - q[:, time_index - 1, None, :, :]
            ).abs().amax(dim=-1)
            transition = torch.where(
                absolute_step <= maximum_joint_step_rad,
                transition,
                torch.full_like(transition, torch.inf),
            )
        candidate = accumulated[:, None, :] + transition
        best, parent = candidate.min(dim=-1)
        accumulated = node[:, time_index] + best
        parents.append(parent)
    final_cost, current = accumulated.min(dim=-1)
    indices = [current]
    for parent in reversed(parents):
        current = parent.gather(1, current[:, None]).squeeze(1)
        indices.append(current)
    branch = torch.stack(list(reversed(indices)), dim=1)
    gather_q = branch[:, :, None, None].expand(D, T, 1, dof)
    selected_q = q.gather(2, gather_q).squeeze(2)

    def gather_scalar(value: torch.Tensor) -> torch.Tensor:
        return value.gather(2, branch[:, :, None]).squeeze(2)

    return SelectedIKPath(selected_q, branch, gather_scalar(result.position_error_m),
                          gather_scalar(result.orientation_error_rad), gather_scalar(result.sigma_min),
                          gather_scalar(result.success), final_cost)
