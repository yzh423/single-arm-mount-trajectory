from __future__ import annotations

from dataclasses import dataclass

import torch

from .collision import (dual_capsule_clearance, self_capsule_clearance,
                        table_capsule_clearance)
from .ik import deterministic_seeds, solve_multistart
from .kinematics import joint_world_positions
from .taskspace import BimanualTaskFrames, world_to_base_population
from .topology import DesignBatch


@dataclass
class PopulationMetrics:
    loss: torch.Tensor
    success_rate: torch.Tensor
    position_rmse_m: torch.Tensor
    orientation_rmse_rad: torch.Tensor
    sigma_mean: torch.Tensor
    topology_error_m: torch.Tensor
    motor_clearance_violation_m: torch.Tensor
    minimum_collision_clearance_m: torch.Tensor
    collision_frame_fraction: torch.Tensor
    collision_margin_violation_m: torch.Tensor
    inward_elbow_fraction: torch.Tensor


def _line_distance(point_a, axis_a, point_b, axis_b) -> torch.Tensor:
    axis_a = axis_a.expand_as(point_a)
    axis_b = axis_b.expand_as(point_b)
    cross = torch.linalg.cross(axis_a, axis_b)
    denominator = torch.linalg.vector_norm(cross, dim=-1).clamp_min(1e-10)
    skew_distance = ((point_b - point_a) * cross).sum(dim=-1).abs() / denominator
    parallel_distance = torch.linalg.vector_norm(torch.linalg.cross(point_b - point_a, axis_a), dim=-1)
    return torch.where(denominator > 1e-7, skew_distance, parallel_distance)


def topology_intersection_error(designs: DesignBatch) -> torch.Tensor:
    """Maximum drift of the topology-defining wrist-axis intersections."""
    pairs = {
        "doosan": ((3, 4), (4, 5), (3, 5)),
        "xarm6": ((3, 4),),
        "ur5": ((3, 4), (4, 5)),
        "kinova": ((2, 3), (3, 4), (4, 5)),
    }[designs.template.name]
    axes = designs.template.axes
    errors = [_line_distance(designs.points[:, a], axes[a], designs.points[:, b], axes[b])
              for a, b in pairs]
    return torch.stack(errors, dim=-1).amax(dim=-1)


def motor_clearance_violation(designs: DesignBatch, minimum_m: float = 0.057) -> torch.Tensor:
    """Violation of topology-defining physical wrist/motor offsets.

    xArm/UR keep an absolute 57 mm motor-thickness floor. Kinova is allowed
    smaller offsets, but each pairwise-intersection segment must retain at
    least 45% of its vendor magnitude (same physical order of magnitude).
    """
    name = designs.template.name
    indices = {"doosan": (), "xarm6": (4,), "ur5": (3, 4, 5), "kinova": (2, 3, 4)}[name]
    if not indices:
        return torch.zeros(designs.count, dtype=designs.deltas.dtype, device=designs.deltas.device)
    lengths = torch.linalg.vector_norm(designs.deltas[:, indices, :], dim=-1)
    if name == "kinova":
        vendor = torch.linalg.vector_norm(designs.template.deltas[list(indices)], dim=-1)
        floor = 0.45 * vendor
    else:
        floor = torch.full_like(lengths, minimum_m)
    return torch.relu(floor - lengths).sum(dim=-1)


def evaluate_population(
    designs: DesignBatch,
    task: BimanualTaskFrames,
    *,
    left_base_xyz=(-0.36, 0.0, 0.0),
    right_base_xyz=(0.36, 0.0, 0.0),
    left_base_world: torch.Tensor | None = None,
    right_base_world: torch.Tensor | None = None,
    seed_count: int = 24,
    ik_iterations: int = 90,
    compute_singular_values: bool = False,
    collision_aware_branch_selection: bool = True,
    outer_elbow_branch_weight: float = 0.0,
) -> PopulationMetrics:
    device, dtype = designs.deltas.device, designs.deltas.dtype

    def translation_mount(xyz):
        xyz = torch.as_tensor(xyz, dtype=dtype, device=device)
        if xyz.ndim == 1:
            xyz = xyz.expand(designs.count, 3)
        transform = torch.eye(4, dtype=dtype, device=device).repeat(designs.count, 1, 1)
        transform[:, :3, 3] = xyz
        return transform

    left_mount = translation_mount(left_base_xyz) if left_base_world is None else left_base_world
    right_mount = translation_mount(right_base_xyz) if right_base_world is None else right_base_world
    targets = torch.cat((world_to_base_population(task.left, left_mount),
                         world_to_base_population(task.right, right_mount)), dim=1)
    seeds = deterministic_seeds(designs.template, seed_count)
    result = solve_multistart(designs, targets, seeds, iterations=ik_iterations,
                              compute_singular_values=compute_singular_values)
    frame_count = task.left.shape[0]
    left_result_q, right_result_q = result.q[:, :frame_count], result.q[:, frame_count:]
    left_node = (400.0 * result.position_error_m[:, :frame_count] +
                 8.0 * result.orientation_error_rad[:, :frame_count])
    right_node = (400.0 * result.position_error_m[:, frame_count:] +
                  8.0 * result.orientation_error_rad[:, frame_count:])
    def elbow_inward(q, mount, side):
        elbow = joint_world_positions(designs, q, include_flange=False)[..., 2, :]
        rotation = mount[:, :3, :3]
        translation = mount[:, :3, 3]
        world = torch.einsum("dij,d...j->d...i", rotation, elbow)
        world = world + translation.view(designs.count, *([1] * (world.ndim - 2)), 3)
        shoulder_x = translation[:, 0].view(designs.count, *([1] * (world.ndim - 2)))
        return world[..., 0] - shoulder_x if side == "left" else shoulder_x - world[..., 0]

    left_inward_all = elbow_inward(left_result_q, left_mount, "left")
    right_inward_all = elbow_inward(right_result_q, right_mount, "right")
    if outer_elbow_branch_weight > 0:
        left_node = left_node + outer_elbow_branch_weight * torch.relu(
            left_inward_all + .02).square()
        right_node = right_node + outer_elbow_branch_weight * torch.relu(
            right_inward_all + .02).square()
    left_self_all = self_capsule_clearance(designs, left_result_q)
    right_self_all = self_capsule_clearance(designs, right_result_q)
    left_table_all = table_capsule_clearance(designs, left_result_q, left_mount)
    right_table_all = table_capsule_clearance(designs, right_result_q, right_mount)
    dual_all = dual_capsule_clearance(
        designs, left_result_q[:, :, :, None, :], designs,
        right_result_q[:, :, None, :, :], left_mount, right_mount)
    pair_clearance = torch.minimum(
        torch.minimum(left_self_all[..., :, None], right_self_all[..., None, :]),
        torch.minimum(dual_all, torch.minimum(left_table_all[..., :, None],
                                              right_table_all[..., None, :])))
    # Select a collision-aware left/right branch pair at every sampled frame.
    # Safety margin is a soft Pareto objective; actual penetration receives a
    # much stronger cost so reach is not improved by choosing colliding IK.
    pair_cost = (left_node[..., :, None] + right_node[..., None, :] +
                 4000.0 * torch.relu(0.015 - pair_clearance).square() +
                 20000.0 * torch.relu(-pair_clearance))
    branch_count = result.q.shape[2]
    if collision_aware_branch_selection:
        best_pair = pair_cost.flatten(-2).argmin(dim=-1)
    else:
        # Explicit ablation/oracle: measure pure kinematic reach first, then
        # report collisions without letting the proxy alter IK branch choice.
        best_pair = left_node.argmin(dim=-1) * branch_count + right_node.argmin(dim=-1)
    left_branch, right_branch = best_pair // branch_count, best_pair % branch_count

    def gather_side(value, branch):
        return value.gather(2, branch[..., None]).squeeze(-1)

    left_position = gather_side(result.position_error_m[:, :frame_count], left_branch)
    right_position = gather_side(result.position_error_m[:, frame_count:], right_branch)
    left_orientation = gather_side(result.orientation_error_rad[:, :frame_count], left_branch)
    right_orientation = gather_side(result.orientation_error_rad[:, frame_count:], right_branch)
    left_success = gather_side(result.success[:, :frame_count], left_branch)
    right_success = gather_side(result.success[:, frame_count:], right_branch)
    left_sigma = gather_side(result.sigma_min[:, :frame_count], left_branch)
    right_sigma = gather_side(result.sigma_min[:, frame_count:], right_branch)
    left_inward = gather_side(left_inward_all, left_branch)
    right_inward = gather_side(right_inward_all, right_branch)
    position = torch.cat((left_position, right_position), dim=1)
    orientation = torch.cat((left_orientation, right_orientation), dim=1)
    success = torch.cat((left_success, right_success), dim=1)
    sigma = torch.cat((left_sigma, right_sigma), dim=1)
    dof = designs.template.dof
    left_q = left_result_q.gather(2, left_branch[..., None, None].expand(
        designs.count, frame_count, 1, dof)).squeeze(2)
    right_q = right_result_q.gather(2, right_branch[..., None, None].expand(
        designs.count, frame_count, 1, dof)).squeeze(2)
    left_self = self_capsule_clearance(designs, left_q)
    right_self = self_capsule_clearance(designs, right_q)
    dual = dual_capsule_clearance(designs, left_q, designs, right_q, left_mount, right_mount)
    left_table = table_capsule_clearance(designs, left_q, left_mount)
    right_table = table_capsule_clearance(designs, right_q, right_mount)
    clearance = torch.minimum(torch.minimum(left_self, right_self),
                              torch.minimum(dual, torch.minimum(left_table, right_table)))
    minimum_clearance = clearance.amin(dim=1)
    collision_fraction = (clearance < 0).to(dtype).mean(dim=1)
    collision_violation = torch.relu(0.015 - clearance).mean(dim=1)
    position_rmse = torch.sqrt(position.square().mean(dim=1))
    orientation_rmse = torch.sqrt(orientation.square().mean(dim=1))
    success_rate = success.to(dtype).mean(dim=1)
    sigma_mean = sigma.mean(dim=1)
    inward_elbow_fraction = torch.cat((left_inward, right_inward), dim=1).gt(0).to(dtype).mean(dim=1)
    topology_error = topology_intersection_error(designs)
    clearance_violation = motor_clearance_violation(designs)
    # Strong hard-like penalty: 0.1 mm topology drift already costs 1.0.
    topology_penalty = 1e4 * topology_error
    loss = (6.0 * (1.0 - success_rate) + 40.0 * position_rmse +
            2.0 * orientation_rmse + 0.01 / sigma_mean.clamp_min(1e-4) + topology_penalty +
            250.0 * clearance_violation + 8.0 * collision_fraction + 80.0 * collision_violation)
    return PopulationMetrics(loss, success_rate, position_rmse, orientation_rmse,
                             sigma_mean, topology_error, clearance_violation,
                             minimum_clearance, collision_fraction, collision_violation,
                             inward_elbow_fraction)
