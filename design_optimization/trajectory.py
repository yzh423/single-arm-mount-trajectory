from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .collision import (collision_penalty, dual_capsule_clearance,
                        self_capsule_clearance, table_capsule_clearance)
from .ik import IKResult
from .kinematics import joint_world_positions
from .topology import DesignBatch


@dataclass
class BimanualTrajectory:
    left_q: torch.Tensor
    right_q: torch.Tensor
    left_branch: torch.Tensor
    right_branch: torch.Tensor
    left_position_error_m: torch.Tensor
    right_position_error_m: torch.Tensor
    left_orientation_error_rad: torch.Tensor
    right_orientation_error_rad: torch.Tensor
    self_clearance_m: torch.Tensor
    dual_clearance_m: torch.Tensor
    table_clearance_m: torch.Tensor
    total_cost: torch.Tensor

    def metrics(self, dt: float = 1 / 30) -> dict[str, float]:
        position = torch.cat((self.left_position_error_m, self.right_position_error_m))
        orientation = torch.cat((self.left_orientation_error_rad, self.right_orientation_error_rad))
        dq = torch.cat((torch.diff(self.left_q, dim=0), torch.diff(self.right_q, dim=0)), dim=0)
        clearance = torch.minimum(torch.minimum(self.self_clearance_m, self.dual_clearance_m),
                                  self.table_clearance_m)
        return {
            "position_rmse_mm": float(torch.sqrt(position.square().mean()) * 1000),
            "orientation_rmse_deg": float(torch.rad2deg(torch.sqrt(orientation.square().mean()))),
            "position_median_mm": float(position.median() * 1000),
            "position_p95_mm": float(torch.quantile(position, .95) * 1000),
            "position_maximum_mm": float(position.max() * 1000),
            "orientation_median_deg": float(torch.rad2deg(orientation.median())),
            "orientation_p95_deg": float(torch.rad2deg(torch.quantile(orientation, .95))),
            "orientation_maximum_deg": float(torch.rad2deg(orientation.max())),
            "success_rate": float(((position <= .0025) & (orientation <= torch.deg2rad(
                torch.tensor(1.5, device=orientation.device)))).to(position.dtype).mean()),
            "minimum_clearance_mm": float(clearance.min() * 1000),
            "collision_frame_fraction": float((clearance < 0).to(position.dtype).mean()),
            "joint_path_length_rad": float(torch.linalg.vector_norm(dq, dim=-1).sum()),
            "maximum_joint_speed_rad_s": float(dq.abs().max() / dt),
            "maximum_joint_velocity_norm_rad_s": float(
                torch.linalg.vector_norm(dq, dim=-1).max() / dt),
            "joint_jump_fraction": float((torch.linalg.vector_norm(dq, dim=-1) > 0.5).to(
                position.dtype).mean()),
        }


def outer_elbow_penalty(design: DesignBatch, q: torch.Tensor, base,
                        *, side: str, margin_m: float = 0.02) -> torch.Tensor:
    """Geometric inward-elbow penalty independent of any joint-angle sign.

    The elbow is represented by the J3 axis point. For the left arm, crossing
    inward means moving toward larger world X relative to its shoulder; the
    right arm is mirrored. This remains meaningful across different 6R angle
    conventions, unlike a hard ``J3 > 0`` rule.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    points = joint_world_positions(design, q, include_flange=False)[0]
    elbow = points[..., 2, :]
    base = torch.as_tensor(base, dtype=q.dtype, device=q.device)
    if base.shape[-2:] == (4, 4):
        elbow = (base[:3, :3] @ elbow[..., None]).squeeze(-1) + base[:3, 3]
        shoulder_x = base[0, 3]
    else:
        shoulder_x = base[0]
        elbow = elbow + base
    inward = elbow[..., 0] - shoulder_x if side == "left" else shoulder_x - elbow[..., 0]
    return torch.relu(inward + margin_m).square()


def select_bimanual_collision_aware(
    left: IKResult,
    right: IKResult,
    design: DesignBatch,
    left_base_xyz,
    right_base_xyz,
    *,
    beam_width: int = 24,
    continuity_weight: float = 50.0,
    maximum_joint_step_rad: float = 0.5,
    joint_step_barrier_weight: float = 10000.0,
    collision_weight: float = 4000.0,
    collision_margin_m: float = 0.015,
    hard_collision: bool = False,
    hard_minimum_clearance_m: float = 0.0,
    outer_elbow_weight: float = 0.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BimanualTrajectory:
    """Beam search over paired left/right IK branches for one design.

    Pose accuracy, branch continuity, singularity and three collision modes
    are evaluated jointly. This prevents independently selected arm paths from
    becoming mutually inconsistent in the shared workspace.
    """
    if design.count != 1:
        raise ValueError("trajectory branch search currently accepts one finalist at a time")
    lq, rq = left.q[0], right.q[0]
    T, K = lq.shape[:2]; device, dtype = lq.device, lq.dtype
    dof = design.template.dof
    span = design.template.q_max - design.template.q_min

    def branch_node(result: IKResult):
        return (400.0 * result.position_error_m[0] + 8.0 * result.orientation_error_rad[0] +
                0.015 / result.sigma_min[0].clamp_min(1e-4) +
                (~result.success[0]).to(dtype) * 50.0)

    left_node, right_node = branch_node(left), branch_node(right)
    if outer_elbow_weight > 0:
        left_node = left_node + outer_elbow_weight * outer_elbow_penalty(
            design, lq.reshape(T * K, dof), left_base_xyz, side="left").reshape(T, K)
        right_node = right_node + outer_elbow_weight * outer_elbow_penalty(
            design, rq.reshape(T * K, dof), right_base_xyz, side="right").reshape(T, K)
    left_self = self_capsule_clearance(design, lq.reshape(T * K, dof))[0].reshape(T, K)
    right_self = self_capsule_clearance(design, rq.reshape(T * K, dof))[0].reshape(T, K)
    # Preserve the complete mount transform.  Passing ``base[2]`` here selects
    # the third matrix row, causing table_capsule_clearance to interpret a
    # rotation entry as base height (and to report spurious deep penetration).
    left_table = table_capsule_clearance(
        design, lq.reshape(T * K, dof), left_base_xyz)[0].reshape(T, K)
    right_table = table_capsule_clearance(
        design, rq.reshape(T * K, dof), right_base_xyz)[0].reshape(T, K)
    pair_ids = torch.arange(K * K, device=device)
    pair_left, pair_right = pair_ids // K, pair_ids % K
    saved_pairs, saved_parents = [], []
    beam_cost = None
    if progress_callback is not None:
        progress_callback(0, T)
    for time_index in range(T):
        pair_lq, pair_rq = lq[time_index, pair_left], rq[time_index, pair_right]
        dual = dual_capsule_clearance(design, pair_lq, design, pair_rq,
                                      left_base_xyz, right_base_xyz)[0]
        self_clearance = torch.minimum(left_self[time_index, pair_left], right_self[time_index, pair_right])
        table_clearance = torch.minimum(left_table[time_index, pair_left], right_table[time_index, pair_right])
        collision = (collision_penalty(self_clearance, collision_margin_m) +
                     collision_penalty(dual, collision_margin_m) +
                     collision_penalty(table_clearance, collision_margin_m))
        node = left_node[time_index, pair_left] + right_node[time_index, pair_right] + collision_weight * collision
        if hard_collision:
            collision_free = (torch.minimum(torch.minimum(self_clearance, dual),
                                             table_clearance) >= hard_minimum_clearance_m)
            node = torch.where(collision_free, node, torch.full_like(node, torch.inf))
        if beam_cost is None:
            candidate_cost = node
            parent = torch.zeros_like(pair_ids)
        else:
            previous_pairs = saved_pairs[-1]
            previous_left, previous_right = previous_pairs // K, previous_pairs % K
            dl = (pair_lq[:, None] - lq[time_index - 1, previous_left][None]) / span
            dr = (pair_rq[:, None] - rq[time_index - 1, previous_right][None]) / span
            normalized_motion = dl.square().sum(-1) + dr.square().sum(-1)
            absolute_step = (torch.linalg.vector_norm(pair_lq[:, None] -
                                                       lq[time_index - 1, previous_left][None], dim=-1) +
                             torch.linalg.vector_norm(pair_rq[:, None] -
                                                       rq[time_index - 1, previous_right][None], dim=-1))
            step_barrier = joint_step_barrier_weight * torch.relu(
                absolute_step - maximum_joint_step_rad).square()
            transition = continuity_weight * normalized_motion + step_barrier
            per_joint_step = torch.maximum(
                (pair_lq[:, None] - lq[time_index - 1, previous_left][None]).abs().amax(dim=-1),
                (pair_rq[:, None] - rq[time_index - 1, previous_right][None]).abs().amax(dim=-1))
            transition = torch.where(per_joint_step <= maximum_joint_step_rad + 1e-6, transition,
                                     torch.full_like(transition, torch.inf))
            all_cost = node[:, None] + transition + beam_cost[None]
            candidate_cost, parent = all_cost.min(dim=1)
        # Never admit an infeasible (+inf) or numerically invalid branch merely
        # to fill the requested beam width.  Doing so silently breaks a hard
        # velocity constraint during parent backtracking when fewer than
        # ``beam_width`` transitions remain feasible.
        feasible = torch.isfinite(candidate_cost)
        feasible_ids = torch.nonzero(feasible, as_tuple=False).squeeze(-1)
        if feasible_ids.numel() == 0:
            detail = ""
            if beam_cost is not None:
                detail = (f"; minimum per-joint step={float(per_joint_step.min()):.6f}, "
                          f"finite nodes={int(torch.isfinite(node).sum())}")
            raise RuntimeError(f"no feasible bimanual branch transition at frame {time_index}{detail}")
        width = min(beam_width, feasible_ids.numel())
        beam_cost, within_feasible = torch.topk(candidate_cost[feasible_ids], width,
                                                largest=False)
        selected = feasible_ids[within_feasible]
        saved_pairs.append(pair_ids[selected]); saved_parents.append(parent[selected])
        if progress_callback is not None:
            progress_callback(time_index + 1, T)
    beam_index = torch.argmin(beam_cost)
    chosen = []
    for time_index in range(T - 1, -1, -1):
        chosen.append(saved_pairs[time_index][beam_index])
        if time_index: beam_index = saved_parents[time_index][beam_index]
    chosen = torch.stack(list(reversed(chosen)))
    li, ri = chosen // K, chosen % K
    time = torch.arange(T, device=device)
    selected_lq, selected_rq = lq[time, li], rq[time, ri]
    if T > 1 and maximum_joint_step_rad is not None:
        largest_selected_step = torch.maximum(
            torch.diff(selected_lq, dim=0).abs().amax(),
            torch.diff(selected_rq, dim=0).abs().amax(),
        )
        if largest_selected_step > maximum_joint_step_rad + 1e-5:
            left_steps = torch.diff(selected_lq, dim=0).abs().amax(dim=-1)
            right_steps = torch.diff(selected_rq, dim=0).abs().amax(dim=-1)
            bad_time = int(torch.maximum(left_steps, right_steps).argmax()) + 1
            raise RuntimeError(
                "branch beam violated its hard per-joint step constraint: "
                f"{float(largest_selected_step):.6f} > {maximum_joint_step_rad:.6f} rad "
                f"at frame {bad_time}; pair {int(chosen[bad_time - 1])} -> "
                f"{int(chosen[bad_time])}"
            )
    self_clearance = torch.minimum(self_capsule_clearance(design, selected_lq)[0],
                                   self_capsule_clearance(design, selected_rq)[0])
    dual_clearance = dual_capsule_clearance(design, selected_lq, design, selected_rq,
                                            left_base_xyz, right_base_xyz)[0]
    table_clearance = torch.minimum(
        table_capsule_clearance(design, selected_lq, left_base_xyz)[0],
        table_capsule_clearance(design, selected_rq, right_base_xyz)[0])
    return BimanualTrajectory(selected_lq, selected_rq, li, ri,
                              left.position_error_m[0, time, li], right.position_error_m[0, time, ri],
                              left.orientation_error_rad[0, time, li], right.orientation_error_rad[0, time, ri],
                              self_clearance, dual_clearance, table_clearance, beam_cost.min())
