"""Open-source trust-region SLP refinement for short joint-path windows.

PyTorch supplies exact FK/Jacobians and SciPy HiGHS solves each linearized
subproblem. This is intentionally a local post-Pareto refinement, not a global
IK solver and not a claim of reproducing every term in B*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import linprog
import torch

from .ik import pose_error
from .kinematics import fk_tcp, geometric_jacobian
from .topology import DesignBatch


@dataclass
class SLPRefinementResult:
    q: torch.Tensor
    accepted_iterations: int
    attempted_iterations: int
    initial_merit: float
    final_merit: float
    final_trust_region_rad: float
    history: list[dict[str, float | bool]]


def _true_merit(design: DesignBatch, targets: torch.Tensor, q: torch.Tensor,
                position_weight: float, orientation_weight: float,
                smoothness_weight: float) -> torch.Tensor:
    error = pose_error(fk_tcp(design, q)[0], targets)
    weights = torch.tensor((position_weight,) * 3 + (orientation_weight,) * 3,
                           dtype=q.dtype, device=q.device)
    return (error.abs() * weights).sum() + smoothness_weight * torch.diff(
        q, dim=0).abs().sum()


def refine_joint_window_slp(
    design: DesignBatch,
    targets: torch.Tensor,
    q_initial: torch.Tensor,
    *,
    iterations: int = 20,
    trust_region_rad: float = 0.15,
    minimum_trust_region_rad: float = 1e-4,
    trust_shrink_ratio: float = 0.25,
    trust_expand_ratio: float = 1.5,
    acceptance_ratio: float = 0.1,
    position_weight: float = 400.0,
    orientation_weight: float = 8.0,
    smoothness_weight: float = 1.0,
    maximum_joint_step_rad: float | None = 0.5,
    fix_endpoints: bool = True,
    feasibility_fn: Callable[[torch.Tensor], bool] | None = None,
) -> SLPRefinementResult:
    """Refine one local trajectory window with linearized pose constraints.

    The LP minimizes L1 pose residuals and L1 joint path length inside a trust
    region. Every proposed step is re-evaluated with exact PyTorch FK; an
    optional callback can reject collision-infeasible proposals.
    """
    if design.count != 1:
        raise ValueError("SLP refinement accepts exactly one design")
    if q_initial.ndim != 2 or q_initial.shape[-1] != 6:
        raise ValueError("q_initial must have shape [T,6]")
    if targets.shape != (len(q_initial), 4, 4):
        raise ValueError("targets must have shape [T,4,4]")
    if len(q_initial) < 2:
        raise ValueError("at least two trajectory knots are required")
    q = q_initial.detach().clone()
    T, dof = q.shape
    pose_dim, delta_count = 6 * T, 6 * T
    slack_start, smooth_start = delta_count, delta_count + pose_dim
    smooth_count = 6 * (T - 1)
    variable_count = delta_count + pose_dim + smooth_count
    weights = np.tile(np.asarray((position_weight,) * 3 +
                                 (orientation_weight,) * 3), T)
    objective = np.zeros(variable_count)
    objective[slack_start:smooth_start] = weights
    objective[smooth_start:] = smoothness_weight
    initial_merit = float(_true_merit(design, targets, q, position_weight,
                                      orientation_weight, smoothness_weight))
    merit = initial_merit
    trust = float(trust_region_rad)
    accepted, history = 0, []
    lo = design.template.q_min.detach().cpu().numpy()
    hi = design.template.q_max.detach().cpu().numpy()
    for iteration in range(iterations):
        current = fk_tcp(design, q)[0]
        error = pose_error(current, targets).detach().cpu().numpy().reshape(-1)
        jacobian = geometric_jacobian(design, q)[0].detach().cpu().numpy()
        q_np = q.detach().cpu().numpy()
        rows, rhs = [], []

        # |J dq - error| <= pose_slack.
        for knot in range(T):
            for component in range(6):
                row = np.zeros(variable_count)
                d_slice = slice(6 * knot, 6 * knot + 6)
                slack = slack_start + 6 * knot + component
                row[d_slice] = jacobian[knot, component]
                row[slack] = -1
                rows.append(row); rhs.append(error[6 * knot + component])
                rows.append(-row.copy()); rows[-1][slack] = -1
                rhs.append(-error[6 * knot + component])

        # L1 path auxiliaries and optional hard per-joint step bounds.
        for knot in range(1, T):
            base_step = q_np[knot] - q_np[knot - 1]
            for joint in range(dof):
                smooth = smooth_start + 6 * (knot - 1) + joint
                row = np.zeros(variable_count)
                row[6 * knot + joint] = 1
                row[6 * (knot - 1) + joint] = -1
                row[smooth] = -1
                rows.append(row); rhs.append(-base_step[joint])
                rows.append(-row.copy()); rows[-1][smooth] = -1
                rhs.append(base_step[joint])
                if maximum_joint_step_rad is not None:
                    step_row = np.zeros(variable_count)
                    step_row[6 * knot + joint] = 1
                    step_row[6 * (knot - 1) + joint] = -1
                    rows.append(step_row)
                    rhs.append(maximum_joint_step_rad - base_step[joint])
                    rows.append(-step_row)
                    rhs.append(maximum_joint_step_rad + base_step[joint])

        bounds = []
        for knot in range(T):
            for joint in range(dof):
                lower = max(-trust, lo[joint] - q_np[knot, joint])
                upper = min(trust, hi[joint] - q_np[knot, joint])
                if fix_endpoints and knot in (0, T - 1):
                    lower = upper = 0.0
                bounds.append((lower, upper))
        bounds.extend([(0, None)] * (pose_dim + smooth_count))
        solution = linprog(objective, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                           bounds=bounds, method="highs")
        if not solution.success:
            history.append({"iteration": float(iteration), "accepted": False,
                            "trust_region_rad": trust, "lp_failed": True})
            trust *= trust_shrink_ratio
            if trust < minimum_trust_region_rad:
                break
            continue
        proposal = q + torch.as_tensor(
            solution.x[:delta_count].reshape(T, dof), dtype=q.dtype, device=q.device)
        proposal_merit = float(_true_merit(
            design, targets, proposal, position_weight, orientation_weight,
            smoothness_weight))
        predicted_improvement = max(merit - float(solution.fun), 1e-12)
        actual_improvement = merit - proposal_merit
        ratio = actual_improvement / predicted_improvement
        feasible = feasibility_fn is None or bool(feasibility_fn(proposal))
        accept = feasible and actual_improvement > 0 and ratio >= acceptance_ratio
        history.append({"iteration": float(iteration), "accepted": accept,
                        "trust_region_rad": trust, "merit": merit,
                        "proposal_merit": proposal_merit,
                        "improvement_ratio": ratio})
        if accept:
            q, merit, accepted = proposal, proposal_merit, accepted + 1
            trust *= trust_expand_ratio
        else:
            trust *= trust_shrink_ratio
        if trust < minimum_trust_region_rad:
            break
    return SLPRefinementResult(q, accepted, len(history), initial_merit, merit,
                               trust, history)
