from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock

import torch

from .ik import rotation_log
from .kinematics import axis_angle_matrix


class ServoMode(str, Enum):
    TRACK = "track"
    CAUTION = "caution"
    HOLD = "hold"


@dataclass(frozen=True)
class RealtimeBudget:
    control_hz: float = 50.0
    deadline_fraction: float = 0.72
    nominal_iterations: int = 3
    caution_iterations: int = 5
    branch_manager_hz: float = 3.0
    minimum_sigma: float = 0.025
    caution_clearance_m: float = 0.030
    stop_clearance_m: float = 0.004
    maximum_target_age_s: float = 0.10

    @property
    def solve_deadline_s(self) -> float:
        return self.deadline_fraction / self.control_hz


@dataclass(frozen=True)
class ServoObservation:
    target_age_s: float
    sigma_min: float
    predicted_clearance_m: float
    previous_solve_s: float
    branch_available: bool = True


@dataclass(frozen=True)
class SolveSchedule:
    mode: ServoMode
    local_iterations: int
    deadline_s: float
    target_velocity_scale: float
    request_async_branch_search: bool
    reason: str


@dataclass(frozen=True)
class BranchProposal:
    generation: int
    created_s: float
    q: torch.Tensor
    minimum_clearance_m: float
    sigma_min: float


class BranchProposalMailbox:
    """Single-slot, non-blocking handoff from slow planner to 50 Hz servo."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: BranchProposal | None = None

    def publish(self, proposal: BranchProposal) -> bool:
        with self._lock:
            if self._latest is not None and proposal.generation <= self._latest.generation:
                return False
            self._latest = BranchProposal(proposal.generation, proposal.created_s,
                                          proposal.q.detach().clone(),
                                          proposal.minimum_clearance_m, proposal.sigma_min)
            return True

    def latest(self, now_s: float, maximum_age_s: float) -> BranchProposal | None:
        with self._lock:
            proposal = self._latest
            if proposal is None or now_s - proposal.created_s > maximum_age_s:
                return None
            return proposal


def schedule_bounded_solve(observation: ServoObservation,
                           budget: RealtimeBudget = RealtimeBudget()) -> SolveSchedule:
    """Choose a deterministic, fixed-deadline online solve policy.

    The high-rate loop never performs global branch search. It requests that
    work from a lower-rate asynchronous manager and either tracks the current
    branch, slows down, or holds the last certified command.
    """
    if observation.target_age_s > budget.maximum_target_age_s:
        return SolveSchedule(ServoMode.HOLD, 0, budget.solve_deadline_s, 0.0, False,
                             "stale Cartesian target")
    if observation.predicted_clearance_m <= budget.stop_clearance_m:
        return SolveSchedule(ServoMode.HOLD, 0, budget.solve_deadline_s, 0.0, True,
                             "predicted collision stop margin")
    deadline_missed = observation.previous_solve_s > budget.solve_deadline_s
    near_singular = observation.sigma_min < budget.minimum_sigma
    near_collision = observation.predicted_clearance_m < budget.caution_clearance_m
    if not observation.branch_available:
        return SolveSchedule(ServoMode.HOLD, 0, budget.solve_deadline_s, 0.0, True,
                             "no certified IK branch")
    if deadline_missed or near_singular or near_collision:
        reasons = []
        if deadline_missed: reasons.append("previous deadline miss")
        if near_singular: reasons.append("low manipulability")
        if near_collision: reasons.append("low collision clearance")
        # A missed deadline reduces work rather than trying harder and causing
        # a cascade of missed 20 ms control periods.
        iterations = budget.nominal_iterations if deadline_missed else budget.caution_iterations
        return SolveSchedule(ServoMode.CAUTION, iterations, budget.solve_deadline_s, 0.25,
                             near_singular or near_collision, ", ".join(reasons))
    return SolveSchedule(ServoMode.TRACK, budget.nominal_iterations, budget.solve_deadline_s,
                         1.0, False, "nominal warm-start tracking")


def govern_cartesian_target(current: torch.Tensor, requested: torch.Tensor, dt_s: float,
                            schedule: SolveSchedule, *, maximum_linear_m_s: float = .5,
                            maximum_angular_rad_s: float = 2.0) -> torch.Tensor:
    """Rate-limit an SE(3) target before the fixed-deadline local MPC."""
    if current.shape != (4, 4) or requested.shape != (4, 4):
        raise ValueError("current and requested targets must be 4x4 transforms")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if schedule.mode is ServoMode.HOLD or schedule.target_velocity_scale <= 0:
        return current.clone()
    scale = schedule.target_velocity_scale
    result = current.clone()
    translation = requested[:3, 3] - current[:3, 3]
    translation_limit = maximum_linear_m_s * scale * dt_s
    translation_norm = torch.linalg.vector_norm(translation)
    fraction = torch.clamp(translation_limit / translation_norm.clamp_min(1e-12), max=1.0)
    result[:3, 3] = current[:3, 3] + translation * fraction

    error_vector = rotation_log(requested[:3, :3] @ current[:3, :3].T)
    angle = torch.linalg.vector_norm(error_vector)
    step_angle = torch.clamp(angle, max=maximum_angular_rad_s * scale * dt_s)
    axis = error_vector / angle.clamp_min(1e-12)
    result[:3, :3] = axis_angle_matrix(axis, step_angle) @ current[:3, :3]
    return result
