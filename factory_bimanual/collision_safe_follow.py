"""Collision-hard paired following with deterministic pose-relaxation tiers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mujoco
import numpy as np

from .mujoco_candidate_generator import (
    CandidateGeneratorConfig, MuJoCoCandidateGenerator,
)
from .bounded_orientation_adaptation import (
    OrientationAdaptationConfig, orientation_adaptation_candidates,
)
from .strict_bimanual_ik import IKCandidate


@dataclass(frozen=True)
class PoseToleranceTier:
    name: str
    position_tolerance_m: float
    orientation_tolerance_rad: float
    orientation_weight: float
    separation_offset_m: float
    priority: int


DEFAULT_TIERS = (
    PoseToleranceTier("exact", .001, np.deg2rad(1.5), 1.0, 0.0, 0),
    PoseToleranceTier("clearance_3mm", .003, np.deg2rad(5.), .5, .003, 1),
    PoseToleranceTier("clearance_6mm", .006, np.deg2rad(10.), .25, .006, 2),
    PoseToleranceTier("position_priority", .010, np.pi, 0.0, .010, 3),
    # The official Piper X fingers are wider than the demonstrated hand TCP
    # separation in parts of Seal Bag.  This last-resort yielding layer is
    # generated only when all tighter tiers lack a collision-free swept edge.
    PoseToleranceTier("collision_clearance_20mm", .025, np.pi, 0.0, .020, 4),
)


CandidateProvider = Callable[
    [Any, Any, Any, int, str, PoseToleranceTier], Sequence[IKCandidate]
]


@dataclass(frozen=True)
class CollisionSafeFollowConfig:
    collision_checker: Any
    tiers: tuple[PoseToleranceTier, ...] = DEFAULT_TIERS
    candidate_provider: CandidateProvider | None = None
    name_map: dict[str, dict[str, Any]] | None = None
    beam_width: int = 64
    maximum_pairs_per_row: int = 64
    maximum_candidates_per_tier: int = 4
    candidate_iterations: int = 80
    global_seed_count: int = 12
    constrained_fallback_enabled: bool = True
    stratified_seed_enabled: bool = False
    bounded_optimizer_enabled: bool = False
    wrist_risk_enabled: bool = False
    stratified_refresh_interval: int = 20
    rolling_early_stop_candidates: int = 2
    reference_aware_enabled: bool = False
    orientation_adaptation_enabled: bool = False
    orientation_adaptation: OrientationAdaptationConfig = (
        OrientationAdaptationConfig())
    orientation_reference_budget: int = 4
    full_orientation_trial_budget: int = 12
    orientation_continuity_weight: float = 1e-3
    generate_all_tiers: bool = True
    max_velocity_rad_s: float | np.ndarray = 3.0
    max_jump_rad: float | np.ndarray = np.deg2rad(35.)
    periodic_joints: tuple[np.ndarray, np.ndarray] | None = None
    initial_left_q: np.ndarray | None = None
    initial_right_q: np.ndarray | None = None


@dataclass
class CollisionSafeFollowResult:
    left_q: np.ndarray
    right_q: np.ndarray
    left_actual_tcp: np.ndarray
    right_actual_tcp: np.ndarray
    left_position_error_m: np.ndarray
    right_position_error_m: np.ndarray
    left_orientation_error_rad: np.ndarray
    right_orientation_error_rad: np.ndarray
    left_tier: np.ndarray
    right_tier: np.ndarray
    paired_branch_indices: list[tuple[int, int] | None]
    followed: np.ndarray
    recovery_mode: np.ndarray
    collision_count: int
    source_row_indices: np.ndarray
    state_clearance_m: np.ndarray
    swept_clearance_m: np.ndarray
    minimum_clearance_m: float
    limiting_clearance_pair: str | None
    failure_reason: np.ndarray
    selected_joint_limit_margin_rad: dict[str, np.ndarray]
    selected_singularity_margin: dict[str, np.ndarray]
    target_state_safe: np.ndarray
    target_connectable: np.ndarray
    candidate_count: dict[str, np.ndarray]
    safe_pair_count: np.ndarray
    orientation_adaptation_level: dict[str, np.ndarray]
    adapted_target_quaternion_wxyz: dict[str, np.ndarray]
    tool_axis_offset_deg: dict[str, np.ndarray]
    swing_offset_deg: dict[str, np.ndarray]


@dataclass(frozen=True)
class _Tagged:
    candidate: IKCandidate
    tier: PoseToleranceTier


@dataclass
class _PairNode:
    left: _Tagged
    right: _Tagged
    cost: tuple[int, int, float, float]
    parent: "_PairNode | None"
    row: int
    followed: bool
    recovery_mode: str


def _delta(new, old, periodic):
    value = np.asarray(new, float) - np.asarray(old, float)
    value = value.copy()
    value[periodic] = (value[periodic] + np.pi) % (2 * np.pi) - np.pi
    return value


def _orientation_continuity_cost(previous, current):
    """Squared change of the auditable local orientation offset, in deg²."""
    tool = (float(current.tool_axis_offset_deg) -
            float(previous.tool_axis_offset_deg))
    swing = (float(current.swing_offset_deg) -
             float(previous.swing_offset_deg))
    return tool*tool + swing*swing


def _motion_valid(previous, current, dt, config):
    if dt <= 0:
        return False
    periodic = config.periodic_joints or (
        np.zeros_like(previous[0], bool), np.zeros_like(previous[1], bool))
    for index, (old, new) in enumerate(zip(previous, current)):
        amount = np.abs(_delta(new, old, periodic[index]))
        if np.any(amount > np.asarray(config.max_jump_rad)):
            return False
        if np.any(amount / dt > np.asarray(config.max_velocity_rad_s)):
            return False
    return True


def _motion_rejection(previous, current, dt, config):
    """Classify a failed fixed-time connection without hiding missing IK."""
    if dt <= 0:
        return "dynamic_limit"
    periodic = config.periodic_joints or (
        np.zeros_like(previous[0], bool), np.zeros_like(previous[1], bool))
    jump_rejected = False
    velocity_rejected = False
    for index, (old, new) in enumerate(zip(previous, current)):
        amount = np.abs(_delta(new, old, periodic[index]))
        jump_rejected |= bool(np.any(
            amount > np.asarray(config.max_jump_rad)))
        velocity_rejected |= bool(np.any(
            amount / dt > np.asarray(config.max_velocity_rad_s)))
    if velocity_rejected and not jump_rejected:
        return "dynamic_limit"
    if jump_rejected:
        return "branch_discontinuity"
    return None


def _target_blocker(frontier, left, right, safe_pairs, row, times, config):
    if not left and not right:
        return "both_pose_unreachable"
    if not left:
        return "left_pose_unreachable"
    if not right:
        return "right_pose_unreachable"
    if not safe_pairs:
        return "state_collision_or_clearance"
    if row == 0:
        return "target_available"
    dt = times[row] - times[row - 1]
    rejection_classes = set()
    motion_feasible = False
    for previous in frontier:
        old = (previous.left.candidate.q, previous.right.candidate.q)
        for left_item, right_item in safe_pairs:
            current = (left_item.candidate.q, right_item.candidate.q)
            rejected = _motion_rejection(old, current, dt, config)
            if rejected is not None:
                rejection_classes.add(rejected)
                continue
            motion_feasible = True
            if config.collision_checker.transition(old, current).valid:
                return "target_available"
    if motion_feasible:
        return "swept_collision_or_clearance"
    if "dynamic_limit" in rejection_classes:
        return "dynamic_limit"
    return "branch_discontinuity"


def _bounded_motion(previous, target, dt, config):
    periodic = config.periodic_joints or (
        np.zeros_like(previous[0], bool), np.zeros_like(previous[1], bool))
    output = []
    for index, (old, new) in enumerate(zip(previous, target)):
        delta = _delta(new, old, periodic[index])
        jump = np.broadcast_to(np.asarray(config.max_jump_rad, float), delta.shape)
        velocity = np.broadcast_to(
            np.asarray(config.max_velocity_rad_s, float), delta.shape) * dt
        cap = np.minimum(jump, velocity)
        output.append(np.asarray(old, float) + np.clip(delta, -cap, cap))
    return tuple(output)


def _recovery_tag(tagged, q):
    item = tagged.candidate
    return _Tagged(IKCandidate(
        np.asarray(q, float), item.branch_index, item.pose_cost, None,
        np.nan, np.nan, item.joint_limit_margin_rad,
        item.singularity_margin, item.wrist_risk), tagged.tier)


def _pair_sort_key(pair, clearance=0.0):
    left, right = pair
    priority = max(left.tier.priority, right.tier.priority)
    pose = (left.candidate.position_error_m + right.candidate.position_error_m
            + .1 * (left.candidate.orientation_error_rad
                    + right.candidate.orientation_error_rad))
    margin = min(left.candidate.joint_limit_margin_rad,
                 right.candidate.joint_limit_margin_rad)
    singularity = min(left.candidate.singularity_margin,
                      right.candidate.singularity_margin)
    wrist = left.candidate.wrist_risk + right.candidate.wrist_risk
    return (priority, -float(clearance), wrist, pose,
            -singularity, -margin,
            left.candidate.branch_index, right.candidate.branch_index)


def _bounded_tier_diverse_pairs(
        pairs, tiers, maximum, clearance_evaluator=None):
    """Keep transition bridges from every tier, not only exact-state ties."""
    clearances = {
        (id(pair[0]), id(pair[1])): (
            float(clearance_evaluator(pair))
            if clearance_evaluator is not None else 0.0)
        for pair in pairs
    }
    key = lambda pair: _pair_sort_key(
        pair, clearances[(id(pair[0]), id(pair[1]))])
    groups = {tier.priority: [] for tier in tiers}
    for pair in pairs:
        priority = max(pair[0].tier.priority, pair[1].tier.priority)
        groups.setdefault(priority, []).append(pair)
    quota = max(1, maximum // max(1, len(groups)))
    selected = []
    remainder = []
    for priority in sorted(groups):
        ordered = sorted(groups[priority], key=key)
        selected.extend(ordered[:quota]); remainder.extend(ordered[quota:])
    if len(selected) < maximum:
        selected.extend(sorted(remainder, key=key)[
            :maximum-len(selected)])
    return selected[:maximum]


def _bounded_recovery_diverse_frontier(nodes, beam_width):
    key = lambda node: (
        node.cost, node.left.candidate.branch_index,
        node.right.candidate.branch_index)
    targets = sorted((node for node in nodes if node.followed), key=key)
    moving = sorted((node for node in nodes
                     if not node.followed
                     and node.recovery_mode != "collision_safe_hold"), key=key)
    holds = sorted((node for node in nodes
                    if node.recovery_mode == "collision_safe_hold"), key=key)
    # A zero-motion hold is always cheaper than an escape step and used to
    # consume every recovery slot.  Reserve independent capacity for actual
    # movement so a collision cul-de-sac can be exited over several rows.
    hold_quota = max(1, beam_width // 8)
    moving_quota = max(1, 3 * beam_width // 8)
    target_quota = max(0, beam_width - hold_quota - moving_quota)
    chosen = targets[:target_quota]
    chosen.extend(moving[:moving_quota])
    chosen.extend(holds[:hold_quota])
    if len(chosen) < beam_width:
        used = {id(node) for node in chosen}
        chosen.extend(node for node in sorted(nodes, key=key)
                      if id(node) not in used)
    return chosen[:beam_width]


def _has_feasible_target_edge(frontier, safe_pairs, row, times, config):
    """Tell coarse generation whether this tier is usable, not merely safe.

    A state-safe IK pair can still require a swept arm-arm collision from every
    live branch.  In that case the next tolerance tier is a necessary local
    bridge and must be generated before stopping refinement.
    """
    if not safe_pairs:
        return False
    if row == 0:
        return True
    dt = times[row] - times[row - 1]
    for previous in frontier:
        old = (previous.left.candidate.q, previous.right.candidate.q)
        for left, right in safe_pairs:
            current = (left.candidate.q, right.candidate.q)
            if (_motion_valid(old, current, dt, config)
                    and config.collision_checker.transition(old, current).valid):
                return True
    return False


def _default_provider(model, data, contract, task, config):
    if config.name_map is None:
        raise ValueError("name_map is required without a candidate_provider")
    generators = {}
    for tier in config.tiers:
        generators[tier.name] = MuJoCoCandidateGenerator(
            model, data, contract, name_map=config.name_map,
            config=CandidateGeneratorConfig(
                position_tolerance_m=tier.position_tolerance_m,
                orientation_tolerance_rad=tier.orientation_tolerance_rad,
                orientation_weight=tier.orientation_weight,
                max_iterations=config.candidate_iterations,
                global_seed_count=config.global_seed_count,
                maximum_candidates=config.maximum_candidates_per_tier,
                constrained_fallback_enabled=(
                    tier.priority == 0 and
                    config.constrained_fallback_enabled),
                stratified_seed_enabled=config.stratified_seed_enabled,
                bounded_optimizer_enabled=config.bounded_optimizer_enabled,
                wrist_risk_enabled=config.wrist_risk_enabled,
                stratified_refresh_interval=(
                    config.stratified_refresh_interval),
                rolling_early_stop_candidates=(
                    config.rolling_early_stop_candidates),
            ))

    def generate(row, side, tier, *, force_stratified):
        own = np.asarray(getattr(task, f"{side}_position_m")[row], float)
        other_side = "right" if side == "left" else "left"
        other = np.asarray(
            getattr(task, f"{other_side}_position_m")[row], float)
        away = own - other
        norm = float(np.linalg.norm(away))
        if norm > 1e-12:
            away = away / norm
        quaternion = getattr(task, f"{side}_quaternion_wxyz")[row]
        directions = [away] if norm > 1e-12 else [np.array([1., 0., 0.])]
        generated = []
        for direction in directions:
            direction = direction / np.linalg.norm(direction)
            shifted = own + tier.separation_offset_m * direction
            generated.extend(generators[tier.name].generate_target(
                side, shifted, quaternion,
                force_stratified=force_stratified))
        # Rank/report against the original demonstration target, not the
        # deliberately shifted collision-clearance target.
        result = []
        target_q = np.asarray(quaternion, float)
        target_q /= np.linalg.norm(target_q)
        for item in generated:
            actual_q = item.actual_tcp[3:]
            residual = np.empty(3)
            mujoco.mju_subQuat(residual, target_q, actual_q)
            pe = float(np.linalg.norm(item.actual_tcp[:3] -
                                      getattr(task, f"{side}_position_m")[row]))
            oe = float(np.linalg.norm(residual))
            normalized_pose = max(
                pe / .001,
                oe / np.deg2rad(1.5),
            )
            result.append(IKCandidate(
                item.q, item.branch_index,
                normalized_pose + 10.0 * item.wrist_risk,
                item.actual_tcp, pe, oe, item.joint_limit_margin_rad,
                item.singularity_margin, item.wrist_risk))
        result.sort(key=lambda item: (
            item.wrist_risk, item.pose_cost,
            -item.singularity_margin, -item.joint_limit_margin_rad,
            item.branch_index))
        unique = []
        for item in result:
            if any(np.linalg.norm(item.q - old.q) < 1e-4 for old in unique):
                continue
            unique.append(item)
            if len(unique) >= config.maximum_candidates_per_tier:
                break
        return unique

    def provide(_model, _contract, _task, row, side, tier):
        return generate(row, side, tier, force_stratified=False)

    def diversify(_model, _contract, _task, row, side, tier):
        return generate(row, side, tier, force_stratified=True)

    def reference(_model, _contract, _task, row, side, tier, reference_q):
        if tier.priority != 0:
            return []
        candidate = generators[tier.name].generate_reference_candidate(
            side, getattr(task, f"{side}_position_m")[row],
            getattr(task, f"{side}_quaternion_wxyz")[row],
            reference_q=np.asarray(reference_q, dtype=float))
        return [] if candidate is None else [candidate]

    def adapted(_model, _contract, _task, row, side, tier, references):
        if tier.priority != 0:
            return []
        original = getattr(task, f"{side}_quaternion_wxyz")[row]
        alternatives = orientation_adaptation_candidates(
            original, config=config.orientation_adaptation)
        by_level = {
            level: [item for item in alternatives if item.level == level]
            for level in ("tool_axis", "full_pose")}
        reference_budget = int(config.orientation_reference_budget)
        if reference_budget < 1:
            raise ValueError("orientation_reference_budget must be positive")
        references = references[:reference_budget]
        full_budget = int(config.full_orientation_trial_budget)
        if full_budget < 1:
            raise ValueError("full_orientation_trial_budget must be positive")
        full = by_level["full_pose"]
        if len(full) > full_budget:
            indices = np.linspace(
                0, len(full)-1, full_budget, dtype=int)
            by_level["full_pose"] = [full[index] for index in indices]
        for level in ("tool_axis", "full_pose"):
            generated = []
            for alternative in by_level[level]:
                for reference_q in references:
                    item = generators[tier.name].generate_reference_candidate(
                        side, getattr(task, f"{side}_position_m")[row],
                        alternative.quaternion_wxyz,
                        reference_q=np.asarray(reference_q, dtype=float))
                    if item is None:
                        continue
                    penalty = (
                        abs(alternative.tool_axis_offset_deg) /
                        config.orientation_adaptation.tool_hard_limit_deg +
                        alternative.swing_offset_deg /
                        config.orientation_adaptation.swing_hard_limit_deg)
                    generated.append(IKCandidate(
                        q=item.q, branch_index=10000+len(generated),
                        pose_cost=item.pose_cost+penalty,
                        actual_tcp=item.actual_tcp,
                        position_error_m=item.position_error_m,
                        orientation_error_rad=item.orientation_error_rad,
                        joint_limit_margin_rad=item.joint_limit_margin_rad,
                        singularity_margin=item.singularity_margin,
                        wrist_risk=item.wrist_risk,
                        adaptation_level=alternative.level,
                        adapted_target_quaternion_wxyz=(
                            alternative.quaternion_wxyz.copy()),
                        tool_axis_offset_deg=(
                            alternative.tool_axis_offset_deg),
                        swing_offset_deg=alternative.swing_offset_deg))
            if generated:
                return generated[:config.maximum_candidates_per_tier]
        return []

    provide.diversify = diversify
    provide.reference = reference
    provide.adapted = adapted

    return provide


def _extend_unique_tagged(output, candidates, tier):
    for candidate in candidates:
        if any(np.linalg.norm(candidate.q - old.candidate.q) <= 1e-6
               for old in output):
            continue
        output.append(_Tagged(candidate, tier))


def _initial_q(model, contract, config):
    if config.initial_left_q is not None and config.initial_right_q is not None:
        return (np.asarray(config.initial_left_q, float),
                np.asarray(config.initial_right_q, float))
    if model is None or config.name_map is None:
        dof = int(getattr(contract, "dof_per_arm", 0))
        return np.zeros(dof), np.zeros(dof)
    values = []
    for side in ("left", "right"):
        ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in config.name_map[side]["joints"]]
        values.append(np.mean(model.jnt_range[ids], axis=1))
    return tuple(values)


def solve_collision_safe_follow(model: Any, contract: Any, task: Any,
                                config: CollisionSafeFollowConfig
                                ) -> CollisionSafeFollowResult:
    """Select one source-row-aligned pair path; collisions are infeasible."""
    if not config.tiers or config.tiers[0].priority != 0:
        raise ValueError("the first tolerance tier must be exact priority zero")
    if config.beam_width < 1 or config.maximum_pairs_per_row < 1:
        raise ValueError("beam and pair bounds must be positive")
    times = np.asarray(task.time_s, float)
    rows = np.asarray(task.source_row_indices)
    if len(times) != len(rows) or len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("task timestamps/source rows are invalid")
    data = mujoco.MjData(model) if model is not None else None
    provider = config.candidate_provider or _default_provider(
        model, data, contract, task, config)
    initial = _initial_q(model, contract, config)
    initial_candidate = lambda q: IKCandidate(
        np.asarray(q, float), -1, actual_tcp=None,
        position_error_m=np.nan, orientation_error_rad=np.nan,
        joint_limit_margin_rad=np.nan, singularity_margin=np.nan)
    exact = config.tiers[0]
    seed = _PairNode(_Tagged(initial_candidate(initial[0]), exact),
                     _Tagged(initial_candidate(initial[1]), exact),
                     (0, 0, 0., 0.), None, -1, False, "initial")
    frontier = [seed]
    target_blockers = []
    candidate_counts = {"left": [], "right": []}
    safe_pair_counts = []

    for row in range(len(times)):
        left: list[_Tagged] = []
        right: list[_Tagged] = []
        safe_pairs = []
        for tier in config.tiers:
            left.extend(_Tagged(item, tier) for item in provider(
                model, contract, task, row, "left", tier))
            right.extend(_Tagged(item, tier) for item in provider(
                model, contract, task, row, "right", tier))
            safe_pairs = [(l, r) for l in left for r in right
                          if config.collision_checker.state(
                              l.candidate.q, r.candidate.q).valid]
            blocker = _target_blocker(
                frontier, left, right, safe_pairs, row, times, config)
            reference = getattr(provider, "reference", None)
            if (config.reference_aware_enabled and tier.priority == 0 and
                    callable(reference) and blocker in {
                        "dynamic_limit", "branch_discontinuity",
                        "state_collision_or_clearance",
                        "swept_collision_or_clearance"}):
                seen = {"left": [], "right": []}
                for previous in frontier:
                    for side, tagged, output in (
                            ("left", previous.left, left),
                            ("right", previous.right, right)):
                        reference_q = tagged.candidate.q
                        if any(np.linalg.norm(reference_q-old) <= 1e-6
                               for old in seen[side]):
                            continue
                        seen[side].append(reference_q.copy())
                        _extend_unique_tagged(output, reference(
                            model, contract, task, row, side, tier,
                            reference_q), tier)
                safe_pairs = [(l, r) for l in left for r in right
                              if config.collision_checker.state(
                                  l.candidate.q, r.candidate.q).valid]
                blocker = _target_blocker(
                    frontier, left, right, safe_pairs, row, times, config)
            adapted = getattr(provider, "adapted", None)
            if (config.orientation_adaptation_enabled and tier.priority == 0
                    and callable(adapted) and blocker != "target_available"):
                references = {"left": [], "right": []}
                for node in frontier:
                    for side, tagged in (("left", node.left),
                                         ("right", node.right)):
                        q = tagged.candidate.q
                        if not any(np.linalg.norm(q-old) <= 1e-6
                                   for old in references[side]):
                            references[side].append(q)
                _extend_unique_tagged(left, adapted(
                    model, contract, task, row, "left", tier,
                    references["left"]), tier)
                _extend_unique_tagged(right, adapted(
                    model, contract, task, row, "right", tier,
                    references["right"]), tier)
                safe_pairs = [(l, r) for l in left for r in right
                              if config.collision_checker.state(
                                  l.candidate.q, r.candidate.q).valid]
                blocker = _target_blocker(
                    frontier, left, right, safe_pairs, row, times, config)
            diversify = getattr(provider, "diversify", None)
            if (callable(diversify) and blocker in {
                    "branch_discontinuity",
                    "state_collision_or_clearance",
                    "swept_collision_or_clearance"}):
                _extend_unique_tagged(left, diversify(
                    model, contract, task, row, "left", tier), tier)
                _extend_unique_tagged(right, diversify(
                    model, contract, task, row, "right", tier), tier)
                safe_pairs = [(l, r) for l in left for r in right
                              if config.collision_checker.state(
                                  l.candidate.q, r.candidate.q).valid]
            if (not config.generate_all_tiers
                    and _has_feasible_target_edge(
                        frontier, safe_pairs, row, times, config)):
                break
        candidate_counts["left"].append(len(left))
        candidate_counts["right"].append(len(right))
        safe_pair_counts.append(len(safe_pairs))
        target_blockers.append(_target_blocker(
            frontier, left, right, safe_pairs, row, times, config))
        clearance_fn = getattr(config.collision_checker, "clearance", None)
        safe_pairs = _bounded_tier_diverse_pairs(
            safe_pairs, config.tiers, config.maximum_pairs_per_row,
            clearance_evaluator=(
                (lambda pair: clearance_fn(
                    pair[0].candidate.q,
                    pair[1].candidate.q).minimum_m)
                if callable(clearance_fn) else None))
        next_frontier = []
        for previous in frontier:
            old = (previous.left.candidate.q, previous.right.candidate.q)
            for left_item, right_item in safe_pairs:
                current = (left_item.candidate.q, right_item.candidate.q)
                if previous.row >= 0:
                    dt = times[row] - times[row - 1]
                    if not _motion_valid(old, current, dt, config):
                        continue
                    if not config.collision_checker.transition(old, current).valid:
                        continue
                motion = float(sum(np.sum((new-old_q) ** 2)
                                   for old_q, new in zip(old, current)))
                motion += config.orientation_continuity_weight * (
                    _orientation_continuity_cost(
                        previous.left.candidate, left_item.candidate) +
                    _orientation_continuity_cost(
                        previous.right.candidate, right_item.candidate))
                tier_cost = max(left_item.tier.priority,
                                right_item.tier.priority)
                pose = float(left_item.candidate.pose_cost
                             + right_item.candidate.pose_cost)
                cost = (previous.cost[0], previous.cost[1] + tier_cost,
                        previous.cost[2] + pose,
                        previous.cost[3] + motion)
                next_frontier.append(_PairNode(
                    left_item, right_item, cost, previous, row, True,
                    "target"))
            if previous.row >= 0:
                dt = times[row] - times[row - 1]
                recovery_pairs = (safe_pairs if safe_pairs else [
                    (left_item, right_item)
                    for left_item in left for right_item in right
                ])
                for left_item, right_item in safe_pairs[:8]:
                    target = (left_item.candidate.q,
                              right_item.candidate.q)
                    current = _bounded_motion(old, target, dt, config)
                    if all(np.array_equal(a, b)
                           for a, b in zip(old, current)):
                        continue
                    if not config.collision_checker.state(*current).valid:
                        continue
                    if not config.collision_checker.transition(old, current).valid:
                        continue
                    motion = float(sum(np.sum((new-old_q) ** 2)
                                       for old_q, new in zip(old, current)))
                    remaining = min((
                        float(np.sum((pair_left.candidate.q-current[0]) ** 2) +
                              np.sum((pair_right.candidate.q-current[1]) ** 2))
                        for pair_left, pair_right in safe_pairs), default=0.0)
                    cost = (previous.cost[0] + 1,
                            previous.cost[1] + max(
                                left_item.tier.priority,
                                right_item.tier.priority),
                            previous.cost[2],
                            previous.cost[3] + remaining + 1e-3 * motion)
                    next_frontier.append(_PairNode(
                        _recovery_tag(left_item, current[0]),
                        _recovery_tag(right_item, current[1]),
                        cost, previous, row, False,
                        "bounded_collision_safe_recovery"))
                # An exact endpoint pair may collide even though the first
                # bounded step toward it is state- and edge-safe.  Advancing
                # both arms to that verified boundary preserves their IK
                # branches better than yielding one arm or holding still.
                if not safe_pairs:
                    collision_targets = sorted(
                        recovery_pairs,
                        key=_pair_sort_key)[:8]
                    for left_item, right_item in collision_targets:
                        target = (left_item.candidate.q,
                                  right_item.candidate.q)
                        current = _bounded_motion(old, target, dt, config)
                        if all(np.array_equal(a, b)
                               for a, b in zip(old, current)):
                            continue
                        if not config.collision_checker.state(*current).valid:
                            continue
                        if not config.collision_checker.transition(
                                old, current).valid:
                            continue
                        motion = float(sum(np.sum((new-old_q) ** 2)
                                           for old_q, new in zip(old, current)))
                        remaining = float(
                            np.sum((target[0]-current[0]) ** 2) +
                            np.sum((target[1]-current[1]) ** 2))
                        cost = (
                            previous.cost[0] + 1,
                            previous.cost[1] + max(
                                left_item.tier.priority,
                                right_item.tier.priority),
                            previous.cost[2],
                            previous.cost[3] + remaining + 1e-3 * motion,
                        )
                        next_frontier.append(_PairNode(
                            _recovery_tag(left_item, current[0]),
                            _recovery_tag(right_item, current[1]),
                            cost, previous, row, False,
                            "bounded_collision_approach"))
                # Coordinated yielding: advance only one arm when simultaneous
                # motion would sweep the grippers through each other. These
                # are explicit source-row states, never hidden interpolation.
                for tagged, advance_left in (
                        *((item, True) for item in left),
                        *((item, False) for item in right)):
                    target = ((tagged.candidate.q,
                               previous.right.candidate.q) if advance_left
                              else (previous.left.candidate.q,
                                    tagged.candidate.q))
                    current = _bounded_motion(old, target, dt, config)
                    left_item = (_recovery_tag(tagged, current[0])
                                 if advance_left else previous.left)
                    right_item = (previous.right if advance_left else
                                  _recovery_tag(tagged, current[1]))
                    if not _motion_valid(old, current, dt, config):
                        continue
                    if not config.collision_checker.state(*current).valid:
                        continue
                    if not config.collision_checker.transition(old, current).valid:
                        continue
                    motion = float(sum(np.sum((new-old_q) ** 2)
                                       for old_q, new in zip(old, current)))
                    remaining = min((
                        float(np.sum((pair_left.candidate.q-current[0]) ** 2) +
                              np.sum((pair_right.candidate.q-current[1]) ** 2))
                        for pair_left, pair_right in recovery_pairs),
                        default=0.0)
                    cost = (previous.cost[0] + 1,
                            previous.cost[1] + tagged.tier.priority,
                            previous.cost[2] + tagged.candidate.pose_cost,
                            previous.cost[3] + remaining + 1e-3 * motion)
                    next_frontier.append(_PairNode(
                        left_item, right_item, cost, previous, row, False,
                        ("left_only_collision_avoidance" if advance_left
                         else "right_only_collision_avoidance")))
            # A stationary safe state is always a valid conservative fallback.
            if previous.row >= 0:
                # A safe hold remains available, but should not dominate a
                # bounded safe recovery forever merely because it has zero
                # motion. The unit penalty makes continued progress preferable
                # while preserving HOLD when every moving edge is unsafe.
                hold_remaining = min((
                    float(np.sum((left_item.candidate.q-old[0]) ** 2) +
                          np.sum((right_item.candidate.q-old[1]) ** 2))
                    for left_item, right_item in recovery_pairs), default=1.0)
                hold_cost = (previous.cost[0] + 1, previous.cost[1],
                             previous.cost[2],
                             previous.cost[3] + hold_remaining + 1.0)
                next_frontier.append(_PairNode(
                    previous.left, previous.right, hold_cost, previous, row,
                    False, "collision_safe_hold"))
        if row == 0 and config.collision_checker.state(*initial).valid:
            next_frontier.append(_PairNode(
                seed.left, seed.right, (1, 0, 0., 0.), seed, row, False,
                "collision_safe_hold"))
        if not next_frontier:
            # Row zero may itself be impossible. Hold the verified initial pair.
            if not config.collision_checker.state(*initial).valid:
                raise RuntimeError("initial bimanual state is colliding")
            next_frontier = [_PairNode(
                seed.left, seed.right, (1, 0, 0., 0.), seed, row, False,
                "collision_safe_hold")]
        frontier = _bounded_recovery_diverse_frontier(
            next_frontier, config.beam_width)

    final = min(frontier, key=lambda node: node.cost)
    selected = [None] * len(times)
    node = final
    while node is not None and node.row >= 0:
        selected[node.row] = node
        node = node.parent
    left_dof, right_dof = len(initial[0]), len(initial[1])
    left_q = np.zeros((len(times), left_dof)); right_q = np.zeros((len(times), right_dof))
    tcp = {side: np.full((len(times), 7), np.nan) for side in ("left", "right")}
    pe = {side: np.full(len(times), np.nan) for side in ("left", "right")}
    oe = {side: np.full(len(times), np.nan) for side in ("left", "right")}
    joint_margin = {
        side: np.full(len(times), np.nan) for side in ("left", "right")}
    singularity = {
        side: np.full(len(times), np.nan) for side in ("left", "right")}
    tiers = {side: np.empty(len(times), dtype=object) for side in ("left", "right")}
    branches = []
    followed = np.zeros(len(times), bool); recovery = np.empty(len(times), object)
    collision_count = 0
    failure_reason = np.empty(len(times), dtype=object)
    adaptation_level = {
        side: np.full(len(times), "original", dtype=object)
        for side in ("left", "right")}
    adapted_quaternion = {
        side: np.asarray(getattr(
            task, f"{side}_quaternion_wxyz"), dtype=float).copy()
        for side in ("left", "right")}
    tool_offset = {
        side: np.zeros(len(times), dtype=float) for side in ("left", "right")}
    swing_offset = {
        side: np.zeros(len(times), dtype=float) for side in ("left", "right")}
    for row, chosen in enumerate(selected):
        assert chosen is not None
        for side, tagged, output in (("left", chosen.left, left_q),
                                     ("right", chosen.right, right_q)):
            output[row] = tagged.candidate.q
            tiers[side][row] = tagged.tier.name if chosen.followed else "hold"
            pe[side][row] = tagged.candidate.position_error_m
            oe[side][row] = tagged.candidate.orientation_error_rad
            joint_margin[side][row] = (
                tagged.candidate.joint_limit_margin_rad)
            singularity[side][row] = tagged.candidate.singularity_margin
            adaptation_level[side][row] = tagged.candidate.adaptation_level
            tool_offset[side][row] = tagged.candidate.tool_axis_offset_deg
            swing_offset[side][row] = tagged.candidate.swing_offset_deg
            if tagged.candidate.adapted_target_quaternion_wxyz is not None:
                adapted_quaternion[side][row] = (
                    tagged.candidate.adapted_target_quaternion_wxyz)
            if tagged.candidate.actual_tcp is not None:
                tcp[side][row] = tagged.candidate.actual_tcp
        branches.append((chosen.left.candidate.branch_index,
                         chosen.right.candidate.branch_index))
        followed[row] = chosen.followed; recovery[row] = chosen.recovery_mode
        failure_reason[row] = (
            "ok" if chosen.followed else
            ("safe_recovery_for_connectivity"
             if target_blockers[row] == "target_available"
             else target_blockers[row]))
        if not config.collision_checker.state(left_q[row], right_q[row]).valid:
            collision_count += 1
        if row and not config.collision_checker.transition(
                (left_q[row-1], right_q[row-1]),
                (left_q[row], right_q[row])).valid:
            collision_count += 1
    state_clearance = np.full(len(times), np.nan)
    swept_clearance = np.full(len(times), np.nan)
    state_limiting = np.full(len(times), None, dtype=object)
    swept_limiting = np.full(len(times), None, dtype=object)
    clearance_fn = getattr(config.collision_checker, "clearance", None)
    transition_clearance_fn = getattr(
        config.collision_checker, "transition_clearance", None)
    if callable(clearance_fn):
        for row in range(len(times)):
            report = clearance_fn(left_q[row], right_q[row])
            state_clearance[row] = report.minimum_m
            state_limiting[row] = report.limiting_pair
    if callable(transition_clearance_fn):
        for row in range(1, len(times)):
            report = transition_clearance_fn(
                (left_q[row-1], right_q[row-1]),
                (left_q[row], right_q[row]))
            swept_clearance[row] = report.minimum_m
            swept_limiting[row] = report.limiting_pair
    combined = np.r_[state_clearance, swept_clearance]
    if np.any(np.isfinite(combined)):
        flat_index = int(np.nanargmin(combined))
        minimum_clearance = float(combined[flat_index])
        if flat_index < len(times):
            limiting_clearance_pair = state_limiting[flat_index]
        else:
            limiting_clearance_pair = swept_limiting[flat_index-len(times)]
    else:
        minimum_clearance = float("nan")
        limiting_clearance_pair = None
    target_state_safe = np.asarray([
        reason not in {
            "both_pose_unreachable", "left_pose_unreachable",
            "right_pose_unreachable", "state_collision_or_clearance",
        } for reason in target_blockers], dtype=bool)
    target_connectable = np.asarray([
        reason == "target_available" for reason in target_blockers],
        dtype=bool)
    return CollisionSafeFollowResult(
        left_q, right_q, tcp["left"], tcp["right"], pe["left"], pe["right"],
        oe["left"], oe["right"], tiers["left"], tiers["right"], branches,
        followed, recovery, collision_count, rows.copy(),
        state_clearance, swept_clearance, minimum_clearance,
        limiting_clearance_pair, failure_reason, joint_margin, singularity,
        target_state_safe, target_connectable,
        {side: np.asarray(values, dtype=int)
         for side, values in candidate_counts.items()},
        np.asarray(safe_pair_counts, dtype=int), adaptation_level,
        adapted_quaternion, tool_offset, swing_offset)
