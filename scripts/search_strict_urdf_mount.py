"""Rerank task mounts against the native-length real-mesh MuJoCo model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.solve_strict_urdf_task_cache import (ANCHOR_ROTATION, build_model,
    build_mount_model_template, collision_flags, evaluate_q_path, hold_invalid_frames,
    optimization, target_poses_for)
from scripts.strict_mujoco_ik import (joint_periodic_mask, solve_pose_path,
                                      solve_pose_path_layered,
                                      solve_pose_path_multibranch,
                                      solve_position_path, wrapped_joint_delta)
from scripts.strict_urdf_model_audit import MODELS
from design_optimization.episode_follow_metrics import (episode_follow_metrics,
    failure_reason_diagnostics, follow_rank, planner_failure_diagnostics)
from design_optimization.best_first_mount_search import (
    SearchBudget, generate_local_population, representative_frame_indices,
    representative_frame_windows, select_diverse_regions)
from design_optimization.hierarchical_mount_search import (
    MountSearchBudget, diverse_region_indices, global_mount_candidates,
    local_mount_candidates)
from design_optimization.installation_search_space import first_version_bounds, mount_rotation_matrix, tabletop_mount_feasible


def _compress_yaw_only_mounts(mounts: np.ndarray) -> np.ndarray:
    """Project six-field mounts onto active XYZ+yaw search coordinates."""
    values = np.asarray(mounts, dtype=float)
    if values.shape[-1] != 6:
        raise ValueError("full mount coordinates must have six fields")
    return values[..., [0, 1, 2, 4]].copy()


def _expand_yaw_only_mounts(mounts: np.ndarray) -> np.ndarray:
    """Restore active XYZ+yaw coordinates to the six-field mount contract."""
    values = np.asarray(mounts, dtype=float)
    if values.shape[-1] != 4:
        raise ValueError("active mount coordinates must contain XYZ and yaw")
    expanded = np.zeros(values.shape[:-1] + (6,), dtype=float)
    expanded[..., [0, 1, 2, 4]] = values
    return expanded


def _evaluate_search_candidates(evaluate_fn, *, candidates: np.ndarray, **kwargs):
    """Evaluate active coordinates while keeping search records in active space."""
    active = np.asarray(candidates, dtype=float)
    yaw_only = active.shape[-1] == 4
    evaluated = evaluate_fn(
        candidates=_expand_yaw_only_mounts(active) if yaw_only else active, **kwargs)
    if yaw_only:
        for candidate, record in zip(active, evaluated):
            record["candidate"] = candidate.copy()
    return evaluated


def build_parser() -> argparse.ArgumentParser:
    budget = MountSearchBudget()
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("local", "droid", "egodex"), required=True)
    parser.add_argument("--robot", choices=tuple(MODELS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-artifact", type=Path,
                        help="explicit processed Local episode; bypasses split-based selection")
    parser.add_argument("--search-policy", choices=("legacy", "best-first"), default="legacy")
    parser.add_argument("--candidates", type=int, default=None,
                        help="deprecated alias for --global-candidates")
    parser.add_argument("--global-candidates", type=int, default=budget.global_candidates)
    parser.add_argument("--global-retain", type=int, default=budget.global_retain)
    parser.add_argument("--local-regions", type=int, default=budget.local_regions)
    parser.add_argument("--local-per-region", type=int, default=budget.local_per_region)
    parser.add_argument("--medium-retain", type=int, default=budget.medium_retain)
    parser.add_argument("--strict-retain", type=int, default=budget.strict_retain)
    parser.add_argument("--pose-finalists", type=int, default=None,
                        help="deprecated alias for --strict-retain")
    parser.add_argument("--final-centers", type=int, default=budget.final_centers)
    parser.add_argument("--final-per-center", type=int, default=budget.final_per_center)
    parser.add_argument("--final-retain", type=int, default=16)
    parser.add_argument("--screen-frames", type=int, default=64)
    parser.add_argument("--medium-frames", type=int, default=256)
    parser.add_argument("--best-first-global-candidates", type=int, default=128)
    parser.add_argument("--best-first-rank-frames", type=int, default=24)
    parser.add_argument("--best-first-medium-frames", type=int, default=64)
    parser.add_argument("--best-first-regions", type=int, default=8)
    parser.add_argument("--best-first-local-per-region", type=int, default=8)
    parser.add_argument("--best-first-dense-before-success", type=int, default=3)
    parser.add_argument("--best-first-dense-fallback", type=int, default=5)
    parser.add_argument("--best-first-post-success-expansions", type=int, default=8)
    parser.add_argument("--best-first-post-success-dense", type=int, default=2)
    parser.add_argument("--best-first-final-candidates", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        help="optional cache path; defaults to the project strict-cache location")
    parser.add_argument("--incumbent-json", type=Path,
                        help="explicit GPU coarse-search result; overrides stale strict caches")
    parser.add_argument("--input-fingerprint", help="result-affecting input hash recorded for safe resume")
    return parser


def resolved_best_first_budget(args: argparse.Namespace) -> SearchBudget:
    """Translate CLI controls into one validated best-first budget."""
    return SearchBudget(
        global_candidates=args.best_first_global_candidates,
        retained_regions=args.best_first_regions,
        local_per_region=args.best_first_local_per_region,
        rank_frames=args.best_first_rank_frames,
        medium_frames=args.best_first_medium_frames,
        dense_before_success=args.best_first_dense_before_success,
        dense_fallback=args.best_first_dense_fallback,
        post_success_expansions=args.best_first_post_success_expansions,
        post_success_dense=args.best_first_post_success_dense,
        final_candidates=args.best_first_final_candidates,
    )


def resolved_hierarchy_budget(args: argparse.Namespace) -> dict[str, int]:
    """Resolve formal defaults or the backward-compatible compact smoke profile."""
    if args.candidates is None:
        return {
            "global": args.global_candidates, "global_retain": args.global_retain,
            "local_regions": args.local_regions, "local_per_region": args.local_per_region,
            "medium_retain": args.medium_retain,
            "strict_retain": args.pose_finalists or args.strict_retain,
            "final_centers": args.final_centers,
            "final_per_center": args.final_per_center,
            "final_retain": args.final_retain,
        }
    count = args.candidates
    return {
        "global": count,
        "global_retain": min(16, count),
        "local_regions": min(4, count),
        "local_per_region": 8,
        "medium_retain": min(16, count),
        "strict_retain": args.pose_finalists or min(6, count),
        "final_centers": min(2, count),
        "final_per_center": 16,
        "final_retain": min(4, count),
    }


def registered_targets(model, entry, q0, absolute_positions, absolute_quaternions):
    data = __import__("mujoco").MjData(model)
    joint_ids = [__import__("mujoco").mj_name2id(model, __import__("mujoco").mjtObj.mjOBJ_JOINT, n) for n in entry.joints]
    data.qpos[[int(model.jnt_qposadr[j]) for j in joint_ids]] = q0
    __import__("mujoco").mj_forward(model, data)
    site = __import__("mujoco").mj_name2id(model, __import__("mujoco").mjtObj.mjOBJ_SITE, "strict_tracking_tcp")
    initial = data.site_xmat[site].reshape(3, 3).copy()
    anchor = np.asarray(absolute_positions[0], dtype=float)
    relative_positions = (np.asarray(absolute_positions) - anchor) @ ANCHOR_ROTATION
    positions = anchor + relative_positions @ initial.T
    output = []
    for quat in absolute_quaternions:
        matrix = np.zeros(9); __import__("mujoco").mju_quat2Mat(matrix, quat)
        relative = ANCHOR_ROTATION.T @ matrix.reshape(3, 3)
        target = initial @ relative
        target_quat = np.zeros(4); __import__("mujoco").mju_mat2Quat(target_quat, target.reshape(-1))
        output.append(target_quat)
    return positions, np.asarray(output)


def _uniform_indices(frame_count: int, requested: int) -> np.ndarray:
    count = min(max(1, requested), frame_count)
    return np.linspace(0, frame_count - 1, count).round().astype(int)


def _dense_evaluation_indices(
    frame_count: int, *, previously_ranked: np.ndarray | None = None
) -> np.ndarray:
    """Dense success always means one fresh chronological full-episode solve."""
    del previously_ranked
    if frame_count < 1:
        raise ValueError("dense evaluation needs at least one frame")
    return np.arange(frame_count, dtype=int)


def _evaluate_mount_candidates(
    *, candidates: np.ndarray, targets: np.ndarray, target_quaternions: np.ndarray,
    indices: np.ndarray, robot: str, entry, iterations: int, restarts: int,
    label: str, initial_qs: list[np.ndarray | None] | None = None,
    model_template=None, index_windows: tuple[np.ndarray, ...] | None = None,
) -> list[dict[str, object]]:
    """Evaluate one hierarchy stage and retain ranks with their mount vectors."""
    if index_windows is not None:
        windows = tuple(np.asarray(window, dtype=int) for window in index_windows)
        if not windows or any(len(window) < 2 or np.any(np.diff(window) != 1)
                              for window in windows):
            raise ValueError("rank windows must be non-empty and contiguous")
        per_window = [
            _evaluate_mount_candidates(
                candidates=candidates, targets=targets,
                target_quaternions=target_quaternions, indices=window,
                robot=robot, entry=entry, iterations=iterations, restarts=restarts,
                label=f"{label}-window{window_index + 1}", initial_qs=None,
                model_template=model_template)
            for window_index, window in enumerate(windows)
        ]
        combined: list[dict[str, object]] = []
        for candidate_index in range(len(candidates)):
            records = [batch[candidate_index] for batch in per_window]
            if not all(_record_is_hard_feasible(record) for record in records):
                combined.append(records[0])
                continue
            merged = records[0]
            for record in records[1:]:
                merged = _merge_evaluation_records(merged, record)
            merged["episode_success"] = False
            merged["rank"] = (0.0, *tuple(merged["rank"])[1:])
            merged["rank_scope"] = "independent_contiguous_windows"
            combined.append(merged)
        return combined
    stage_targets = targets[indices]
    stage_quaternions = target_quaternions[indices]
    records: list[dict[str, object]] = []
    invalid_rank = (-1.0, -1.0, -float("inf"), -float("inf"), -float("inf"), -float("inf"))
    if initial_qs is not None and len(initial_qs) != len(candidates):
        raise ValueError("initial_qs must match the candidate population")
    for index, candidate in enumerate(np.asarray(candidates, dtype=float)):
        rotation = mount_rotation_matrix(
            tilt_pitch_deg=float(candidate[3]), yaw_deg=float(candidate[4]),
            roll_deg=float(candidate[5]))
        if not tabletop_mount_feasible(candidate[:3], rotation):
            records.append({"rank": invalid_rank, "candidate": candidate.copy(), "feasible": False})
            print(f"[{label} {index + 1}/{len(candidates)}] physically infeasible", flush=True)
            continue
        if model_template is None:
            model = build_model(robot, candidate[:3], float(candidate[3]),
                                float(candidate[4]), float(candidate[5]))
        else:
            model, _ = model_template.apply(
                candidate[:3], float(candidate[3]), float(candidate[4]), float(candidate[5]))
        inherited_q = None if initial_qs is None else initial_qs[index]
        if inherited_q is None:
            anchor_seed = solve_position_path(
                model, "strict_tracking_tcp", entry.joints, targets[:1],
                tolerance_m=1e-3, iterations=max(160, iterations), restarts=max(6, restarts))
            inherited_q = anchor_seed.q[0]
        result = solve_pose_path(
            model, "strict_tracking_tcp", entry.joints, stage_targets, stage_quaternions,
            position_tolerance_m=1e-3, orientation_tolerance_rad=np.deg2rad(1.5),
            iterations=iterations, restarts=restarts, initial_q=inherited_q)
        table, self_collision = collision_flags(model, entry.joints, result.q)
        q_path = hold_invalid_frames(result.q, ~result.success | table | self_collision)
        _, _, position_error, orientation_error = evaluate_q_path(
            model, entry.joints, q_path, stage_targets, stage_quaternions)
        pose_success = ((position_error <= 1e-3)
                        & (orientation_error <= np.deg2rad(1.5)))
        table, self_collision = collision_flags(model, entry.joints, q_path)
        metrics = episode_follow_metrics(
            pose_success=pose_success, collision=table | self_collision,
            position_error_m=position_error,
            orientation_error_rad=orientation_error)
        hard_feasible = not bool((table | self_collision).any())
        rank = follow_rank(metrics) if hard_feasible else invalid_rank
        records.append({
            "rank": rank, "candidate": candidate.copy(), "feasible": hard_feasible,
            "episode_success": bool(metrics["episode_success"]),
            "frame_indices": np.asarray(indices, dtype=int).copy(),
            "q": q_path.copy(), "success": pose_success.copy(),
            "position_error_m": position_error.copy(),
            "orientation_error_rad": orientation_error.copy(),
            "collision": (table | self_collision).copy(),
            "table_collision": table.copy(), "self_collision": self_collision.copy(),
            "joint_discontinuity": result.joint_discontinuity.copy(),
            "joint_limits_respected": bool(result.joint_limits_respected),
            "new_frames": len(indices), "reused_frames": 0,
        })
        print(
            f"[{label} {index + 1}/{len(candidates)}] safe={metrics['frame_coverage']:.1%} "
            f"pos_mm={1000*position_error.mean():.2f} "
            f"ori_deg={np.degrees(orientation_error.mean()):.2f}", flush=True)
    return records


def _evaluate_authoritative_mount_candidate(
    *, candidate: np.ndarray, targets: np.ndarray, target_quaternions: np.ndarray,
    time_s: np.ndarray, indices: np.ndarray, robot: str, entry, model_template=None,
) -> dict[str, object]:
    """Evaluate one promoted mount with the report-time strict solver contract."""
    import mujoco

    active = np.asarray(candidate, dtype=float)
    full = _expand_yaw_only_mounts(active) if active.shape == (4,) else active.copy()
    expected = np.arange(len(targets), dtype=int)
    if not np.array_equal(np.asarray(indices, dtype=int), expected):
        raise ValueError("authoritative evaluation requires the full chronological episode")
    timeline = np.asarray(time_s, dtype=float)
    if timeline.shape != (len(targets),) or np.any(np.diff(timeline) <= 0.0):
        raise ValueError("authoritative evaluation requires the real increasing timeline")
    rotation = mount_rotation_matrix(
        tilt_pitch_deg=float(full[3]), yaw_deg=float(full[4]), roll_deg=float(full[5]))
    invalid_rank = (-1.0, -1.0, -float("inf"), -float("inf"), -float("inf"), -float("inf"))
    if not tabletop_mount_feasible(full[:3], rotation):
        return {"rank": invalid_rank, "candidate": active.copy(), "feasible": False,
                "planner_type": "rolling_multibranch"}
    if model_template is None:
        model = build_model(robot, full[:3], float(full[3]), float(full[4]), float(full[5]))
    else:
        model, _ = model_template.apply(
            full[:3], float(full[3]), float(full[4]), float(full[5]))
    position_seed = solve_position_path(
        model, "strict_tracking_tcp", entry.joints, targets[:1],
        tolerance_m=1e-3, iterations=280, restarts=16)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in entry.joints]
    joint_ranges = np.asarray([model.jnt_range[joint_id] for joint_id in joint_ids])
    periodic = (joint_ranges[:, 1] - joint_ranges[:, 0]) >= (2.0 * np.pi - 1e-6)
    state_cache: dict[tuple[float, ...], bool] = {}
    edge_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], bool] = {}

    def state_collision_free(q: np.ndarray) -> bool:
        key = tuple(np.round(np.asarray(q, dtype=float), 7))
        if key not in state_cache:
            table, self_hit = collision_flags(model, entry.joints, np.asarray(q)[None, :])
            state_cache[key] = not bool(table[0] or self_hit[0])
        return state_cache[key]

    def transition_collision_free(previous_q: np.ndarray, candidate_q: np.ndarray) -> bool:
        first = tuple(np.round(np.asarray(previous_q, dtype=float), 7))
        second = tuple(np.round(np.asarray(candidate_q, dtype=float), 7))
        key = (first, second)
        if key not in edge_cache:
            delta = wrapped_joint_delta(candidate_q, previous_q, periodic)
            samples = max(2, int(np.ceil(np.max(np.abs(delta)) / np.deg2rad(5.0))) + 1)
            path = (np.asarray(previous_q)[None, :]
                    + np.linspace(0.0, 1.0, samples)[:, None] * delta[None, :])
            table, self_hit = collision_flags(model, entry.joints, path)
            edge_cache[key] = not bool(table.any() or self_hit.any())
        return edge_cache[key]

    result = solve_pose_path_multibranch(
        model, "strict_tracking_tcp", entry.joints, targets, target_quaternions,
        time_s=timeline, branch_candidates=8, horizon=12, beam_width=8,
        velocity_limit_rad_s=np.deg2rad(720.0), maximum_frame_jump_rad=np.deg2rad(25.0),
        position_tolerance_m=1e-3, orientation_tolerance_rad=np.deg2rad(1.5),
        iterations=240, initial_q=position_seed.q[0],
        candidate_collision_free=state_collision_free,
        transition_collision_free=transition_collision_free)
    table, self_hit = collision_flags(model, entry.joints, result.q)
    pose_success = ((result.position_error_m <= 1e-3)
                    & (result.orientation_error_rad <= np.deg2rad(1.5))
                    & ~result.joint_discontinuity)
    metrics = episode_follow_metrics(
        pose_success=pose_success, collision=table | self_hit,
        position_error_m=result.position_error_m,
        orientation_error_rad=result.orientation_error_rad)
    planner_failure = planner_failure_diagnostics(
        rolling_success=pose_success & ~table & ~self_hit,
        collision=table | self_hit,
        jump_violation=result.joint_discontinuity,
        recovery_mode=result.recovery_mode)
    hard_feasible = not bool((table | self_hit).any())
    return {
        "rank": follow_rank(metrics) if hard_feasible else invalid_rank,
        "candidate": active.copy(), "feasible": hard_feasible,
        "episode_success": bool(metrics["episode_success"]),
        "frame_indices": expected, "q": result.q.copy(), "success": pose_success,
        "position_error_m": result.position_error_m.copy(),
        "orientation_error_rad": result.orientation_error_rad.copy(),
        "collision": (table | self_hit).copy(), "table_collision": table.copy(),
        "self_collision": self_hit.copy(),
        "joint_discontinuity": result.joint_discontinuity.copy(),
        "joint_limits_respected": bool(result.joint_limits_respected),
        "planner_type": "rolling_multibranch", "timeline_used": True,
        "planner_failure_reason": planner_failure["per_frame"],
        "planner_failure_counts": planner_failure["counts"],
        "recovery_mode": result.recovery_mode.copy(),
        "new_frames": len(expected), "reused_frames": 0,
    }


def _record_is_hard_feasible(record: dict[str, object]) -> bool:
    if not bool(record["feasible"]):
        return False
    collision = record.get("collision")
    return collision is None or not bool(np.asarray(collision, dtype=bool).any())


def _layered_record_metrics(
    *,
    scope: str,
    candidate_id: int,
    pose_success: np.ndarray,
    position_error_m: np.ndarray,
    orientation_error_rad: np.ndarray,
    state_collision: np.ndarray,
    edge_collision: np.ndarray,
) -> dict[str, object]:
    """Finalize one traceable layered-IK stage without overstating windows."""
    if scope not in {"window", "full_episode"}:
        raise ValueError("scope must be window or full_episode")
    pose = np.asarray(pose_success, dtype=bool)
    position = np.asarray(position_error_m, dtype=float)
    orientation = np.asarray(orientation_error_rad, dtype=float)
    state = np.asarray(state_collision, dtype=bool)
    edge = np.asarray(edge_collision, dtype=bool)
    if not (pose.ndim == 1 and pose.shape == position.shape == orientation.shape
            == state.shape == edge.shape):
        raise ValueError("layered record arrays must have matching one-dimensional shapes")
    collision = state | edge
    metrics = episode_follow_metrics(
        pose_success=pose, collision=collision,
        position_error_m=position, orientation_error_rad=orientation)
    feasible = not bool(collision.any())
    rank = follow_rank(metrics)
    if scope == "window":
        rank = (0.0, *rank[1:])
    return {
        "scope": scope,
        "candidate_id": int(candidate_id),
        "rank": rank,
        "feasible": feasible,
        "episode_success": bool(
            scope == "full_episode" and feasible and metrics["episode_success"]),
        "frame_coverage": float(metrics["frame_coverage"]),
        "longest_failure_run_frames": int(metrics["longest_failure_run_frames"]),
        "state_collision_frames": int(state.sum()),
        "edge_collision_frames": int(edge.sum()),
        "collision": collision,
    }


def _evaluate_layered_model_path(
    *,
    model,
    joint_names: tuple[str, ...],
    candidate_id: int,
    targets: np.ndarray,
    target_quaternions: np.ndarray,
    time_s: np.ndarray,
    scope: str,
    candidates_per_frame: int,
    global_seed_count: int,
    iterations: int,
    velocity_limit_rad_s: float | np.ndarray,
    maximum_frame_jump_rad: float | np.ndarray,
    horizon: int = 12,
    beam_width: int = 8,
) -> dict[str, object]:
    """Evaluate one already-mounted real model with layered IK and swept edges."""
    import mujoco

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in joint_names]
    periodic = joint_periodic_mask(model, joint_ids)
    state_cache: dict[tuple[float, ...], bool] = {}
    edge_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], bool] = {}

    def state_collision_free(q: np.ndarray) -> bool:
        key = tuple(np.round(np.asarray(q, dtype=float), 7))
        if key not in state_cache:
            table, self_hit = collision_flags(
                model, joint_names, np.asarray(q, dtype=float)[None, :])
            state_cache[key] = not bool(table[0] or self_hit[0])
        return state_cache[key]

    def transition_collision_free(previous: np.ndarray, current: np.ndarray) -> bool:
        first = tuple(np.round(np.asarray(previous, dtype=float), 7))
        second = tuple(np.round(np.asarray(current, dtype=float), 7))
        key = (first, second)
        if key not in edge_cache:
            delta = wrapped_joint_delta(current, previous, periodic)
            samples = max(2, int(np.ceil(
                np.max(np.abs(delta)) / np.deg2rad(5.0))) + 1)
            q_path = (np.asarray(previous, dtype=float)[None, :]
                      + np.linspace(0.0, 1.0, samples)[:, None] * delta[None, :])
            table, self_hit = collision_flags(model, joint_names, q_path)
            edge_cache[key] = not bool(table.any() or self_hit.any())
        return edge_cache[key]

    result = solve_pose_path_layered(
        model, "strict_tracking_tcp", joint_names, targets, target_quaternions,
        time_s=time_s, candidates_per_frame=candidates_per_frame,
        global_seed_count=global_seed_count, horizon=horizon,
        beam_width=beam_width, iterations=iterations,
        velocity_limit_rad_s=velocity_limit_rad_s,
        maximum_frame_jump_rad=maximum_frame_jump_rad,
        candidate_collision_free=state_collision_free,
        transition_collision_free=transition_collision_free,
    )
    table, self_hit = collision_flags(model, joint_names, result.q)
    edge_collision = np.zeros(len(result.q), dtype=bool)
    for index in range(1, len(result.q)):
        edge_collision[index] = not transition_collision_free(
            result.q[index - 1], result.q[index])
    pose_success = (result.success & ~result.velocity_violation
                    & ~result.joint_discontinuity)
    record = _layered_record_metrics(
        scope=scope, candidate_id=candidate_id,
        pose_success=pose_success,
        position_error_m=result.position_error_m,
        orientation_error_rad=result.orientation_error_rad,
        state_collision=table | self_hit,
        edge_collision=edge_collision,
    )
    record.update({
        "q": result.q.copy(),
        "success": pose_success.copy(),
        "position_error_m": result.position_error_m.copy(),
        "orientation_error_rad": result.orientation_error_rad.copy(),
        "table_collision": table.copy(),
        "self_collision": self_hit.copy(),
        "edge_collision": edge_collision,
        "joint_discontinuity": result.joint_discontinuity.copy(),
        "velocity_violation": result.velocity_violation.copy(),
        "branch_count": result.branch_count.copy(),
        "chosen_branch_index": result.chosen_branch_index.copy(),
        "recovery_mode": result.recovery_mode.copy(),
        "joint_limit_margin": result.joint_limit_margin.copy(),
        "singularity_margin": result.singularity_margin.copy(),
        "joint_limits_respected": result.joint_limits_respected,
        "planner_type": "layered_receding_horizon",
        "timeline_used": True,
    })
    return record


def _top_records(records: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    feasible = [record for record in records if _record_is_hard_feasible(record)]
    return sorted(feasible, key=lambda record: record["rank"], reverse=True)[:count]


def _merge_evaluation_records(
    prior: dict[str, object], added: dict[str, object]
) -> dict[str, object]:
    """Merge disjoint fidelity results and recompute the combined rank."""
    prior_indices = np.asarray(prior["frame_indices"], dtype=int)
    added_indices = np.asarray(added["frame_indices"], dtype=int)
    if np.intersect1d(prior_indices, added_indices).size:
        raise ValueError("promotion frame sets overlap; existing IK work must be reused")
    indices = np.r_[prior_indices, added_indices]
    order = np.argsort(indices)

    def combined(name: str) -> np.ndarray:
        return np.concatenate((np.asarray(prior[name]), np.asarray(added[name])), axis=0)[order]

    success = combined("success").astype(bool)
    position_error = combined("position_error_m").astype(float)
    orientation_error = combined("orientation_error_rad").astype(float)
    collision = combined("collision").astype(bool)
    metrics = episode_follow_metrics(
        pose_success=success, collision=collision,
        position_error_m=position_error, orientation_error_rad=orientation_error)
    merged = dict(added)
    merged.update({
        "rank": follow_rank(metrics),
        "feasible": (_record_is_hard_feasible(prior)
                     and _record_is_hard_feasible(added)
                     and not bool(collision.any())),
        "episode_success": bool(metrics["episode_success"]),
        "frame_indices": indices[order],
        "q": combined("q"),
        "success": success,
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "collision": collision,
        "new_frames": len(added_indices),
        "reused_frames": len(prior_indices),
    })
    return merged


def _inherited_seed(record: dict[str, object], first_new_frame: int) -> np.ndarray | None:
    if "frame_indices" not in record:
        return None
    indices = np.asarray(record["frame_indices"], dtype=int)
    success = np.asarray(record["success"], dtype=bool)
    if not np.any(success):
        return None
    successful_positions = np.flatnonzero(success)
    selected = min(successful_positions,
                   key=lambda pos: (abs(int(indices[pos]) - first_new_frame), int(indices[pos])))
    return np.asarray(record["q"])[selected].copy()


def _should_use_incumbent_replay(
    *, primary_episode_success: bool, primary_joint_limits: bool,
    incumbent_known_pass: bool,
) -> bool:
    return incumbent_known_pass and not (primary_episode_success and primary_joint_limits)


def _best_first_mount_candidate(
    *, args: argparse.Namespace, targets: np.ndarray, target_quaternions: np.ndarray,
    time_s: np.ndarray | None = None,
    robot: str, entry, lower: np.ndarray, upper: np.ndarray,
    warm_starts: np.ndarray, evaluate_fn=_evaluate_mount_candidates,
    authoritative_evaluate_fn=None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run a bounded rank-first search and return its best dense candidate."""
    budget = resolved_best_first_budget(args)
    rank_indices = representative_frame_indices(
        targets, target_quaternions, requested=budget.rank_frames)
    rank_windows = representative_frame_windows(
        targets, target_quaternions, requested_windows=3,
        window_length=max(2, budget.rank_frames // 3))
    medium_indices = _uniform_indices(len(targets), budget.medium_frames)
    full_indices = np.arange(len(targets), dtype=int)
    timeline = (np.arange(len(targets), dtype=float) if time_s is None
                else np.asarray(time_s, dtype=float))
    started = time.perf_counter()
    model_template = build_mount_model_template(robot) if evaluate_fn is _evaluate_mount_candidates else None

    def evaluate(**kwargs):
        if model_template is not None:
            kwargs["model_template"] = model_template
        candidates = kwargs.pop("candidates")
        return _evaluate_search_candidates(
            evaluate_fn, candidates=candidates, **kwargs)

    broad = global_mount_candidates(
        lower, upper, warm_starts, count=budget.global_candidates, seed=750)
    global_records = evaluate(
        candidates=broad, targets=targets, target_quaternions=target_quaternions,
        indices=rank_indices, index_windows=rank_windows,
        robot=robot, entry=entry, iterations=100,
        restarts=5, label="global-rank")
    feasible_global = [record for record in global_records if _record_is_hard_feasible(record)]
    if len(feasible_global) < budget.retained_regions:
        raise RuntimeError(
            f"best-first search needs {budget.retained_regions} feasible global regions; "
            f"found {len(feasible_global)}")
    feasible_mounts = np.asarray([record["candidate"] for record in feasible_global])
    feasible_scores = [record["rank"] for record in feasible_global]
    selected = select_diverse_regions(
        feasible_mounts, feasible_scores, lower=lower, upper=upper,
        count=budget.retained_regions,
        shortlist=min(24, len(feasible_global)))
    centers = feasible_mounts[selected]
    local_candidates, _, _ = generate_local_population(
        centers, lower, upper, per_region=budget.local_per_region, seed=751)
    local_records = evaluate(
        candidates=local_candidates, targets=targets,
        target_quaternions=target_quaternions, indices=rank_indices,
        index_windows=rank_windows,
        robot=robot, entry=entry, iterations=100, restarts=2, label="local")

    frontier = sorted(
        [record for record in feasible_global + local_records if _record_is_hard_feasible(record)],
        key=lambda record: record["rank"], reverse=True)
    dense_records: list[dict[str, object]] = []
    new_frame_solves = sum(int(record.get("new_frames", 0)) for record in global_records + local_records)
    reused_frame_results = 0
    fallback_used = False
    dense_limit = budget.dense_before_success + budget.dense_fallback
    cursor = 0

    def promote_medium(record: dict[str, object]) -> dict[str, object] | None:
        nonlocal new_frame_solves, reused_frame_results
        mount = np.asarray(record["candidate"], dtype=float)[None, :]
        medium_requested = medium_indices
        if "frame_indices" in record:
            medium_requested = np.setdiff1d(
                medium_indices, np.asarray(record["frame_indices"], dtype=int))
        if len(medium_requested) == 0:
            medium = record
        else:
            medium_added = evaluate(
                candidates=mount, targets=targets, target_quaternions=target_quaternions,
                indices=medium_requested, robot=robot, entry=entry, iterations=140,
                restarts=3, label="medium",
                initial_qs=[_inherited_seed(record, int(medium_requested[0]))])[0]
            if not _record_is_hard_feasible(medium_added):
                return None
            medium = (_merge_evaluation_records(record, medium_added)
                      if "frame_indices" in record and "frame_indices" in medium_added
                      else medium_added)
        if not _record_is_hard_feasible(medium):
            return None
        new_frame_solves += int(medium.get("new_frames", len(medium_requested)))
        reused_frame_results += int(medium.get("reused_frames", 0))
        return medium

    def promote_dense(medium: dict[str, object]) -> dict[str, object] | None:
        nonlocal new_frame_solves, reused_frame_results
        mount = np.asarray(medium["candidate"], dtype=float)[None, :]
        dense_requested = _dense_evaluation_indices(
            len(full_indices), previously_ranked=np.asarray(
                medium.get("frame_indices", ()), dtype=int))
        strict_fn = authoritative_evaluate_fn
        if strict_fn is None and evaluate_fn is _evaluate_mount_candidates:
            strict_fn = _evaluate_authoritative_mount_candidate
        if strict_fn is None:
            dense = evaluate(
                candidates=mount, targets=targets, target_quaternions=target_quaternions,
                indices=dense_requested, robot=robot, entry=entry, iterations=240,
                restarts=5, label="dense", initial_qs=[None])[0]
        else:
            dense = strict_fn(
                candidate=mount[0], targets=targets,
                target_quaternions=target_quaternions, time_s=timeline,
                indices=dense_requested, robot=robot, entry=entry,
                model_template=model_template)
        if not _record_is_hard_feasible(dense):
            return None
        new_frame_solves += int(dense.get("new_frames", len(dense_requested)))
        reused_frame_results += 0
        return dense

    first_success = False
    while cursor < len(frontier) and len(dense_records) < dense_limit and not first_success:
        medium = promote_medium(frontier[cursor])
        cursor += 1
        if medium is None:
            continue
        dense = promote_dense(medium)
        if dense is None:
            continue
        dense_records.append(dense)
        first_success = bool(dense.get("episode_success", False))
        if not first_success and len(dense_records) >= budget.dense_before_success:
            fallback_used = True

    post_success_medium: list[dict[str, object]] = []
    if first_success:
        while (cursor < len(frontier)
               and len(post_success_medium) < budget.post_success_expansions):
            medium = promote_medium(frontier[cursor])
            cursor += 1
            if medium is not None:
                post_success_medium.append(medium)
        challengers = sorted(
            post_success_medium, key=lambda record: record["rank"], reverse=True
        )[:budget.post_success_dense]
        for medium in challengers:
            dense = promote_dense(medium)
            if dense is not None:
                dense_records.append(dense)

    candidates_for_final = dense_records or frontier[:1]
    if not candidates_for_final:
        raise RuntimeError("best-first search produced no physically feasible finalist")
    winner = max(candidates_for_final, key=lambda record: record["rank"])
    stop_reason = (
        "post_success_expansion_budget" if first_success
        else "dense_fallback_exhausted")
    audit = {
        "search_policy": "best-first",
        "near_global_claim": "budget_bounded_no_better_candidate_observed",
        "stop_reason": stop_reason,
        "search_budget": vars(budget),
        "global_candidate_count": len(broad),
        "local_candidate_count": len(local_candidates),
        "dense_candidate_count": len(dense_records),
        "post_success_expansions": len(post_success_medium),
        "post_success_dense_candidates": (
            min(len(post_success_medium), budget.post_success_dense) if first_success else 0),
        "_authoritative_dense_record": winner,
        "verified_region_count": budget.retained_regions,
        "fallback_used": fallback_used,
        "new_frame_solves": new_frame_solves,
        "reused_frame_results": reused_frame_results,
        "elapsed_s": time.perf_counter() - started,
    }
    winner_candidate = np.asarray(winner["candidate"], dtype=float)
    if winner_candidate.shape[-1] == 4:
        winner_candidate = _expand_yaw_only_mounts(winner_candidate)
    return winner_candidate, audit


def _legacy_mount_candidate(
    *, budget: dict[str, int], targets: np.ndarray, target_quaternions: np.ndarray,
    screen_indices: np.ndarray, medium_indices: np.ndarray, full_indices: np.ndarray,
    robot: str, entry, lower: np.ndarray, upper: np.ndarray,
    warm_starts: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    def evaluate(*, candidates: np.ndarray, **kwargs):
        return _evaluate_search_candidates(
            _evaluate_mount_candidates, candidates=candidates, **kwargs)

    broad = global_mount_candidates(
        lower, upper, warm_starts, count=budget["global"], seed=750)
    global_records = evaluate(
        candidates=broad, targets=targets, target_quaternions=target_quaternions,
        indices=screen_indices, robot=robot, entry=entry, iterations=120,
        restarts=4, label="global")
    global_ranks = [record["rank"] for record in global_records]
    center_indices = diverse_region_indices(
        broad, global_ranks, shortlist=min(budget["global_retain"], len(broad)),
        regions=min(budget["local_regions"], len(broad)), lower=lower, upper=upper,
        minimum_distance=0.12)
    region_centers = broad[center_indices]
    local_radius = (np.asarray((0.08, 0.08, 0.06, 20.0)) if lower.shape == (4,)
                    else np.asarray((0.08, 0.08, 0.06, 15.0, 20.0, 12.0)))
    local_candidates = local_mount_candidates(
        region_centers, lower, upper, per_center=budget["local_per_region"],
        radii=local_radius, seed=751)
    local_records = evaluate(
        candidates=local_candidates, targets=targets,
        target_quaternions=target_quaternions, indices=screen_indices,
        robot=robot, entry=entry, iterations=140, restarts=6, label="local")
    screened_pool = _top_records(global_records, budget["global_retain"]) + local_records
    medium_candidates = np.asarray([
        record["candidate"] for record in _top_records(screened_pool, budget["medium_retain"])
    ])
    medium_records = evaluate(
        candidates=medium_candidates, targets=targets,
        target_quaternions=target_quaternions, indices=medium_indices,
        robot=robot, entry=entry, iterations=160, restarts=8, label="medium")
    strict_candidates = np.asarray([
        record["candidate"] for record in _top_records(medium_records, budget["strict_retain"])
    ])
    strict_records = evaluate(
        candidates=strict_candidates, targets=targets,
        target_quaternions=target_quaternions, indices=full_indices,
        robot=robot, entry=entry, iterations=240, restarts=12, label="strict")
    final_center_records = _top_records(strict_records, budget["final_centers"])
    final_centers = np.asarray([record["candidate"] for record in final_center_records])
    final_radius = (np.asarray((0.02, 0.02, 0.015, 3.0)) if lower.shape == (4,)
                    else np.asarray((0.02, 0.02, 0.015, 3.0, 3.0, 2.0)))
    final_candidates = local_mount_candidates(
        final_centers, lower, upper, per_center=budget["final_per_center"],
        radii=final_radius, seed=752)
    final_screen_records = evaluate(
        candidates=final_candidates, targets=targets,
        target_quaternions=target_quaternions, indices=medium_indices,
        robot=robot, entry=entry, iterations=180, restarts=8, label="micro")
    final_full_candidates = np.asarray([
        record["candidate"] for record in _top_records(final_screen_records, budget["final_retain"])
    ])
    final_full_records = evaluate(
        candidates=final_full_candidates, targets=targets,
        target_quaternions=target_quaternions, indices=full_indices,
        robot=robot, entry=entry, iterations=280, restarts=16, label="final")
    finalists = _top_records(final_full_records, 1)
    if not finalists:
        raise RuntimeError("hierarchical search produced no physically feasible finalist")
    audit = {
        "search_policy": "legacy",
        "mount_search_candidates": int(
            len(broad) + len(local_candidates) + len(medium_candidates)
            + len(strict_candidates) + len(final_candidates) + len(final_full_candidates)),
        "candidate_policy": "hierarchical broad Sobol + diverse multi-basin local + strict full-episode + micro-refinement",
        "search_stages": {
            "global": len(broad), "global_retain": budget["global_retain"],
            "local_regions": len(region_centers), "local": len(local_candidates),
            "medium": len(medium_candidates), "strict": len(strict_candidates),
            "final_centers": len(final_centers), "micro": len(final_candidates),
            "final_full": len(final_full_candidates),
        },
        "pose_finalists": len(final_full_candidates),
    }
    winner = np.asarray(finalists[0]["candidate"], dtype=float)
    if winner.shape[-1] == 4:
        winner = _expand_yaw_only_mounts(winner)
    return winner, audit


def main() -> None:
    args = build_parser().parse_args()
    targets, target_quaternions, time_s, trajectory_source = target_poses_for(
        args.domain, args.task, split="validation" if args.domain == "local" else "test",
        episode_artifact=args.episode_artifact)
    try:
        if args.incumbent_json is not None:
            incumbent_path = args.incumbent_json if args.incumbent_json.is_absolute() else ROOT / args.incumbent_json
            old = json.loads(incumbent_path.read_text(encoding="utf-8"))["robots"][args.robot]["per_task"][args.task]
        else:
            old = optimization(args.domain, args.robot, args.task)
        existing = np.r_[np.asarray(old["base_xyz_m"], dtype=float), float(old["tilt_deg"]),
                         float(old.get("yaw_deg", 0.0)), float(old.get("roll_deg", 0.0))]
    except (FileNotFoundError, KeyError):
        # A repaired experiment must not require stale dense-search artifacts.
        existing = np.asarray((0.0, -0.15, 0.335, 0.0, 0.0, 0.0))
    installation = first_version_bounds(0)
    lower, upper = installation.lower, installation.upper
    warm_starts = [existing, np.asarray((0.0, -0.15, 0.335, 0.0, 0.0, 0.0))]
    protected_incumbent: np.ndarray | None = None
    strict_cache = ((args.output if args.output.is_absolute() else ROOT / args.output).with_suffix(".json")
                    if args.output is not None else
                    ROOT / "videos/single_arm/strict_cache" / args.domain / args.robot / f"{args.task}.json")
    if strict_cache.is_file():
        try:
            cached = json.loads(strict_cache.read_text(encoding="utf-8"))
            cached_mount = np.r_[np.asarray(cached["base_xyz_m"], dtype=float),
                                 float(cached["tilt_deg"]), float(cached.get("yaw_deg", 0.0)),
                                 float(cached.get("roll_deg", 0.0))]
            warm_starts.append(cached_mount)
            if cached.get("status") == "pass":
                protected_incumbent = cached_mount
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    lower = _compress_yaw_only_mounts(lower)
    upper = _compress_yaw_only_mounts(upper)
    warm_starts = list(_compress_yaw_only_mounts(np.asarray(warm_starts)))
    if protected_incumbent is not None and not np.allclose(
            protected_incumbent[[3, 5]], 0.0, atol=1e-12):
        protected_incumbent = None
    unique_warm_starts = []
    for candidate in warm_starts:
        if np.all(candidate >= lower) and np.all(candidate <= upper) and not any(
                np.allclose(candidate, prior, atol=1e-12) for prior in unique_warm_starts):
            unique_warm_starts.append(candidate)
    budget = resolved_hierarchy_budget(args)
    global_count = budget["global"]
    strict_retain = budget["strict_retain"]
    numeric_counts = tuple(budget.values())
    if any(value < 1 for value in numeric_counts) or global_count < 2:
        raise ValueError("all hierarchy budgets must be positive and global candidates >= 2")
    entry = MODELS[args.robot]
    screen_indices = _uniform_indices(len(targets), args.screen_frames)
    medium_indices = _uniform_indices(len(targets), args.medium_frames)
    full_indices = np.arange(len(targets), dtype=int)

    policy_audit: dict[str, object] = {"search_policy": "legacy"}
    if args.search_policy == "best-first":
        candidate, policy_audit = _best_first_mount_candidate(
            args=args, targets=targets, target_quaternions=target_quaternions,
            time_s=time_s,
            robot=args.robot, entry=entry, lower=lower, upper=upper,
            warm_starts=np.asarray(unique_warm_starts))
    else:
        candidate, policy_audit = _legacy_mount_candidate(
            budget=budget, targets=targets, target_quaternions=target_quaternions,
            screen_indices=screen_indices, medium_indices=medium_indices,
            full_indices=full_indices, robot=args.robot, entry=entry,
            lower=lower, upper=upper, warm_starts=np.asarray(unique_warm_starts))
    selected_pose_index = 0
    selected_targets, selected_target_quaternions = targets, target_quaternions
    selected_model = build_model(args.robot, candidate[:3], float(candidate[3]), float(candidate[4]), float(candidate[5]))
    authoritative = policy_audit.pop("_authoritative_dense_record", None)
    actual_planner_type = "legacy_pose_path"
    timeline_used = False
    planner_failure_counts: dict[str, int] = {}
    planner_failure_reason = np.full(len(targets), "unavailable", dtype=str)
    if authoritative is not None and "q" in authoritative:
        from types import SimpleNamespace
        q_path = np.asarray(authoritative["q"]).copy()
        success = np.asarray(authoritative["success"], dtype=bool).copy()
        position_error = np.asarray(authoritative["position_error_m"], dtype=float).copy()
        orientation_error = np.asarray(authoritative["orientation_error_rad"], dtype=float).copy()
        table = np.asarray(authoritative["table_collision"], dtype=bool).copy()
        self_collision = np.asarray(authoritative["self_collision"], dtype=bool).copy()
        reached_xyz, reached_quaternion, _, _ = evaluate_q_path(
            selected_model, entry.joints, q_path, selected_targets, selected_target_quaternions)
        result = SimpleNamespace(
            q=q_path, success=success,
            joint_discontinuity=np.asarray(authoritative["joint_discontinuity"], dtype=bool),
            joint_limits_respected=bool(authoritative["joint_limits_respected"]))
        actual_planner_type = str(authoritative.get("planner_type", "legacy_pose_path"))
        timeline_used = bool(authoritative.get("timeline_used", False))
        planner_failure_counts = dict(authoritative.get("planner_failure_counts", {}))
        planner_failure_reason = np.asarray(
            authoritative.get("planner_failure_reason", planner_failure_reason)).astype(str)
        policy_audit["final_evaluation_source"] = "authoritative_dense_record"
    else:
        position_seed = solve_position_path(
            selected_model, "strict_tracking_tcp", entry.joints, targets[:1],
            tolerance_m=1e-3, iterations=280, restarts=16)
        result = solve_pose_path(
            selected_model, "strict_tracking_tcp", entry.joints, selected_targets,
            selected_target_quaternions, position_tolerance_m=1e-3,
            orientation_tolerance_rad=np.deg2rad(1.5), iterations=280,
            restarts=16, initial_q=position_seed.q[0])
        table, self_collision = collision_flags(selected_model, entry.joints, result.q)
        q_path = hold_invalid_frames(result.q, ~result.success | table | self_collision)
        reached_xyz, reached_quaternion, position_error, orientation_error = evaluate_q_path(
            selected_model, entry.joints, q_path, selected_targets, selected_target_quaternions)
        success = (position_error <= 1e-3) & (orientation_error <= np.deg2rad(1.5))
        table, self_collision = collision_flags(selected_model, entry.joints, q_path)
        policy_audit["final_evaluation_source"] = "legacy_full_replay"
    follow = episode_follow_metrics(
        pose_success=success, collision=table | self_collision,
        position_error_m=position_error,
        orientation_error_rad=orientation_error)
    policy_audit["incumbent_fallback_used"] = False
    if (args.search_policy == "best-first" and protected_incumbent is not None
            and not np.allclose(candidate, protected_incumbent, atol=1e-12)
            and _should_use_incumbent_replay(
                primary_episode_success=bool(follow["episode_success"]),
                primary_joint_limits=bool(result.joint_limits_respected),
                incumbent_known_pass=True)):
        incumbent_model = build_model(
            args.robot, protected_incumbent[:3], float(protected_incumbent[3]),
            float(protected_incumbent[4]), float(protected_incumbent[5]))
        incumbent_seed = solve_position_path(
            incumbent_model, "strict_tracking_tcp", entry.joints, targets[:1],
            tolerance_m=1e-3, iterations=280, restarts=16)
        incumbent_result = solve_pose_path(
            incumbent_model, "strict_tracking_tcp", entry.joints, selected_targets,
            selected_target_quaternions, position_tolerance_m=1e-3,
            orientation_tolerance_rad=np.deg2rad(1.5), iterations=280,
            restarts=16, initial_q=incumbent_seed.q[0])
        incumbent_table, incumbent_self = collision_flags(
            incumbent_model, entry.joints, incumbent_result.q)
        incumbent_q = hold_invalid_frames(
            incumbent_result.q,
            ~incumbent_result.success | incumbent_table | incumbent_self)
        incumbent_xyz, incumbent_quat, incumbent_position, incumbent_orientation = evaluate_q_path(
            incumbent_model, entry.joints, incumbent_q, selected_targets,
            selected_target_quaternions)
        incumbent_success = ((incumbent_position <= 1e-3)
                             & (incumbent_orientation <= np.deg2rad(1.5)))
        incumbent_table, incumbent_self = collision_flags(
            incumbent_model, entry.joints, incumbent_q)
        incumbent_follow = episode_follow_metrics(
            pose_success=incumbent_success,
            collision=incumbent_table | incumbent_self,
            position_error_m=incumbent_position,
            orientation_error_rad=incumbent_orientation)
        if incumbent_follow["episode_success"] and incumbent_result.joint_limits_respected:
            candidate = protected_incumbent.copy()
            selected_model, result, q_path = incumbent_model, incumbent_result, incumbent_q
            reached_xyz, reached_quaternion = incumbent_xyz, incumbent_quat
            position_error, orientation_error = incumbent_position, incumbent_orientation
            success, table, self_collision = incumbent_success, incumbent_table, incumbent_self
            follow = incumbent_follow
            policy_audit["incumbent_fallback_used"] = True
    output = args.output or (
        ROOT / "videos/single_arm/strict_cache" / args.domain / args.robot / f"{args.task}.npz")
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    failure = failure_reason_diagnostics(
        position_error_m=position_error, orientation_error_rad=orientation_error,
        table_collision=table, self_collision=self_collision,
        joint_discontinuity=result.joint_discontinuity)
    np.savez_compressed(output, q=q_path, target_xyz_m=selected_targets, target_quaternion_wxyz=selected_target_quaternions,
                        time_s=time_s, reached_xyz_m=reached_xyz,
                        reached_quaternion_wxyz=reached_quaternion,
                        position_error_m=position_error, success=success,
                        orientation_error_rad=orientation_error,
                        joint_discontinuity=result.joint_discontinuity,
                        planner_failure_reason=planner_failure_reason,
                        failure_reason=failure["primary_per_frame"],
                        table_collision=table, self_collision=self_collision,
                        base_xyz_m=candidate[:3], tilt_deg=np.asarray(candidate[3]),
                        yaw_deg=np.asarray(candidate[4]), roll_deg=np.asarray(candidate[5]),
                        joint_names=np.asarray(entry.joints))
    safe = success & ~table & ~self_collision
    audit = {
        "domain": args.domain, "robot": args.robot, "task": args.task,
        "source_model": str(entry.path), "trajectory_source": trajectory_source, "frames": len(targets),
        "mount_search_candidates": int(policy_audit.get(
            "mount_search_candidates",
            int(policy_audit.get("global_candidate_count", 0))
            + int(policy_audit.get("local_candidate_count", 0))
            + int(policy_audit.get("dense_candidate_count", 0)))),
        "screen_frames": len(screen_indices), "medium_frames": len(medium_indices),
        "candidate_policy": policy_audit.get(
            "candidate_policy", "best-first rank + bounded dense promotion + final replay"),
        "mount_selection_priority": "whole-episode success > frame coverage > shortest failure run > collision frames > position RMSE > orientation RMSE",
        "actual_planner_type": actual_planner_type,
        "timeline_used": timeline_used,
        "planner_failure_counts": planner_failure_counts,
        "search_stages": policy_audit.get("search_stages", {
            "global": policy_audit.get("global_candidate_count", 0),
            "local": policy_audit.get("local_candidate_count", 0),
            "medium": policy_audit.get("dense_candidate_count", 0),
            "dense": policy_audit.get("dense_candidate_count", 0),
        }),
        "pose_finalists": policy_audit.get("pose_finalists", 1),
        "selected_pose_finalist": selected_pose_index,
        "episode_success": follow["episode_success"],
        "frame_coverage": follow["frame_coverage"],
        "longest_failure_run_frames": follow["longest_failure_run_frames"],
        "failure_reasons": {
            "primary_frame_counts": failure["primary_frame_counts"],
            "affected_frame_counts": failure["affected_frame_counts"],
        },
        "position_error_m": {"mean": float(position_error.mean()),
                             "p95": float(np.quantile(position_error, .95)),
                             "max": float(position_error.max())},
        "orientation_error_deg": {"mean": float(np.degrees(orientation_error.mean())),
                                  "p95": float(np.degrees(np.quantile(orientation_error, .95))),
                                  "max": float(np.degrees(orientation_error.max()))},
        "joint_limits_respected": result.joint_limits_respected,
        "table_collision_frames": int(table.sum()), "self_collision_frames": int(self_collision.sum()),
        "base_xyz_m": candidate[:3].tolist(), "tilt_deg": float(candidate[3]),
        "yaw_deg": float(candidate[4]), "roll_deg": float(candidate[5]),
        "cache": str(output.relative_to(ROOT)).replace("\\", "/"),
        "input_fingerprint": args.input_fingerprint,
        "status": "pass" if follow["episode_success"] and result.joint_limits_respected else "fail",
        **policy_audit,
    }
    if args.search_policy == "best-first":
        audit["screen_frames"] = resolved_best_first_budget(args).rank_frames
        audit["medium_frames"] = min(
            len(targets), resolved_best_first_budget(args).medium_frames)
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
