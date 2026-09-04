"""Build and validate evidence for the multitask fixed-time mount study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from factory_bimanual.mount_topology import (
    MountTopologyConfig,
    MuJoCoMountTopologyChecker,
)
from factory_bimanual.mujoco_collision_adapter import (
    MuJoCoPairedCollisionChecker,
)
from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig,
    MuJoCoCandidateGenerator,
)
from factory_bimanual.multitask_fixed_time_study import (
    SHARD_SCHEMA,
    STUDY_MODES,
    discover_dual_hand_trajectories,
    validate_shard,
)
from factory_bimanual.robot_contracts import (
    ROBOT_CONTRACTS,
    robot_geometry_sha256,
)
from factory_bimanual.scene_builder import build_same_model_scene
from scripts import render_factory_dual_piperx_fixed_time as fixed_runner
from scripts import run_piperx_multitask_fixed_time_mount_study as search_runner
from scripts.render_factory_dual_xarm6_se3_follow import audit_bimanual_collisions
from scripts.search_fold_box_piperx_paired_mount import _scene_mount_kwargs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/piperx_multitask_fixed_time_mount_study"
BUNDLE_SCHEMA = "piperx-multitask-fixed-time-bundle-v1"
FORMAL_SOLVER_PROTOCOL = "piperx-fixed-time-paired-topology-safe-recovery-v3"


def _piperx_collision_checker_kwargs():
    """Return the single collision contract shared with the final audit."""
    return {"transition_steps": 5, "clearance_margin_m": .015}


class _ConjunctivePairChecker:
    """Apply every paired state/transition safety contract as a hard gate."""

    def __init__(self, *checkers):
        if not checkers:
            raise ValueError("at least one paired checker is required")
        self.checkers = tuple(checkers)

    def state(self, left, right):
        return SimpleNamespace(valid=all(
            checker.state(left, right).valid for checker in self.checkers))

    def transition(self, previous, current):
        return SimpleNamespace(valid=all(
            checker.transition(previous, current).valid
            for checker in self.checkers))


def build_shard_arrays(*, mode, source_time_s, qpos, position_error_m,
                       orientation_error_rad, collision, edge_collision,
                       topology_valid, left_source_valid, right_source_valid,
                       velocity_rad_s,
                       acceleration_rad_s2):
    """Assemble exact-source-time arrays while keeping dynamics separate."""
    if mode not in STUDY_MODES:
        raise ValueError("unsupported fixed-time mount mode")
    source_time = np.asarray(source_time_s, dtype=float)
    qpos = np.asarray(qpos, dtype=float)
    position = np.asarray(position_error_m, dtype=float)
    orientation = np.asarray(orientation_error_rad, dtype=float)
    left_source_valid = np.asarray(left_source_valid, dtype=bool)
    right_source_valid = np.asarray(right_source_valid, dtype=bool)
    count = len(source_time)
    if source_time.shape != (count,) or not np.all(np.diff(source_time) > 0):
        raise ValueError("source timestamps must be strictly increasing")
    if qpos.ndim != 2 or qpos.shape[0] != count:
        raise ValueError("qpos must have one row per source timestamp")
    if position.shape != (count, 2) or orientation.shape != (count, 2):
        raise ValueError("pose errors must have shape (frames, 2)")
    if (left_source_valid.shape != (count,)
            or right_source_valid.shape != (count,)):
        raise ValueError("source validity masks must have one row per timestamp")
    left = (left_source_valid
            & (position[:, 0] <= 0.001 + 1e-12)
            & (orientation[:, 0] <= np.deg2rad(0.5) + 1e-12))
    right = (right_source_valid
             & (position[:, 1] <= 0.001 + 1e-12)
             & (orientation[:, 1] <= np.deg2rad(0.5) + 1e-12))
    payload = {
        "schema": SHARD_SCHEMA,
        "mode": mode,
        "source_time_s": source_time.copy(),
        "fixed_time_s": source_time.copy(),
        "retiming_applied": False,
        "qpos": qpos.copy(),
        "left_source_valid": left_source_valid.copy(),
        "right_source_valid": right_source_valid.copy(),
        "left_accept": left,
        "right_accept": right,
        "both_accept": left & right,
        "position_error_m": position.copy(),
        "orientation_error_rad": orientation.copy(),
        "collision": np.asarray(collision, dtype=bool).copy(),
        "edge_collision": np.asarray(edge_collision, dtype=bool).copy(),
        "topology_valid": np.asarray(topology_valid, dtype=bool).copy(),
        "velocity_rad_s": np.asarray(velocity_rad_s, dtype=float).copy(),
        "acceleration_rad_s2": np.asarray(
            acceleration_rad_s2, dtype=float).copy(),
    }
    aggregate_shard(payload)
    return payload


def _longest_false_run(values) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_fingerprint(spec, mode, state, *, source_prefix=None) -> str:
    """Bind a formal shard to every input that can change its solution."""
    payload = {
        "schema": "piperx-fixed-time-formal-input-v1",
        "solver_protocol": FORMAL_SOLVER_PROTOCOL,
        "source_sha256": spec.source_sha256,
        "source_prefix": source_prefix,
        "mode": mode,
        "mount": state.get("selected_mount"),
        "search_job_fingerprint": state.get("job_fingerprint"),
        "target_contract": search_runner._target_contract(spec),
        "robot_geometry_sha256": robot_geometry_sha256("piperx"),
        "table_height_m": fixed_runner.TABLE_HEIGHT_M,
        "collision": _piperx_collision_checker_kwargs(),
        "topology_transition_steps": 5,
        "branch_guard_rad": 0.30,
        "position_tolerance_m": 0.001,
        "orientation_tolerance_deg": 0.5,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_search_job_fingerprint(spec, mode, specs):
    recommended = search_runner.load_recommended_config().mounts[spec.family.key]
    representative = search_runner.family_representative_spec(
        specs, spec, recommended.source_take)
    return search_runner._job_fingerprint(
        spec, mode, search_runner.STUDY_SEARCH_CONFIG,
        search_runner._target_contract(spec),
        dependency_source_sha256=(
            representative.source_sha256 if mode == "baseline" else None))


def _formal_summary_reusable(path: Path, expected_fingerprint: str) -> bool:
    """Return whether a cached shard was produced by the current solver."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        summary.get("schema") == "piperx-multitask-fixed-time-summary-v1"
        and summary.get("solver_protocol") == FORMAL_SOLVER_PROTOCOL
        and summary.get("formal_fingerprint") == expected_fingerprint
    )


def _joint_derivatives(model, qpos, time_s):
    addresses = []
    for side in ("left", "right"):
        for name in ROBOT_CONTRACTS["piperx"].prefixed_joint_names(side):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"missing PiperX joint {name}")
            addresses.append(model.jnt_qposadr[joint_id])
    q = np.asarray(qpos, dtype=float)[:, addresses]
    time_s = np.asarray(time_s, dtype=float)
    edge_order = 2 if len(time_s) >= 3 else 1
    velocity = np.gradient(q, time_s, axis=0, edge_order=edge_order)
    acceleration = np.gradient(
        velocity, time_s, axis=0, edge_order=edge_order)
    return velocity, acceleration


def _dynamics_summary(payload):
    velocity = np.asarray(payload["velocity_rad_s"], dtype=float)
    acceleration = np.asarray(payload["acceleration_rad_s2"], dtype=float)
    maximum_velocity = float(np.max(np.abs(velocity)))
    maximum_acceleration = float(np.max(np.abs(acceleration)))
    return {
        "maximum_velocity_rad_s": maximum_velocity,
        "maximum_acceleration_rad_s2": maximum_acceleration,
        "limits_passed": bool(
            maximum_velocity <= 1.0 + 1e-12
            and maximum_acceleration <= 4.0 + 1e-12),
    }


def _select_collision_safe_pair(candidate_lists, *, previous, checker,
                                branch_guard_rad):
    """Choose a connected candidate pair that passes state and edge gates."""
    left = list(candidate_lists.get("left", ()))
    right = list(candidate_lists.get("right", ()))
    if previous is not None:
        left = [item for item in left if np.max(np.abs(
            item.q - previous[0])) <= branch_guard_rad + 1e-12]
        right = [item for item in right if np.max(np.abs(
            item.q - previous[1])) <= branch_guard_rad + 1e-12]
    pairs = []
    for left_item in left:
        for right_item in right:
            current = (left_item.q, right_item.q)
            if not checker.state(*current).valid:
                continue
            if (previous is not None
                    and not checker.transition(previous, current).valid):
                continue
            delta = (0.0 if previous is None else max(
                float(np.max(np.abs(left_item.q - previous[0]))),
                float(np.max(np.abs(right_item.q - previous[1])))))
            pairs.append((
                delta,
                float(left_item.pose_cost + right_item.pose_cost),
                -(float(left_item.joint_limit_margin_rad)
                  + float(right_item.joint_limit_margin_rad)),
                left_item, right_item,
            ))
    if not pairs:
        return None
    selected = min(pairs, key=lambda item: item[:3])
    return selected[-2], selected[-1]


def _select_safe_recovery_step(candidate_lists, *, previous, checker,
                               maximum_step_rad):
    """Take one bounded collision-free step toward the nearest safe pair."""
    maximum_step = float(maximum_step_rad)
    if maximum_step <= 0.0:
        raise ValueError("maximum_step_rad must be positive")
    proposals = []
    for left in candidate_lists.get("left", ()):
        for right in candidate_lists.get("right", ()):
            target = (np.asarray(left.q, float), np.asarray(right.q, float))
            if not checker.state(*target).valid:
                continue
            trial = tuple(old + np.clip(new - old, -maximum_step, maximum_step)
                          for old, new in zip(previous, target))
            if (not checker.state(*trial).valid
                    or not checker.transition(previous, trial).valid):
                continue
            remaining = sum(float(np.linalg.norm(new - current))
                            for new, current in zip(target, trial))
            distance = sum(float(np.linalg.norm(new - old))
                           for new, old in zip(target, previous))
            proposals.append((remaining, distance, trial))
    if not proposals:
        return None
    return min(proposals, key=lambda item: item[:2])[-1]


def _continuity_reference(initialized, previous_pair):
    """Apply branch continuity only after a safe task branch is initialized."""
    return previous_pair if initialized else None


def _initializer_probe_rows(selected_result, *, source_count):
    """Prefer search samples that already exhibited a connected safe pair."""
    count = int(source_count)
    if count < 1:
        raise ValueError("source_count must be positive")
    selected_result = selected_result or {}
    disconnected = {
        int(row) for row in selected_result.get("disconnected_rows", ())}
    runs = []
    current = []
    for value in selected_result.get("sampled_source_rows", ()):
        row = int(value)
        if not 0 <= row < count:
            continue
        if row in disconnected:
            if current:
                runs.append(current)
                current = []
        elif row not in current:
            current.append(row)
    if current:
        runs.append(current)
    if runs:
        runs.sort(key=lambda run: (-len(run), run[0]))
        return [row for run in runs for row in run]
    return np.unique(np.rint(np.linspace(
        0, count - 1, min(count, 32))).astype(int)).tolist()


def _requires_pair_rescue(selected, *, state_safe, edge_safe):
    """Require joint recovery for safety or when either arm lacks a target."""
    return (not state_safe or not edge_safe
            or any(selected.get(side) is None
                   for side in ("left", "right")))


def _recovery_allowed(row, *, initialization_row):
    return initialization_row is None or int(row) >= int(initialization_row)


def _solve_candidate_dls_hold(model, task, mapped, *,
                              branch_guard_rad=0.30,
                              initializer_rows=None):
    """Paired pre-initialization, warm-start, safe recovery, and HOLD."""
    data = mujoco.MjData(model)
    contract = ROBOT_CONTRACTS["piperx"]
    names = {side: {
        "joints": contract.prefixed_joint_names(side),
        "site": f"{side}_tcp",
    } for side in ("left", "right")}
    config = CandidateGeneratorConfig(
        position_tolerance_m=0.001,
        orientation_tolerance_rad=np.deg2rad(0.5),
        damping=0.3,
        step_scale=1.0,
        maximum_step_rad=0.3,
        position_error_clip_m=0.05,
        orientation_error_clip_rad=0.3,
        max_iterations=200,
        maximum_candidates=16,
        global_seed_count=5,
        stratified_seed_enabled=True,
        stratified_refresh_interval=20,
        rolling_early_stop_candidates=2,
        dedup_rad=np.deg2rad(1.0),
        constrained_fallback_enabled=True,
        constrained_fallback_seed_count=4,
        constrained_fallback_max_iterations=80,
        wrist_risk_enabled=True,
    )
    generators = {side: MuJoCoCandidateGenerator(
        model, data, contract, name_map={side: names[side]}, config=config)
        for side in ("left", "right")}
    collision_checker = MuJoCoPairedCollisionChecker(
        model, data, names, **_piperx_collision_checker_kwargs())
    topology_checker = MuJoCoMountTopologyChecker(
        model, data, names,
        config=MountTopologyConfig(transition_steps=5))
    safety_checker = _ConjunctivePairChecker(
        collision_checker, topology_checker)
    qids = {}
    for side in ("left", "right"):
        joint_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names[side]["joints"]]
        qids[side] = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
    count = len(task.time_s)
    qpos = np.repeat(model.qpos0[None, :], count, axis=0)
    actual = {side: np.zeros((count, 7), dtype=float)
              for side in ("left", "right")}
    position_error = {side: np.zeros(count, dtype=float)
                      for side in ("left", "right")}
    orientation_error = {side: np.zeros(count, dtype=float)
                         for side in ("left", "right")}
    strict = {side: np.zeros(count, dtype=bool)
              for side in ("left", "right")}
    discontinuity = {side: np.zeros(count, dtype=bool)
                     for side in ("left", "right")}
    mode = np.full((count, 2), "hold", dtype="U16")
    side_index = {"left": 0, "right": 1}
    last_rescue = {"left": -20, "right": -20}
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    initialization_row = None
    initial_pair = None
    probe_rows = ([0] if initializer_rows is None
                  else [int(row) for row in initializer_rows])
    for probe_row in probe_rows:
        if not 0 <= probe_row < count:
            continue
        candidate_lists = {}
        for side in ("left", "right"):
            candidate_lists[side] = generators[side].generate_target(
                side,
                getattr(task, f"{side}_position_m")[probe_row],
                mapped[side][probe_row],
                force_stratified=True)
        initial_pair = _select_collision_safe_pair(
            candidate_lists, previous=None, checker=safety_checker,
            branch_guard_rad=branch_guard_rad)
        if initial_pair is not None:
            initialization_row = probe_row
            for side, item in zip(("left", "right"), initial_pair):
                data.qpos[qids[side]] = item.q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            break
    initialized = initial_pair is not None
    for row in range(count):
        previous = {side: data.qpos[qids[side]].copy()
                    for side in ("left", "right")}
        selected = {}
        for side in ("left", "right"):
            target_p = getattr(task, f"{side}_position_m")[row]
            target_q = mapped[side][row]
            if row == 0:
                candidates = generators[side].generate_target(
                    side, target_p, target_q, force_stratified=True)
                selected[side] = min(candidates, key=lambda item: (
                    item.pose_cost, -item.joint_limit_margin_rad,
                    -item.singularity_margin)) if candidates else None
                mode[row, side_index[side]] = (
                    "anchor" if selected[side] is not None else "hold")
            else:
                selected[side] = generators[side].generate_warm_start_candidate(
                    side, target_p, target_q, reference_q=previous[side])
                if selected[side] is not None:
                    mode[row, side_index[side]] = "warm_start"
                else:
                    rescue = []
                    if row - last_rescue[side] >= 20:
                        last_rescue[side] = row
                        rescue = generators[side].generate_target(
                            side, target_p, target_q,
                            force_stratified=True)
                    admissible = [item for item in rescue
                                  if np.max(np.abs(
                                      item.q - previous[side]))
                                  <= branch_guard_rad + 1e-12]
                    selected[side] = min(admissible, key=lambda item: (
                        np.linalg.norm(item.q - previous[side]),
                        item.pose_cost)) if admissible else None
                    mode[row, side_index[side]] = (
                        "rescue" if selected[side] is not None else "hold")
            item = selected[side]
            if item is None:
                data.qpos[qids[side]] = previous[side]
                continue
            delta = np.abs(item.q - previous[side])
            if row > 0 and np.any(delta > branch_guard_rad + 1e-12):
                discontinuity[side][row] = True
                selected[side] = None
                mode[row, side_index[side]] = "branch_guard"
                data.qpos[qids[side]] = previous[side]
            else:
                data.qpos[qids[side]] = item.q
        previous_pair = (previous["left"], previous["right"])
        current_pair = tuple(
            data.qpos[qids[side]].copy() for side in ("left", "right"))
        state_safe = safety_checker.state(*current_pair).valid
        edge_safe = (row == 0 or safety_checker.transition(
            previous_pair, current_pair).valid)
        if _requires_pair_rescue(
                selected, state_safe=state_safe, edge_safe=edge_safe):
            rescue_lists = {}
            for side in ("left", "right"):
                target_p = getattr(task, f"{side}_position_m")[row]
                rescue_lists[side] = generators[side].generate_target(
                    side, target_p, mapped[side][row],
                    force_stratified=True)
                if selected.get(side) is not None:
                    rescue_lists[side].append(selected[side])
            safe_pair = _select_collision_safe_pair(
                rescue_lists,
                previous=_continuity_reference(initialized, previous_pair),
                checker=safety_checker,
                branch_guard_rad=branch_guard_rad)
            if safe_pair is None:
                recovery = (None if not _recovery_allowed(
                    row, initialization_row=initialization_row)
                    else _select_safe_recovery_step(
                        rescue_lists, previous=previous_pair,
                        checker=safety_checker,
                        maximum_step_rad=branch_guard_rad))
                for side, recovery_q in zip(("left", "right"),
                                            recovery or previous_pair):
                    data.qpos[qids[side]] = recovery_q
                    selected[side] = None
                    mode[row, side_index[side]] = (
                        "recovery_step" if recovery is not None
                        else "collision_hold")
            else:
                for side, item in zip(("left", "right"), safe_pair):
                    data.qpos[qids[side]] = item.q
                    selected[side] = item
                    mode[row, side_index[side]] = "collision_rescue"
                initialized = True
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side in ("left", "right"):
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, names[side]["site"])
            pose_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(pose_quaternion, data.site_xmat[site_id])
            residual = np.empty(3)
            mujoco.mju_subQuat(residual, mapped[side][row], pose_quaternion)
            actual[side][row] = np.r_[data.site_xpos[site_id], pose_quaternion]
            position_error[side][row] = np.linalg.norm(
                getattr(task, f"{side}_position_m")[row]
                - data.site_xpos[site_id])
            orientation_error[side][row] = np.linalg.norm(residual)
            strict[side][row] = (
                position_error[side][row] <= 0.001 + 1e-12
                and orientation_error[side][row]
                <= np.deg2rad(0.5) + 1e-12)
        qpos[row] = data.qpos
    diagnostics = {
        "failure_reason": np.where(
            np.isin(mode, (
                "anchor", "warm_start", "rescue", "collision_rescue")),
            "ok", np.where(mode == "hold", "dls_exhausted", mode)),
        "solve_mode": mode,
        "protocol": "v3.3_paired_preinit_collision_topology_safe_recovery",
        "maximum_iterations": 200,
        "branch_guard_rad": branch_guard_rad,
        "initialization_source_row": initialization_row,
    }
    return (qpos, actual, position_error, orientation_error, strict,
            discontinuity, diagnostics)


def _topology_valid(model, qpos):
    names = {side: {
        "joints": ROBOT_CONTRACTS["piperx"].prefixed_joint_names(side),
        "site": f"{side}_tcp",
    } for side in ("left", "right")}
    checker = MuJoCoMountTopologyChecker(
        model, mujoco.MjData(model), names,
        config=MountTopologyConfig(transition_steps=5))
    qids = {}
    for side in ("left", "right"):
        joint_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names[side]["joints"]]
        qids[side] = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
    valid = np.ones(len(qpos), dtype=bool)
    previous = None
    for index, row in enumerate(qpos):
        current = (row[qids["left"]], row[qids["right"]])
        state = checker.state(*current)
        edge = None if previous is None else checker.transition(previous, current)
        valid[index] = state.valid and (edge is None or edge.valid)
        previous = current
    return valid


def _checkpoint_path(root, spec, mode):
    return (Path(root) / "checkpoints" / spec.family.date
            / spec.family.task / spec.take / f"{mode}.json")


def solve_selected_shard(spec, mode, output_root=DEFAULT_OUTPUT, *,
                         source_prefix=None):
    """Run strict paired IK on the native timestamps of one selected mount."""
    checkpoint = _checkpoint_path(output_root, spec, mode)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (state.get("status") not in {"complete", "infeasible"}
            or not state.get("selected_mount")):
        raise ValueError(f"{spec.key}/{mode}: selected mount is unavailable")
    mount = state["selected_mount"]
    formal_fingerprint = _formal_fingerprint(
        spec, mode, state, source_prefix=source_prefix)
    task, _registration = search_runner._load_registered_spec(spec)
    task, mapped, conditioning_audit = (
        search_runner.prepare_family_follow_targets(spec, task))
    task = search_runner._prefix_task(task, source_prefix)
    mapped = {side: np.asarray(mapped[side])[:len(task.time_s)]
              for side in ("left", "right")}
    shard_dir = (Path(output_root) / "shards" / spec.family.date
                 / spec.family.task / spec.take / mode)
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.family.date}_{spec.family.task}_{spec.take}_{mode}_fixed_time"
    scene = shard_dir / f"{stem}.scene.xml"
    separation = float(np.linalg.norm(
        np.asarray(mount["xy"]["left"], float)
        - np.asarray(mount["xy"]["right"], float)))
    build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], separation, scene,
        table_height_m=fixed_runner.TABLE_HEIGHT_M,
        mount_xy_m=mount["xy"], mount_yaw_deg=mount["yaw"],
        **_scene_mount_kwargs(task, mount))
    model = mujoco.MjModel.from_xml_path(str(scene))
    prepared = task
    (qpos, actual, position_error, orientation_error, _strict,
     discontinuity, diagnostics) = _solve_candidate_dls_hold(
        model, prepared, mapped, branch_guard_rad=0.30,
        initializer_rows=_initializer_probe_rows(
            state.get("selected_result"), source_count=len(task.time_s)))
    source_time = np.asarray(prepared.time_s, dtype=float)
    source_time = source_time - source_time[0]
    velocity, acceleration = _joint_derivatives(model, qpos, source_time)
    _collision_union, state_classes, edge_classes = audit_bimanual_collisions(
        model, qpos, robot_name="piperx")
    collision = np.asarray([
        np.asarray(value).size > 0 for value in state_classes])
    edge_collision = np.asarray([
        np.asarray(value).size > 0 for value in edge_classes])
    topology_valid = _topology_valid(model, qpos)
    position = np.column_stack((
        position_error["left"], position_error["right"]))
    orientation = np.column_stack((
        orientation_error["left"], orientation_error["right"]))
    payload = build_shard_arrays(
        mode=mode, source_time_s=source_time, qpos=qpos,
        position_error_m=position, orientation_error_rad=orientation,
        collision=collision, edge_collision=edge_collision,
        topology_valid=topology_valid,
        left_source_valid=np.asarray(prepared.left_valid, dtype=bool),
        right_source_valid=np.asarray(prepared.right_valid, dtype=bool),
        velocity_rad_s=velocity,
        acceleration_rad_s2=acceleration)
    payload.update({
        "left_actual_tcp": np.asarray(actual["left"]),
        "right_actual_tcp": np.asarray(actual["right"]),
        "left_target_position_m": np.asarray(prepared.left_position_m),
        "right_target_position_m": np.asarray(prepared.right_position_m),
        "left_target_quaternion_wxyz": np.asarray(mapped["left"]),
        "right_target_quaternion_wxyz": np.asarray(mapped["right"]),
        "left_discontinuity": np.asarray(discontinuity["left"]),
        "right_discontinuity": np.asarray(discontinuity["right"]),
        "state_collision_classes": np.asarray([
            ";".join(map(str, value)) for value in state_classes], dtype=np.str_),
        "edge_collision_classes": np.asarray([
            ";".join(map(str, value)) for value in edge_classes], dtype=np.str_),
        "paired_failure_reason": np.asarray([
            "ok" if all(value == "ok" for value in row)
            else ";".join(value for value in row if value != "ok")
            for row in diagnostics["failure_reason"]], dtype=np.str_),
        "dls_solve_mode": diagnostics["solve_mode"],
    })
    trajectory = shard_dir / f"{stem}.trajectory.npz"
    np.savez_compressed(trajectory, **payload)
    validation_spec = spec
    if source_prefix is not None:
        from dataclasses import replace
        validation_spec = replace(spec, row_count=len(source_time))
    validate_shard(payload, validation_spec, mode)
    metrics = aggregate_shard(payload)
    summary = {
        "schema": "piperx-multitask-fixed-time-summary-v1",
        "solver_protocol": FORMAL_SOLVER_PROTOCOL,
        "formal_fingerprint": formal_fingerprint,
        "search_job_fingerprint": state.get("job_fingerprint"),
        "trajectory": spec.key, "family": spec.family.key,
        "take": spec.take, "mode": mode,
        "timing_mode": "fixed_source_time",
        "retiming_applied": False,
        "source_sha256": spec.source_sha256,
        "target_preparation": {
            "protocol": "piperx-v3.1-family-tool-frame-and-wrist",
            "conditioning": {
                key: value for key, value in conditioning_audit.__dict__.items()
            },
        },
        "mount": mount, "metrics": metrics,
        "dynamics": _dynamics_summary(payload),
        "artifacts": {
            "trajectory_npz": {"path": _repo_relative(trajectory),
                               "sha256": _sha256(trajectory)},
            "scene_xml": {"path": _repo_relative(scene),
                          "sha256": _sha256(scene)},
        },
    }
    summary_path = shard_dir / f"{stem}.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return summary_path


def aggregate_shard(payload):
    left = np.asarray(payload["left_accept"], dtype=bool)
    right = np.asarray(payload["right_accept"], dtype=bool)
    both = np.asarray(payload["both_accept"], dtype=bool)
    count = len(both)
    if not (left.shape == right.shape == both.shape == (count,)):
        raise ValueError("accept arrays must share one frame dimension")
    if not np.array_equal(both, left & right):
        raise ValueError("both_accept must equal the conjunction of both arms")
    position = np.asarray(payload["position_error_m"], dtype=float)
    orientation = np.asarray(payload["orientation_error_rad"], dtype=float)
    if position.shape != (count, 2) or orientation.shape != (count, 2):
        raise ValueError("pose error arrays must have shape (frames, 2)")
    collision = np.asarray(payload["collision"], dtype=bool)
    edge = np.asarray(payload["edge_collision"], dtype=bool)
    topology = np.asarray(payload["topology_valid"], dtype=bool)
    left_source_valid = np.asarray(payload["left_source_valid"], dtype=bool)
    right_source_valid = np.asarray(payload["right_source_valid"], dtype=bool)
    if collision.shape != (count,) or edge.shape != (count,) or topology.shape != (count,):
        raise ValueError("safety arrays must have one entry per frame")
    if (left_source_valid.shape != (count,)
            or right_source_valid.shape != (count,)):
        raise ValueError("source validity arrays must have one entry per frame")
    pair_source_valid = left_source_valid & right_source_valid
    valid_count = int(np.count_nonzero(pair_source_valid))
    accepted_position = position[both]
    accepted_orientation = orientation[both]
    longest_hold = _longest_false_run(both)
    return {
        "source_frames": count,
        "left_accept_frames": int(np.count_nonzero(left)),
        "right_accept_frames": int(np.count_nonzero(right)),
        "both_accept_frames": int(np.count_nonzero(both)),
        "left_accept_coverage": float(np.mean(left)),
        "right_accept_coverage": float(np.mean(right)),
        "both_accept_coverage": float(np.mean(both)),
        "source_valid_pair_frames": valid_count,
        "invalid_source_frames": int(count - valid_count),
        "both_accept_coverage_valid_source": (
            float(np.count_nonzero(both & pair_source_valid) / valid_count)
            if valid_count else 0.0),
        "collision_frames": int(np.count_nonzero(collision)),
        "edge_collision_frames": int(np.count_nonzero(edge)),
        "topology_invalid_frames": int(np.count_nonzero(~topology)),
        "longest_hold_frames": longest_hold,
        "longest_hold_ratio": float(longest_hold / count),
        "maximum_position_error_mm": float(np.max(position) * 1000.0),
        "maximum_orientation_error_deg": float(np.rad2deg(np.max(orientation))),
        "maximum_accepted_position_error_mm": (
            float(np.max(accepted_position) * 1000.0)
            if accepted_position.size else None),
        "maximum_accepted_orientation_error_deg": (
            float(np.rad2deg(np.max(accepted_orientation)))
            if accepted_orientation.size else None),
    }


def validate_manifest(manifest):
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unexpected multitask bundle schema")
    if (manifest.get("trajectory_count") != 27
            or manifest.get("family_count") != 12
            or manifest.get("mode_count") != 4):
        raise ValueError("bundle must describe 27 trajectories and four modes")
    shards = manifest.get("shards", [])
    if len(shards) != 108:
        raise ValueError("bundle must contain exactly 108 shards")
    pairs = {(item.get("trajectory"), item.get("mode")) for item in shards}
    if len(pairs) != 108:
        raise ValueError("bundle contains duplicate trajectory/mode shards")
    if {mode for _, mode in pairs} != set(STUDY_MODES):
        raise ValueError("bundle must contain every published mount mode")
    return manifest


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _resolve_artifact(path):
    path = Path(path)
    if path.is_absolute():
        raise ValueError("artifact paths must be repository-relative")
    root = ROOT.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path resolves outside repository") from exc
    return resolved


def _repo_relative(path):
    try:
        relative = Path(path).resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("artifact must be stored inside the repository") from exc
    return relative.as_posix()


def _validate_summary_contract(summary, shard, spec, *, expected_fingerprint,
                               expected_search_fingerprint=None):
    expected = {
        "schema": "piperx-multitask-fixed-time-summary-v1",
        "solver_protocol": FORMAL_SOLVER_PROTOCOL,
        "formal_fingerprint": expected_fingerprint,
        "trajectory": shard["trajectory"],
        "family": shard["family"],
        "take": shard["take"],
        "mode": shard["mode"],
        "timing_mode": "fixed_source_time",
        "retiming_applied": False,
        "source_sha256": spec.source_sha256,
    }
    if expected_search_fingerprint is not None:
        expected["search_job_fingerprint"] = expected_search_fingerprint
    mismatches = [
        key for key, value in expected.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "summary contract mismatch: " + ", ".join(mismatches))


def _validate_formal_evidence(payload, count):
    shapes = {
        "left_actual_tcp": (count, 7),
        "right_actual_tcp": (count, 7),
        "left_target_position_m": (count, 3),
        "right_target_position_m": (count, 3),
        "left_target_quaternion_wxyz": (count, 4),
        "right_target_quaternion_wxyz": (count, 4),
    }
    for name, shape in shapes.items():
        value = np.asarray(payload.get(name), dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"formal evidence {name} must be finite shape {shape}")
    stored_position = np.asarray(payload["position_error_m"], dtype=float)
    stored_orientation = np.asarray(
        payload["orientation_error_rad"], dtype=float)
    for side_index, side in enumerate(("left", "right")):
        actual = np.asarray(payload[f"{side}_actual_tcp"], dtype=float)
        target_position = np.asarray(
            payload[f"{side}_target_position_m"], dtype=float)
        target_quaternion = np.asarray(
            payload[f"{side}_target_quaternion_wxyz"], dtype=float)
        actual_quaternion = actual[:, 3:]
        if not (np.allclose(
                    np.linalg.norm(target_quaternion, axis=1), 1.0,
                    rtol=0.0, atol=1e-6)
                and np.allclose(
                    np.linalg.norm(actual_quaternion, axis=1), 1.0,
                    rtol=0.0, atol=1e-6)):
            raise ValueError(
                f"formal evidence {side} quaternions are not normalized")
        position = np.linalg.norm(target_position - actual[:, :3], axis=1)
        quaternion_dot = np.abs(np.einsum(
            "ij,ij->i", target_quaternion, actual_quaternion))
        orientation = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
        if not np.allclose(
                stored_position[:, side_index], position,
                rtol=0.0, atol=1e-10):
            raise ValueError(f"formal evidence {side} position error drift")
        if not np.allclose(
                stored_orientation[:, side_index], orientation,
                rtol=0.0, atol=1e-8):
            raise ValueError(f"formal evidence {side} orientation error drift")
    for name in (
            "left_discontinuity", "right_discontinuity",
            "state_collision_classes", "edge_collision_classes",
            "paired_failure_reason"):
        if np.asarray(payload.get(name)).shape != (count,):
            raise ValueError(f"formal evidence {name} must have shape ({count},)")
    if np.asarray(payload.get("dls_solve_mode")).shape != (count, 2):
        raise ValueError(
            f"formal evidence dls_solve_mode must have shape ({count}, 2)")
    state_classes = np.asarray(payload["state_collision_classes"]).astype(str)
    edge_classes = np.asarray(payload["edge_collision_classes"]).astype(str)
    collision = np.asarray(payload["collision"], dtype=bool)
    edge_collision = np.asarray(payload["edge_collision"], dtype=bool)
    if (not np.array_equal(collision, np.char.str_len(state_classes) > 0)
            or not np.array_equal(
                edge_collision, np.char.str_len(edge_classes) > 0)):
        raise ValueError("formal evidence collision class drift")


def _validate_scene_manifest(scene):
    manifest_path = Path(scene).with_suffix(".json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(payload["source_urdf"])
    output = Path(payload["output_xml"])
    if source.is_absolute() or output.is_absolute():
        raise ValueError("scene manifest paths must be scene-relative")
    if (manifest_path.parent / source).resolve() != ROBOT_CONTRACTS["piperx"].source_urdf:
        raise ValueError("scene manifest source URDF mismatch")
    if (manifest_path.parent / output).resolve() != Path(scene).resolve():
        raise ValueError("scene manifest output XML mismatch")


def _validate_aggregate_csv(path, expected_rows):
    """Require the published table to be an exact serialization of shards."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        actual_rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    expected_rows = list(expected_rows)
    expected_fields = list(expected_rows[0]) if expected_rows else []
    if fieldnames != expected_fields or len(actual_rows) != len(expected_rows):
        raise ValueError("aggregate CSV drift: schema or row count mismatch")

    for row_index, (actual, expected) in enumerate(
            zip(actual_rows, expected_rows, strict=True)):
        for name, expected_value in expected.items():
            actual_value = actual[name]
            if expected_value is None:
                matches = actual_value == ""
            elif isinstance(expected_value, bool):
                matches = actual_value == str(expected_value)
            elif isinstance(expected_value, int):
                try:
                    matches = int(actual_value) == expected_value
                except ValueError:
                    matches = False
            elif isinstance(expected_value, float):
                try:
                    matches = np.isclose(
                        float(actual_value), expected_value,
                        rtol=0.0, atol=1e-12)
                except ValueError:
                    matches = False
            else:
                matches = actual_value == str(expected_value)
            if not matches:
                raise ValueError(
                    "aggregate CSV drift: "
                    f"row {row_index} field {name!r} does not match shards")


def validate_bundle_artifacts(manifest_path, *, require_complete=True):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_complete:
        validate_manifest(manifest)
    discovered_specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    specs = {item.key: item for item in discovered_specs}
    aggregate_rows = []
    for shard in manifest.get("shards", []):
        spec = specs[shard["trajectory"]]
        summary_path = _resolve_artifact(shard["summary_json"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_prefix = None
        if (manifest.get("status") == "partial"
                and summary["metrics"]["source_frames"] != spec.row_count):
            from dataclasses import replace
            source_prefix = int(summary["metrics"]["source_frames"])
            spec = replace(
                spec, row_count=source_prefix)
        expected_search_fingerprint = _expected_search_job_fingerprint(
            spec, shard["mode"], discovered_specs)
        summary_state = {
            "selected_mount": summary.get("mount"),
            "job_fingerprint": expected_search_fingerprint,
        }
        expected_fingerprint = _formal_fingerprint(
            spec, shard["mode"], summary_state, source_prefix=source_prefix)
        _validate_summary_contract(
            summary, shard, spec,
            expected_fingerprint=expected_fingerprint,
            expected_search_fingerprint=expected_search_fingerprint)
        trajectory = _resolve_artifact(
            summary["artifacts"]["trajectory_npz"]["path"])
        if _sha256(trajectory) != summary["artifacts"]["trajectory_npz"]["sha256"]:
            raise ValueError(f"{shard['trajectory']}/{shard['mode']}: trajectory hash mismatch")
        with np.load(trajectory, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        payload["schema"] = str(payload["schema"].item())
        payload["mode"] = str(payload["mode"].item())
        payload["retiming_applied"] = bool(payload["retiming_applied"].item())
        validate_shard(payload, spec, shard["mode"])
        _validate_formal_evidence(payload, spec.row_count)
        scene = _resolve_artifact(summary["artifacts"]["scene_xml"]["path"])
        if _sha256(scene) != summary["artifacts"]["scene_xml"]["sha256"]:
            raise ValueError(
                f"{shard['trajectory']}/{shard['mode']}: scene hash mismatch")
        _validate_scene_manifest(scene)
        recomputed = aggregate_shard(payload)
        if recomputed != summary["metrics"]:
            raise ValueError(f"{shard['trajectory']}/{shard['mode']}: summary drift")
        recomputed_dynamics = _dynamics_summary(payload)
        if recomputed_dynamics != summary["dynamics"]:
            raise ValueError(
                f"{shard['trajectory']}/{shard['mode']}: dynamics summary drift")
        aggregate_rows.append({
            "trajectory": spec.key,
            "family": spec.family.key,
            "take": spec.take,
            "mode": shard["mode"],
            "search_status": shard["search_status"],
            **recomputed,
            **recomputed_dynamics,
        })
    aggregate_path = _resolve_artifact(manifest["aggregate_csv"])
    _validate_aggregate_csv(aggregate_path, aggregate_rows)
    return manifest


def build_bundle(output_root=DEFAULT_OUTPUT, *, trajectory=None, mode=None,
                 source_prefix=None, require_complete=True):
    output_root = Path(output_root)
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    jobs = [(spec, job_mode) for spec in specs
            for job_mode in STUDY_MODES]
    if trajectory:
        jobs = [job for job in jobs if job[0].key == trajectory]
    if mode:
        jobs = [job for job in jobs if job[1] == mode]
    if not jobs:
        raise ValueError("no bundle jobs matched the requested filters")
    shards = []
    aggregate_rows = []
    for index, (spec, job_mode) in enumerate(jobs, 1):
        checkpoint = _checkpoint_path(output_root, spec, job_mode)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"missing search checkpoint: {spec.key}/{job_mode}")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("status") not in {"complete", "infeasible"}:
            raise ValueError(
                f"unfinished search checkpoint: {spec.key}/{job_mode}")
        shard_dir = (output_root / "shards" / spec.family.date
                     / spec.family.task / spec.take / job_mode)
        summaries = list(shard_dir.glob("*.summary.json"))
        summary_path = (summaries[0] if len(summaries) == 1
                        and _formal_summary_reusable(
                            summaries[0], _formal_fingerprint(
                                spec, job_mode, state,
                                source_prefix=source_prefix)) else
                        solve_selected_shard(
                            spec, job_mode, output_root,
                            source_prefix=source_prefix))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_link = str(summary_path.resolve().relative_to(ROOT.resolve()))
        shards.append({
            "trajectory": spec.key, "family": spec.family.key,
            "take": spec.take, "mode": job_mode,
            "search_status": state["status"],
            "summary_json": summary_link.replace("\\", "/"),
        })
        aggregate_rows.append({
            "trajectory": spec.key, "family": spec.family.key,
            "take": spec.take, "mode": job_mode,
            "search_status": state["status"],
            **summary["metrics"], **summary["dynamics"],
        })
        partial = {
            "schema": BUNDLE_SCHEMA,
            "trajectory_count": len(specs),
            "family_count": len({spec.family.key for spec in specs}),
            "mode_count": len(STUDY_MODES),
            "status": "running", "completed_shards": index,
            "shards": shards,
        }
        _atomic_json(output_root / "bundle_manifest.partial.json", partial)
    aggregate_path = output_root / "aggregate.csv"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "trajectory_count": len(specs),
        "family_count": len({spec.family.key for spec in specs}),
        "mode_count": len(STUDY_MODES),
        "status": "complete" if len(shards) == 108 else "partial",
        "retiming_applied": False,
        "aggregate_csv": str(aggregate_path.resolve().relative_to(
            ROOT.resolve())).replace("\\", "/"),
        "shards": shards,
    }
    if require_complete:
        validate_manifest(manifest)
    manifest_path = output_root / "bundle_manifest.json"
    _atomic_json(manifest_path, manifest)
    validate_bundle_artifacts(
        manifest_path, require_complete=require_complete)
    return manifest_path


def solve_shards(output_root=DEFAULT_OUTPUT, *, trajectory=None, mode=None):
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    jobs = [(spec, job_mode) for spec in specs for job_mode in STUDY_MODES]
    if trajectory:
        jobs = [job for job in jobs if job[0].key == trajectory]
    if mode:
        jobs = [job for job in jobs if job[1] == mode]
    if not jobs:
        raise ValueError("no shard jobs matched the requested filters")
    outputs = []
    for index, (spec, job_mode) in enumerate(jobs, 1):
        shard_dir = (Path(output_root) / "shards" / spec.family.date
                     / spec.family.task / spec.take / job_mode)
        summaries = list(shard_dir.glob("*.summary.json"))
        state = json.loads(_checkpoint_path(
            output_root, spec, job_mode).read_text(encoding="utf-8"))
        output = (summaries[0] if len(summaries) == 1
                  and _formal_summary_reusable(
                      summaries[0], _formal_fingerprint(
                          spec, job_mode, state))
                  else solve_selected_shard(spec, job_mode, output_root))
        outputs.append(output)
        print(index, len(jobs), spec.key, job_mode, output, flush=True)
    return tuple(outputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument("--trajectory")
    parser.add_argument("--mode", choices=STUDY_MODES)
    parser.add_argument("--source-prefix", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.output / "bundle_manifest.json"
    if args.solve_only:
        solve_shards(
            args.output, trajectory=args.trajectory, mode=args.mode)
        return
    if args.validate_only:
        validate_bundle_artifacts(
            manifest_path, require_complete=not args.allow_partial)
        print(manifest_path)
        return
    print(build_bundle(
        args.output, trajectory=args.trajectory, mode=args.mode,
        source_prefix=args.source_prefix,
        require_complete=not args.allow_partial))


if __name__ == "__main__":
    main()
