"""Search upright dual-xArm6 mounts and render complete SE(3) TCP following."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from factory_bimanual.artifacts import FrameDiagnostics
from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig, MuJoCoCandidateGenerator,
)
from factory_bimanual.mujoco_collision_adapter import MuJoCoPairedCollisionChecker
from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.source_data import load_factory_task
from factory_bimanual.video import VideoRenderConfig, render_mujoco_mp4
from scripts.rolling_multibranch_ik import (
    BranchCandidate, select_receding_horizon_path,
)
from scripts.strict_mujoco_ik import joint_periodic_mask, solve_pose_path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/factory/8-12/PourRawMaterial/handheld_20260812_111542.csv"
OUT = ROOT / "reports/factory_bimanual/pour_raw_material_dual_xarm6/pour_raw_material_full_se3_fixed_time_front_720p.mp4"

SELECTED_MOUNT = {
    "xy": {
        "left": (0.047484946968250535, 0.24922072906595222),
        "right": (-0.2800651432877138, -0.1308664458233453),
    },
    "yaw": {"left": 270.0, "right": 45.0},
    "shared_base_z_m": 0.87,
    "selection_method": (
        "bounded upright mount search; 180 uniform plus workspace-extrema samples; "
        "strict independent 6D pose IK before timestamp validation"),
}

PLATEAU_THRESHOLDS = {
    "coverage": 0.005,
    "p95_position_mm": 1.0,
    "p95_orientation_deg": 0.2,
    "patience": 3,
    "maximum_iterations": 8,
}


def improvement_has_plateaued(history, *, patience=3, coverage_threshold=.005,
                              position_threshold_mm=1.0,
                              orientation_threshold_deg=.2):
    """Stop when every metric changes negligibly for `patience` iterations."""
    if len(history) < patience + 1:
        return False
    recent = history[-(patience + 1):]
    for previous, current in zip(recent[:-1], recent[1:]):
        if current["coverage"] - previous["coverage"] >= coverage_threshold:
            return False
        if previous["p95_position_mm"] - current["p95_position_mm"] >= position_threshold_mm:
            return False
        if previous["p95_orientation_deg"] - current["p95_orientation_deg"] >= orientation_threshold_deg:
            return False
    return True


def shared_mount_height_candidates(*, table_height_m, minimum_adapter_m,
                                   maximum_adapter_m, count):
    """Return capped world-Z values shared by both upright robot bases."""
    if count < 1 or minimum_adapter_m < 0 or maximum_adapter_m < minimum_adapter_m:
        raise ValueError("invalid shared mount height bounds")
    return [round(float(table_height_m + value), 12) for value in
            np.linspace(minimum_adapter_m, maximum_adapter_m, count)]


def synchronous_mount_score(*, left_success, right_success, left_sigma,
                            right_sigma, mean_pose_error):
    """Rank synchronous feasibility before conditioning and residual error."""
    left = np.asarray(left_success, bool); right = np.asarray(right_success, bool)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("left/right success arrays must be matching 1-D arrays")
    sigma = np.minimum(np.asarray(left_sigma, float), np.asarray(right_sigma, float))
    if sigma.shape != left.shape:
        raise ValueError("singularity arrays must match success arrays")
    synchronous_coverage = float(np.mean(left & right))
    conditioning = float(np.quantile(sigma, .1))
    return (-synchronous_coverage, -conditioning, float(mean_pose_error))


def map_source_quaternions_to_tcp(source_quaternions_wxyz, robot_initial_quaternion_wxyz):
    """Apply one fixed source-tool calibration to world-frame source poses."""
    source = np.asarray(source_quaternions_wxyz, dtype=float)
    robot_initial = np.asarray(robot_initial_quaternion_wxyz, dtype=float)
    if source.ndim != 2 or source.shape[1] != 4 or robot_initial.shape != (4,):
        raise ValueError("quaternions must have shapes (frames, 4) and (4,)")
    source_norms = np.linalg.norm(source, axis=1)
    robot_norm = float(np.linalg.norm(robot_initial))
    if np.any(source_norms < 1e-12) or robot_norm < 1e-12:
        raise ValueError("quaternions must be nonzero")
    source = source / source_norms[:, None]
    robot_initial = robot_initial / robot_norm

    source_matrices = []
    for value in source:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, value)
        source_matrices.append(matrix.reshape(3, 3))
    robot_matrix = np.empty(9)
    mujoco.mju_quat2Mat(robot_matrix, robot_initial)
    calibration = source_matrices[0].T @ robot_matrix.reshape(3, 3)

    mapped = []
    for source_matrix in source_matrices:
        target_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(target_quaternion, (source_matrix @ calibration).reshape(-1))
        mapped.append(target_quaternion)
    return np.asarray(mapped)


def classify_ik_failure(*, candidate_count, recovery_mode, position_error_m,
                        orientation_error_rad,
                        realized_velocity_violation):
    """Return the primary reason a realized TCP sample misses strict SE(3)."""
    if realized_velocity_violation:
        return "realized joint velocity exceeded controller limit"
    if int(candidate_count) == 0:
        return "6D pose unreachable (no IK candidate)"
    if recovery_mode in ("limited_step", "hold_no_feasible_edge"):
        return "IK edge infeasible at source timing; bounded recovery"
    if recovery_mode == "hold_recovery_collision":
        return "bounded IK recovery would collide; holding safe state"
    if position_error_m > .001:
        return f"position tolerance exceeded ({1000 * position_error_m:.1f} mm)"
    if orientation_error_rad > np.deg2rad(1.5):
        return f"orientation tolerance exceeded ({np.rad2deg(orientation_error_rad):.1f} deg)"
    return "ok"


def pose_error(data, site, target_p, target_q):
    current_q = np.empty(4); mujoco.mju_mat2Quat(current_q, data.site_xmat[site])
    local = np.empty(3); mujoco.mju_subQuat(local, target_q, current_q)
    rotation = data.site_xmat[site].reshape(3, 3) @ local
    return target_p - data.site_xpos[site], rotation, current_q


def mapped_quaternions_at_joint_midpoint(model, task):
    """Calibrate each source tool frame against the same mounted home pose."""
    mapped = {}
    for side in ("left", "right"):
        data = mujoco.MjData(model)
        joint_names = tuple(f"{side}_joint{i}" for i in range(1, 7))
        joint_ids = np.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names], dtype=int)
        qids = model.jnt_qposadr[joint_ids]
        ranges = model.jnt_range[joint_ids]
        data.qpos[qids] = np.mean(ranges, axis=1)
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        home_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(home_quaternion, data.site_xmat[site])
        mapped[side] = map_source_quaternions_to_tcp(
            getattr(task, f"{side}_quaternion_wxyz"), home_quaternion)
    return mapped


def solve(model, task, *, iterations=65):
    data = mujoco.MjData(model); count = len(task.time_s)
    qpos = np.zeros((count, model.nq)); actual = {}; pe = {}; oe = {}
    mapped_quat = mapped_quaternions_at_joint_midpoint(model, task)
    for side in ("left", "right"):
        joint_names = tuple(f"{side}_joint{i}" for i in range(1, 7))
        joints = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names])
        qids = np.asarray([model.jnt_qposadr[j] for j in joints]); dids = np.asarray([model.jnt_dofadr[j] for j in joints])
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        pos = getattr(task, f"{side}_position_m"); source_quat = getattr(task, f"{side}_quaternion_wxyz")
        ranges = model.jnt_range[joints]
        data.qpos[qids] = np.mean(ranges, axis=1)
        mujoco.mj_forward(model, data)
        quat = mapped_quat[side]
        actual[side] = np.zeros((count, 7)); pe[side] = np.zeros(count); oe[side] = np.zeros(count)
        jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
        for row, (target_p, target_q) in enumerate(zip(pos, quat)):
            target_q = target_q / np.linalg.norm(target_q)
            for _ in range(iterations):
                mujoco.mj_forward(model, data)
                ep, er, _ = pose_error(data, site, target_p, target_q)
                if np.linalg.norm(ep) < .001 and np.linalg.norm(er) < np.deg2rad(1.5): break
                mujoco.mj_jacSite(model, data, jp, jr, site); jac = np.vstack((jp[:, dids], jr[:, dids]))
                err = np.r_[ep, er]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + .0025 * np.eye(6), err)
                data.qpos[qids] += np.clip(.65 * dq, -.13, .13)
                ranges = model.jnt_range[joints]; limited = model.jnt_limited[joints].astype(bool)
                data.qpos[qids[limited]] = np.clip(data.qpos[qids[limited]], ranges[limited, 0], ranges[limited, 1])
            mujoco.mj_forward(model, data); ep, er, current_q = pose_error(data, site, target_p, target_q)
            actual[side][row] = np.r_[data.site_xpos[site], current_q]
            pe[side][row], oe[side][row] = np.linalg.norm(ep), np.linalg.norm(er)
            qpos[row, qids] = data.qpos[qids]
    return qpos, actual, pe, oe, mapped_quat


def subset(task, indices):
    values = {k: (v[indices] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(task.time_s) else v)
              for k, v in task.__dict__.items()}
    return SimpleNamespace(**values)


def solve_strict_single_arm_method(model, task, mapped_quat):
    qpos = np.zeros((len(task.time_s), model.nq)); actual = {}; pe = {}; oe = {}; success = {}; discontinuity = {}; velocity = {}
    for seed, side in enumerate(("left", "right")):
        joints = tuple(f"{side}_joint{i}" for i in range(1, 7))
        result = solve_pose_path(model, f"{side}_tcp", joints,
            getattr(task, f"{side}_position_m"), mapped_quat[side],
            time_s=task.time_s, iterations=120, restarts=4,
            position_tolerance_m=.001, orientation_tolerance_rad=np.deg2rad(1.5),
            velocity_limit_rad_s=3.14, maximum_frame_jump_rad=np.deg2rad(35),
            rng_seed=20260811 + seed, hold_invalid=True)
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joints]
        qids = [model.jnt_qposadr[i] for i in ids]
        qpos[:, qids] = result.q
        actual[side] = np.c_[result.reached_xyz_m, result.reached_quaternion_wxyz]
        pe[side], oe[side], success[side] = result.position_error_m, result.orientation_error_rad, result.success
        discontinuity[side] = result.joint_discontinuity
        velocity[side] = result.velocity_violation
    return qpos, actual, pe, oe, success, discontinuity, velocity


def solve_multibranch_single_arm_method(
    model, task, mapped_quat, *, horizon=12, beam_width=8,
    candidate_iterations=80, velocity_limit_rad_s=3.14,
):
    """Generate raw pose branches, then select a timestamp-feasible path."""
    count = len(task.time_s)
    qpos = np.zeros((count, model.nq))
    actual = {}; pe = {}; oe = {}; success = {}; discontinuity = {}; velocity = {}
    diagnostics = {}
    contract = ROBOT_CONTRACTS["xarm6"]
    names = {side: {"joints": contract.prefixed_joint_names(side), "site": f"{side}_tcp"}
             for side in ("left", "right")}
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    generator = MuJoCoCandidateGenerator(
        model, data, contract, name_map=names,
        config=CandidateGeneratorConfig(
            max_iterations=candidate_iterations, dedup_rad=np.deg2rad(1.0)))
    collision_checker = MuJoCoPairedCollisionChecker(
        model, data, names, transition_steps=5)
    task_values = task.__dict__.copy()
    task_values.update({f"{side}_quaternion_wxyz": mapped_quat[side]
                        for side in ("left", "right")})
    candidate_task = SimpleNamespace(**task_values)
    intervals = np.r_[task.time_s[1] - task.time_s[0], np.diff(task.time_s)]

    for side in ("left", "right"):
        generator.reset()
        joints = tuple(f"{side}_joint{i}" for i in range(1, 7))
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                     for name in joints]
        qids = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
        periodic = joint_periodic_mask(model, joint_ids)
        ranges = model.jnt_range[joint_ids]
        initial_q = np.mean(ranges, axis=1)
        layers = []
        candidate_counts = np.zeros(count, dtype=int)
        for row in range(count):
            proposals = generator(model, contract, candidate_task, row, side)
            layer = tuple(BranchCandidate(
                q=item.q.copy(), pose_valid=True,
                collision_free=collision_checker.side_state(side, item.q).valid,
                position_error_m=float(item.position_error_m),
                orientation_error_rad=float(item.orientation_error_rad),
                joint_limit_margin=float(item.joint_limit_margin_rad),
                singularity_margin=float(item.singularity_margin),
                index=int(item.branch_index),
            ) for item in proposals)
            layers.append(layer); candidate_counts[row] = len(layer)
        path = select_receding_horizon_path(
            layers=layers, initial_q=initial_q, periodic=periodic,
            dt_s=intervals, velocity_limit_rad_s=np.full(6, velocity_limit_rad_s),
            cap_rad=np.full(6, np.deg2rad(35.0)), horizon=60,
            beam_width=beam_width, reseed_after_empty_frames=None,
            minimum_joint_limit_margin_rad=0.0,
            minimum_singularity_margin=0.0,
            maximum_recovery_wrist_distance_rad=np.inf,
            transition_valid=lambda previous, current, selected_side=side:
                collision_checker.side_transition(
                    selected_side, previous, current).valid,
        )
        qpos[:, qids] = path.q
        reached = np.zeros((count, 7)); position_error = np.zeros(count)
        orientation_error = np.zeros(count)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        scratch = mujoco.MjData(model)
        for row, q in enumerate(path.q):
            scratch.qpos[qids] = q; mujoco.mj_forward(model, scratch)
            quaternion = np.empty(4); mujoco.mju_mat2Quat(quaternion, scratch.site_xmat[site_id])
            reached[row] = np.r_[scratch.site_xpos[site_id], quaternion]
            position_error[row] = np.linalg.norm(
                getattr(task, f"{side}_position_m")[row] - scratch.site_xpos[site_id])
            residual = np.empty(3); mujoco.mju_subQuat(
                residual, mapped_quat[side][row], quaternion)
            orientation_error[row] = np.linalg.norm(residual)
        realized_velocity = np.zeros(count, dtype=bool)
        if count > 1:
            deltas = np.diff(path.q, axis=0)
            deltas[:, periodic] = (deltas[:, periodic] + np.pi) % (2 * np.pi) - np.pi
            realized_velocity[1:] = np.any(
                np.abs(deltas) / intervals[1:, None] > velocity_limit_rad_s + 1e-10,
                axis=1)
        strict = ((position_error <= .001) &
                  (orientation_error <= np.deg2rad(1.5)) &
                  ~realized_velocity)
        actual[side] = reached; pe[side] = position_error; oe[side] = orientation_error
        success[side] = strict
        discontinuity[side] = np.isin(
            path.recovery_mode, ("limited_step", "hold_no_feasible_edge",
                                 "hold_recovery_collision"))
        velocity[side] = realized_velocity
        diagnostics[side] = {
            "candidate_count": candidate_counts,
            "collision_free_candidate_count": np.asarray([
                sum(item.collision_free for item in layer) for layer in layers], dtype=int),
            "selected_candidate_id": path.selected_indices,
            "recovery_mode": path.recovery_mode,
            "required_dt_s": intervals.copy(),
            "selected_joint_limit_margin_rad": np.asarray([
                next((item.joint_limit_margin for item in layers[row]
                      if item.index == path.selected_indices[row]), np.nan)
                for row in range(count)]),
            "selected_singularity_margin": np.asarray([
                next((item.singularity_margin for item in layers[row]
                      if item.index == path.selected_indices[row]), np.nan)
                for row in range(count)]),
        }
    return qpos, actual, pe, oe, success, discontinuity, velocity, diagnostics


def audit_bimanual_collisions(model, qpos):
    """Audit final realized states and swept edges with one collision vocabulary."""
    contract = ROBOT_CONTRACTS["xarm6"]
    names = {side: {"joints": contract.prefixed_joint_names(side)}
             for side in ("left", "right")}
    checker = MuJoCoPairedCollisionChecker(
        model, mujoco.MjData(model), names, transition_steps=5)
    qids = {}
    for side in ("left", "right"):
        joints = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names[side]["joints"]]
        qids[side] = np.asarray(model.jnt_qposadr[joints], dtype=int)
    state_classes = []
    edge_classes = []
    previous = None
    for row in np.asarray(qpos):
        current = (row[qids["left"]], row[qids["right"]])
        state_classes.append(tuple(item.value for item in
                                   checker.state(*current).classes))
        edge_classes.append(() if previous is None else tuple(
            item.value for item in checker.transition(previous, current).classes))
        previous = current
    collision = np.asarray([bool(state or edge) for state, edge in
                            zip(state_classes, edge_classes)], dtype=bool)
    return collision, np.asarray(state_classes, dtype=object), np.asarray(edge_classes, dtype=object)


def failure_windows(time_s, failed, reasons):
    """Return complete contiguous cannot-follow windows in source time."""
    values = np.asarray(failed, dtype=bool)
    changes = np.diff(np.r_[False, values, False].astype(int))
    windows = []
    for start, stop in zip(np.flatnonzero(changes == 1),
                           np.flatnonzero(changes == -1)):
        primary = Counter(reasons[start:stop]).most_common(1)[0][0]
        windows.append({
            "start_frame": int(start), "end_frame": int(stop - 1),
            "start_time_s": float(time_s[start] - time_s[0]),
            "end_time_s": float(time_s[stop - 1] - time_s[0]),
            "frame_count": int(stop - start), "primary_reason": primary,
        })
    return windows


def main():
    source = load_factory_task(CSV, "pour_raw_material")
    points = np.vstack((source.left_position_m, source.right_position_m)); table, offset = .75, .15
    translation = np.array([-points[:, 0].mean(), -points[:, 1].mean(), table + offset - points[:, 2].min()])
    task = register_task(source, RigidTaskRegistration(np.eye(3), translation))
    best = dict(SELECTED_MOUNT)
    best["base_distance_m"] = float(np.linalg.norm(
        np.asarray(best["xy"]["left"]) - np.asarray(best["xy"]["right"])))
    best["adapter_height_m"] = best["shared_base_z_m"] - table
    scene = OUT.with_suffix(".scene.xml")
    build_same_model_scene(ROBOT_CONTRACTS["xarm6"], best["base_distance_m"], scene, table_height_m=table,
                           mount_xy_m=best["xy"], mount_yaw_deg=best["yaw"],
                           mount_adapter_height_m=best["adapter_height_m"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    mapped_quat = mapped_quaternions_at_joint_midpoint(model, task)
    (qpos, actual, pe, oe, strict_success, discontinuity, velocity,
     solver_diagnostics) = solve_multibranch_single_arm_method(
        model, task, mapped_quat, horizon=12, beam_width=8,
        candidate_iterations=60, velocity_limit_rad_s=3.14)
    source_intervals = np.r_[task.time_s[1] - task.time_s[0], np.diff(task.time_s)]
    execution_intervals = np.maximum.reduce((
        source_intervals,
        solver_diagnostics["left"]["required_dt_s"],
        solver_diagnostics["right"]["required_dt_s"],
    ))
    execution_time_s = np.r_[0.0, np.cumsum(execution_intervals[1:])]
    collision, state_collision_classes, edge_collision_classes = \
        audit_bimanual_collisions(model, qpos)
    synchronous = (strict_success["left"] & strict_success["right"] & ~collision)
    reasons = []
    for row in range(len(task.time_s)):
        failed = []
        for side in ("left", "right"):
            prefix = "L" if side == "left" else "R"
            reason = classify_ik_failure(
                candidate_count=solver_diagnostics[side]["candidate_count"][row],
                recovery_mode=solver_diagnostics[side]["recovery_mode"][row],
                position_error_m=pe[side][row],
                orientation_error_rad=oe[side][row],
                realized_velocity_violation=velocity[side][row])
            if reason != "ok": failed.append(f"{prefix}: {reason}")
        if collision[row]:
            classes = sorted(set(state_collision_classes[row]) |
                             set(edge_collision_classes[row]))
            failed.append("collision: " + ", ".join(classes))
        reasons.append("; ".join(failed) if failed else "ok")
    diagnostics = [FrameDiagnostics(i, float(t), "", "", False, False, str(CSV)) for i, t in enumerate(execution_time_s)]
    rendered = render_mujoco_mp4(scene, OUT, execution_time_s, qpos=qpos, diagnostics=diagnostics,
        left_targets=task.left_position_m, right_targets=task.right_position_m,
        follow_success=synchronous, failure_reasons=reasons,
        config=VideoRenderConfig(width=1280, height=720, fps=60, trajectory_radius_m=.008,
                                 camera_azimuth_deg=180, camera_elevation_deg=-18,
                                 camera_distance_scale=1.55))
    strict = strict_success
    summary = {"mode": "full_SE3_position_and_quaternion_IK", "source_rows": len(task.time_s),
        "mount": best, "mount_tilt_deg": {"left": 0., "right": 0.}, "registration_translation_m": translation.tolist(),
        "quaternion_registration": "fixed source-tool calibration aligned to mounted joint-midpoint TCP home at source row 0",
        "left": {"position_mean_mm": float(1000*pe["left"].mean()), "position_max_mm": float(1000*pe["left"].max()), "orientation_mean_deg": float(np.rad2deg(oe["left"].mean())), "orientation_max_deg": float(np.rad2deg(oe["left"].max())), "strict_coverage": float(strict["left"].mean())},
        "right": {"position_mean_mm": float(1000*pe["right"].mean()), "position_max_mm": float(1000*pe["right"].max()), "orientation_mean_deg": float(np.rad2deg(oe["right"].mean())), "orientation_max_deg": float(np.rad2deg(oe["right"].max())), "strict_coverage": float(strict["right"].mean())},
        "synchronous_strict_coverage": float(synchronous.mean()),
        "controller_joint_velocity_limit_rad_s": 3.14,
        "retiming_safety_fraction": 1.0,
        "source_duration_s": float(task.time_s[-1] - task.time_s[0]),
        "execution_duration_s": float(execution_time_s[-1]),
        "retimed_frame_count": int(np.sum(execution_intervals > source_intervals + 1e-12)),
        "plateau_stop_policy": PLATEAU_THRESHOLDS,
        "cannot_follow_frame_count": int((~synchronous).sum()),
        "collision_frame_count": int(collision.sum()),
        "failure_reason_counts": dict(Counter(reason for reason in reasons if reason != "ok")),
        "cannot_follow_windows": failure_windows(task.time_s, ~synchronous, reasons),
        "solver": {"candidate_iterations": 60, "maximum_candidates": 8,
                   "path_selection": "fixed-source-time rolling branch continuity",
                   "candidate_self_and_table_collision_filter": True,
                   "bounded_recovery_swept_collision_filter": True,
                   "post_gap_teleport_reinitialization": False},
        "decode": rendered.check.__dict__}
    OUT.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    np.savez_compressed(OUT.with_suffix(".trajectory.npz"), time_s=execution_time_s, source_time_s=task.time_s, execution_dt_s=execution_intervals, qpos=qpos, left_actual=actual["left"], right_actual=actual["right"], left_target_position=task.left_position_m, right_target_position=task.right_position_m, left_source_quaternion=task.left_quaternion_wxyz, right_source_quaternion=task.right_quaternion_wxyz, left_target_quaternion=mapped_quat["left"], right_target_quaternion=mapped_quat["right"], left_position_error_m=pe["left"], right_position_error_m=pe["right"], left_orientation_error_rad=oe["left"], right_orientation_error_rad=oe["right"], left_success=strict_success["left"], right_success=strict_success["right"], synchronous_success=synchronous, left_discontinuity=discontinuity["left"], right_discontinuity=discontinuity["right"], left_velocity_violation=velocity["left"], right_velocity_violation=velocity["right"], left_candidate_count=solver_diagnostics["left"]["candidate_count"], right_candidate_count=solver_diagnostics["right"]["candidate_count"], left_collision_free_candidate_count=solver_diagnostics["left"]["collision_free_candidate_count"], right_collision_free_candidate_count=solver_diagnostics["right"]["collision_free_candidate_count"], left_selected_candidate_index=solver_diagnostics["left"]["selected_candidate_id"], right_selected_candidate_index=solver_diagnostics["right"]["selected_candidate_id"], left_recovery_mode=solver_diagnostics["left"]["recovery_mode"], right_recovery_mode=solver_diagnostics["right"]["recovery_mode"], left_joint_limit_margin_rad=solver_diagnostics["left"]["selected_joint_limit_margin_rad"], right_joint_limit_margin_rad=solver_diagnostics["right"]["selected_joint_limit_margin_rad"], left_singularity_margin=solver_diagnostics["left"]["selected_singularity_margin"], right_singularity_margin=solver_diagnostics["right"]["selected_singularity_margin"], collision=collision, state_collision_classes=state_collision_classes, edge_collision_classes=edge_collision_classes, failure_reason=np.asarray(reasons))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()
