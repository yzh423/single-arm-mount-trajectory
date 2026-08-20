"""Solve and render dual-PiperX factory tasks on immutable source time."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from factory_bimanual.artifacts import FrameDiagnostics
from factory_bimanual.fixed_time_run_contract import (
    source_time_schedule,
    validate_synchronized_mount,
)
from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.source_data import load_factory_task
from factory_bimanual.staged_mount_search import (
    MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M,
)
from factory_bimanual.tool_frame_calibration import (
    CalibrationArtifact,
    source_file_fingerprint,
)
from factory_bimanual.video import VideoRenderConfig, render_mujoco_mp4
from scripts.render_factory_dual_xarm6_se3_follow import (
    audit_bimanual_collisions,
    failure_windows,
    prepare_follow_targets,
    solve_collision_safe_bimanual_method,
    solve_multibranch_single_arm_method,
)
from scripts.render_factory_dual_xarm6_fold_box import subset
from scripts.strict_mujoco_ik import joint_periodic_mask
from scripts.search_fold_box_piperx_paired_mount import _scene_mount_kwargs


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = (
    ROOT / "reports/factory_bimanual/piperx_tool_frame_calibration.json")
TABLE_HEIGHT_M = 0.75
VELOCITY_LIMIT_RAD_S = 3.14


def comparison_planning_indices(count: int, *, stride: int = 4) -> np.ndarray:
    if count < 1 or stride < 1:
        raise ValueError("count and stride must be positive")
    return np.unique(np.r_[np.arange(0, count, stride, dtype=int), count - 1])


def interpolate_joint_path(source_time_s, q, target_time_s, *, periodic):
    source_time_s = np.asarray(source_time_s, float)
    target_time_s = np.asarray(target_time_s, float)
    q = np.asarray(q, float); periodic = np.asarray(periodic, bool)
    if q.shape != (len(source_time_s), len(periodic)):
        raise ValueError("q shape must match source time and periodic mask")
    unwrapped = q.copy()
    if len(q) > 1 and np.any(periodic):
        delta = np.diff(unwrapped[:, periodic], axis=0)
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        unwrapped[1:, periodic] = unwrapped[0, periodic] + np.cumsum(delta, axis=0)
    result = np.column_stack([
        np.interp(target_time_s, source_time_s, unwrapped[:, joint])
        for joint in range(q.shape[1])])
    return result


def _solve_decimated_comparison(model, task, mapped_quat, *, stride,
                                planner="independent"):
    indices = comparison_planning_indices(len(task.time_s), stride=stride)
    planned_task = subset(task, indices)
    planned_quat = {side: np.asarray(mapped_quat[side])[indices]
                    for side in ("left", "right")}
    if planner == "paired":
        result = solve_collision_safe_bimanual_method(
            model, planned_task, planned_quat,
            velocity_limit_rad_s=VELOCITY_LIMIT_RAD_S,
            robot_name="piperx")
    else:
        result = solve_multibranch_single_arm_method(
            model, planned_task, planned_quat,
            horizon=12, beam_width=8, candidate_iterations=60,
            velocity_limit_rad_s=VELOCITY_LIMIT_RAD_S,
            global_retimed_no_flip=False, robot_name="piperx",
            global_seed_count=2, maximum_candidates=3,
            constrained_fallback_enabled=True)
    planned_qpos, _, _, _, _, _, _, planned_diagnostics = result
    qpos = np.repeat(model.qpos0[None, :], len(task.time_s), axis=0)
    qpos_ids = {}; periodic = {}
    for side in ("left", "right"):
        joint_ids = np.asarray([mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{i}")
            for i in range(1, 7)], int)
        qpos_ids[side] = np.asarray(model.jnt_qposadr[joint_ids], int)
        periodic[side] = joint_periodic_mask(model, joint_ids)
        qpos[:, qpos_ids[side]] = interpolate_joint_path(
            planned_task.time_s, planned_qpos[:, qpos_ids[side]], task.time_s,
            periodic=periodic[side])
    actual = {}; pe = {}; oe = {}; success = {}; discontinuity = {}; velocity = {}
    data = mujoco.MjData(model)
    dt = np.r_[task.time_s[1] - task.time_s[0], np.diff(task.time_s)]
    floor_plan = np.searchsorted(indices, np.arange(len(task.time_s)), side="right") - 1
    floor_plan = np.clip(floor_plan, 0, len(indices) - 1)
    diagnostics = {}
    for side in ("left", "right"):
        reached = np.zeros((len(task.time_s), 7)); pos_error = np.zeros(len(task.time_s))
        ori_error = np.zeros(len(task.time_s)); site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        for row in range(len(task.time_s)):
            data.qpos[:] = qpos[row]; mujoco.mj_forward(model, data)
            quaternion = np.empty(4); mujoco.mju_mat2Quat(
                quaternion, data.site_xmat[site_id])
            reached[row] = np.r_[data.site_xpos[site_id], quaternion]
            pos_error[row] = np.linalg.norm(
                getattr(task, f"{side}_position_m")[row] - data.site_xpos[site_id])
            residual = np.empty(3); mujoco.mju_subQuat(
                residual, mapped_quat[side][row], quaternion)
            ori_error[row] = np.linalg.norm(residual)
        joint_delta = np.diff(qpos[:, qpos_ids[side]], axis=0)
        joint_delta[:, periodic[side]] = (
            joint_delta[:, periodic[side]] + np.pi) % (2 * np.pi) - np.pi
        velocity_bad = np.zeros(len(task.time_s), bool)
        velocity_bad[1:] = np.any(
            np.abs(joint_delta) / dt[1:, None] > VELOCITY_LIMIT_RAD_S + 1e-10,
            axis=1)
        actual[side] = reached; pe[side] = pos_error; oe[side] = ori_error
        success[side] = ((pos_error <= .001) &
                         (ori_error <= np.deg2rad(1.5)) & ~velocity_bad)
        discontinuity[side] = np.r_[False, np.any(np.abs(joint_delta) > np.deg2rad(35), axis=1)]
        velocity[side] = velocity_bad
        source = planned_diagnostics[side]
        diagnostics[side] = {
            key: np.asarray(value)[floor_plan]
            for key, value in source.items()
        }
        diagnostics[side]["planning_stride"] = np.full(len(task.time_s), stride)
    return qpos, actual, pe, oe, success, discontinuity, velocity, diagnostics


@dataclass(frozen=True)
class TaskSpec:
    name: str
    csv: Path
    report_directory: Path
    output_stem: str
    max_translation_jump_m: float | None


def task_spec(task_name: str) -> TaskSpec:
    if task_name == "fold_box":
        return TaskSpec(
            task_name,
            ROOT / "data/factory/8-11/Fold_Box/handheld_20260811_160754.csv",
            ROOT / "reports/factory_bimanual/fold_box_dual_piperx",
            "fold_box_piperx_xarm6_style_fixed_time_front_720p",
            0.07,
        )
    if task_name == "seal_bag":
        return TaskSpec(
            task_name,
            ROOT / "data/factory/8-11/Seal_Bag/handheld_20260811_161504.csv",
            ROOT / "reports/factory_bimanual/seal_bag_dual_piperx",
            "seal_bag_piperx_xarm6_style_fixed_time_front_720p",
            None,
        )
    raise ValueError("task_name must be fold_box or seal_bag")


def load_locked_piperx_calibration(path=None):
    calibration_path = CALIBRATION_PATH if path is None else Path(path)
    expected = {
        task: source_file_fingerprint(task_spec(task).csv)
        for task in ("fold_box", "seal_bag")
    }
    return CalibrationArtifact.read(
        calibration_path, expected_source_fingerprints=expected)


def fixed_time_timing_audit(source_time_s, execution_time_s):
    expected, _ = source_time_schedule(source_time_s)
    execution = np.asarray(execution_time_s, dtype=float)
    if (execution.shape != expected.shape or
            not np.allclose(execution, expected, rtol=0.0, atol=1e-12)):
        raise ValueError("fixed-time execution must equal source timestamps exactly")
    duration = float(expected[-1])
    return {
        "timing_mode": "fixed_source_time",
        "source_duration_s": duration,
        "execution_duration_s": duration,
        "retimed_frame_count": 0,
        "added_duration_s": 0.0,
    }


def validate_saved_trajectory(source_time_s, payload):
    """Reject cached solves that are incomplete, retimed, or off-source-time."""
    source = np.asarray(source_time_s, dtype=float)
    expected, _ = source_time_schedule(source)
    cached_source = np.asarray(payload["source_time_s"], dtype=float)
    execution = np.asarray(payload["time_s"], dtype=float)
    retimed = np.asarray(payload["time_retimed"], dtype=bool)
    qpos = np.asarray(payload["qpos"], dtype=float)
    required_rows = ("collision", "failure_reason", "synchronous_success")
    if (cached_source.shape != source.shape or
            not np.allclose(cached_source, source, rtol=0.0, atol=1e-12) or
            execution.shape != expected.shape or
            not np.allclose(execution, expected, rtol=0.0, atol=1e-12)):
        raise ValueError("cached trajectory does not match source timeline")
    if retimed.shape != source.shape or np.any(retimed):
        raise ValueError("cached trajectory contains retimed frames")
    if qpos.ndim != 2 or len(qpos) != len(source):
        raise ValueError("cached qpos row count is incomplete")
    for name in required_rows:
        if len(np.asarray(payload[name])) != len(source):
            raise ValueError(f"cached {name} row count is incomplete")
    return expected


def _registered_task(spec: TaskSpec):
    kwargs = ({"max_translation_jump_m": spec.max_translation_jump_m}
              if spec.max_translation_jump_m is not None else {})
    source = load_factory_task(spec.csv, spec.name, **kwargs)
    points = np.vstack((source.left_position_m, source.right_position_m))
    translation = np.asarray((
        -points[:, 0].mean(),
        -points[:, 1].mean(),
        0.90 - points[:, 2].min(),
    ))
    return register_task(
        source, RigidTaskRegistration(np.eye(3), translation)), translation


def _realized_velocity_violations(model, qpos, intervals_s):
    result = {}
    for side in ("left", "right"):
        joint_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{index}")
            for index in range(1, 7)]
        qids = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
        delta = np.diff(np.asarray(qpos)[:, qids], axis=0)
        violation = np.zeros(len(qpos), dtype=bool)
        violation[1:] = np.any(
            np.abs(delta) / intervals_s[1:, None] >
            VELOCITY_LIMIT_RAD_S + 1e-10,
            axis=1,
        )
        result[side] = violation
    return result


def _camera(task_name, mount):
    if task_name == "seal_bag":
        return 180.0, -18.0, 1.55
    baseline = (np.asarray(mount["xy"]["right"], dtype=float) -
                np.asarray(mount["xy"]["left"], dtype=float))
    azimuth = float((np.degrees(np.arctan2(baseline[1], baseline[0]))
                     - 90.0) % 360.0)
    return azimuth, -12.0, 1.48


def mount_separation_for_run(mount, *, allow_legacy_mount=False):
    """Validate a mount while making legacy close-spacing use explicit."""
    minimum = (np.finfo(float).eps if allow_legacy_mount else
               MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M)
    return validate_synchronized_mount(mount, minimum_separation_m=minimum)


def solve_fixed_time_motion(model, task, mapped_quat, *,
                            solver_method="xarm6-style",
                            solver_profile="default"):
    """Run one independent source-time solve with an explicit planner."""
    if solver_method == "xarm6-style":
        if solver_profile not in ("default", "comparison"):
            raise ValueError("solver_profile must be default or comparison")
        if solver_profile == "comparison":
            return _solve_decimated_comparison(
                model, task, mapped_quat, stride=4)
        kwargs = {}
        return solve_multibranch_single_arm_method(
            model, task, mapped_quat,
            horizon=12, beam_width=8, candidate_iterations=60,
            velocity_limit_rad_s=VELOCITY_LIMIT_RAD_S,
            global_retimed_no_flip=False, robot_name="piperx", **kwargs)
    if solver_method == "paired":
        if solver_profile == "comparison":
            return _solve_decimated_comparison(
                model, task, mapped_quat, stride=4, planner="paired")
        return solve_collision_safe_bimanual_method(
            model, task, mapped_quat,
            velocity_limit_rad_s=VELOCITY_LIMIT_RAD_S,
            robot_name="piperx", strict_pose_only=True,
            reference_aware_enabled=True,
            orientation_adaptation_enabled=True)
    raise ValueError("solver_method must be xarm6-style or paired")


def run_task(task_name, mount, output: Path | None = None, *,
             render_video=True, allow_legacy_mount=False,
             solver_method="xarm6-style", camera_override=None,
             solver_profile="default"):
    spec = task_spec(task_name)
    mount = json.loads(json.dumps(mount))
    separation = mount_separation_for_run(
        mount, allow_legacy_mount=allow_legacy_mount)
    mount["base_distance_m"] = separation
    if "mode" not in mount:
        mount["adapter_height_m"] = (
            float(mount["shared_base_z_m"]) - TABLE_HEIGHT_M)
    if allow_legacy_mount:
        mount["selection_method"] = (
            "original PiperX mount reused for independent fixed-source-time "
            "solve; no mount search")
    else:
        mount["selection_method"] = (
            "paired synchronized PiperX fixed-source-time mount search with "
            "full-timeline state and swept-edge collision audit")
    spec.report_directory.mkdir(parents=True, exist_ok=True)
    output = Path(output) if output is not None else (
        spec.report_directory / f"{spec.output_stem}.mp4")

    task, translation = _registered_task(spec)
    calibration = load_locked_piperx_calibration()
    scene = output.with_suffix(".scene.xml")
    build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], separation, scene,
        table_height_m=TABLE_HEIGHT_M,
        mount_xy_m=mount["xy"], mount_yaw_deg=mount["yaw"],
        **(_scene_mount_kwargs(task, mount) if "mode" in mount else
           {"mount_adapter_height_m": mount["adapter_height_m"]}),
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    task, mapped_quaternion = prepare_follow_targets(
        model, task, calibration=calibration)
    (qpos, actual, position_error, orientation_error, strict_success,
     discontinuity, _planner_velocity, solver_diagnostics) = \
        solve_fixed_time_motion(
            model, task, mapped_quaternion, solver_method=solver_method,
            solver_profile=solver_profile)
    execution_time_s, source_intervals = source_time_schedule(task.time_s)
    timing = fixed_time_timing_audit(task.time_s, execution_time_s)
    velocity = _realized_velocity_violations(
        model, qpos, source_intervals)
    collision, state_classes, edge_classes = audit_bimanual_collisions(
        model, qpos, robot_name="piperx")
    paired = solver_diagnostics.get("paired")
    if paired is not None and (paired.collision_count or np.any(collision)):
        raise RuntimeError("fixed-time paired planner produced a collision")
    synchronous = (
        strict_success["left"] & strict_success["right"] &
        ~velocity["left"] & ~velocity["right"] & ~collision)
    if paired is not None:
        synchronous &= paired.followed

    reasons = []
    for row in range(len(task.time_s)):
        failed = []
        if collision[row]:
            classes = sorted(set(state_classes[row]) | set(edge_classes[row]))
            failed.append("COLLISION: " + ", ".join(classes))
        if paired is not None and not paired.followed[row]:
            failed.append(
                "PAIRED: " + str(paired.failure_reason[row]).upper())
        for side, prefix in (("left", "L"), ("right", "R")):
            if velocity[side][row]:
                failed.append(f"{prefix}: SOURCE TIMING VELOCITY INFEASIBLE")
            elif (not strict_success[side][row] and
                  not (paired is not None and not paired.followed[row])):
                recovery = str(solver_diagnostics[side]["recovery_mode"][row])
                if "collision" in recovery:
                    failed.append(f"{prefix}: COLLISION BLOCKED SAFE HOLD")
                elif int(solver_diagnostics[side]["candidate_count"][row]) == 0:
                    failed.append(f"{prefix}: POSE UNREACHABLE")
                else:
                    failed.append(f"{prefix}: SOURCE TIMING INFEASIBLE")
        reasons.append("; ".join(failed) if failed else "ok")

    rendered = None
    if render_video:
        diagnostics = [FrameDiagnostics(
            row, float(timestamp), reasons[row], reasons[row],
            bool(collision[row]), False, "independent_fixed_source_time",
        ) for row, timestamp in enumerate(execution_time_s)]
        azimuth, elevation, distance = (
            _camera(task_name, mount) if camera_override is None
            else camera_override)
        rendered = render_mujoco_mp4(
            scene, output, execution_time_s,
            qpos=qpos, diagnostics=diagnostics,
            left_targets=task.left_position_m,
            right_targets=task.right_position_m,
            follow_success=synchronous,
            failure_reasons=reasons,
            config=VideoRenderConfig(
                width=1280, height=720, fps=60,
                trajectory_radius_m=0.008,
                interpolate_states=False,
                camera_azimuth_deg=azimuth,
                camera_elevation_deg=elevation,
                camera_distance_scale=distance,
                title=f"Dual official PiPER-X {task_name} | fixed time",
            ),
        )

    summary = {
        "mode": (
            "xarm6_style_fixed_source_time_SE3_IK"
            if solver_method == "xarm6-style" else
            "paired_collision_safe_fixed_source_time_SE3_IK"),
        "task": task_name,
        "robot": "piperx",
        "source_rows": len(task.time_s),
        "mount": mount,
        "registration_translation_m": translation.tolist(),
        "tool_frame_calibration": {
            "artifact": str(CALIBRATION_PATH.resolve()),
            "left_offset_quaternion_wxyz": list(
                calibration.left_offset_quaternion_wxyz),
            "right_offset_quaternion_wxyz": list(
                calibration.right_offset_quaternion_wxyz),
            "metrics": calibration.metrics,
        },
        **timing,
        "controller_joint_velocity_limit_rad_s": VELOCITY_LIMIT_RAD_S,
        "left": {
            "strict_coverage": float(strict_success["left"].mean()),
            "position_mean_mm": float(1000.0 * position_error["left"].mean()),
            "orientation_mean_deg": float(np.rad2deg(
                orientation_error["left"].mean())),
            "velocity_infeasible_frame_count": int(velocity["left"].sum()),
        },
        "right": {
            "strict_coverage": float(strict_success["right"].mean()),
            "position_mean_mm": float(1000.0 * position_error["right"].mean()),
            "orientation_mean_deg": float(np.rad2deg(
                orientation_error["right"].mean())),
            "velocity_infeasible_frame_count": int(velocity["right"].sum()),
        },
        "synchronous_strict_coverage": float(synchronous.mean()),
        "cannot_follow_frame_count": int((~synchronous).sum()),
        "collision_frame_count": int(collision.sum()),
        "clearance": ({
            "hard_margin_m": .015,
            "minimum_state_or_swept_m": float(
                paired.minimum_clearance_m),
            "limiting_pair": paired.limiting_clearance_pair,
            "state_margin_violation_frame_count": int(np.sum(
                paired.state_clearance_m < .015)),
            "swept_margin_violation_edge_count": int(np.sum(
                paired.swept_clearance_m < .015)),
        } if paired is not None else None),
        "paired_failure_taxonomy_counts": (
            dict(Counter(str(reason) for reason in paired.failure_reason
                         if str(reason) != "ok"))
            if paired is not None else {}),
        "failure_reason_counts": dict(Counter(
            reason for reason in reasons if reason != "ok")),
        "cannot_follow_windows": failure_windows(
            task.time_s, ~synchronous, reasons),
        "solver": {
            "method": solver_method,
            "profile": solver_profile,
            "path_selection": (
                "xArm6-style independent rolling multibranch on immutable "
                "source intervals" if solver_method == "xarm6-style" else
                "collision-hard paired beam on immutable source intervals"),
            "source_timestamps_immutable": True,
            "retiming_allowed": False,
        },
        "decode": None if rendered is None else rendered.check.__dict__,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    np.savez_compressed(
        output.with_suffix(".trajectory.npz"),
        time_s=execution_time_s,
        source_time_s=task.time_s,
        source_dt_s=source_intervals,
        execution_dt_s=source_intervals.copy(),
        time_retimed=np.zeros(len(task.time_s), dtype=bool),
        qpos=qpos,
        left_actual=actual["left"], right_actual=actual["right"],
        left_target_position=task.left_position_m,
        right_target_position=task.right_position_m,
        left_target_quaternion=mapped_quaternion["left"],
        right_target_quaternion=mapped_quaternion["right"],
        left_position_error_m=position_error["left"],
        right_position_error_m=position_error["right"],
        left_orientation_error_rad=orientation_error["left"],
        right_orientation_error_rad=orientation_error["right"],
        left_success=strict_success["left"],
        right_success=strict_success["right"],
        synchronous_success=synchronous,
        left_discontinuity=discontinuity["left"],
        right_discontinuity=discontinuity["right"],
        left_velocity_violation=velocity["left"],
        right_velocity_violation=velocity["right"],
        left_tolerance_tier=(
            paired.left_tier if paired is not None else
            np.full(len(task.time_s), "xarm6_style_exact")),
        right_tolerance_tier=(
            paired.right_tier if paired is not None else
            np.full(len(task.time_s), "xarm6_style_exact")),
        collision=collision,
        state_collision_classes=state_classes,
        edge_collision_classes=edge_classes,
        failure_reason=np.asarray(reasons),
        paired_failure_reason=(
            paired.failure_reason if paired is not None else
            np.full(len(task.time_s), "not_paired")),
        state_clearance_m=(
            paired.state_clearance_m if paired is not None else
            np.full(len(task.time_s), np.nan)),
        swept_clearance_m=(
            paired.swept_clearance_m if paired is not None else
            np.full(len(task.time_s), np.nan)),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def _summary_for_trajectory(path: Path) -> Path:
    suffix = ".trajectory.npz"
    if path.name.endswith(suffix):
        return path.with_name(path.name[:-len(suffix)] + ".summary.json")
    return path.with_suffix(".summary.json")


def render_saved_run(task_name, mount, trajectory_path, output,
                     camera_override=None):
    """Render a fully validated fixed-time solve without running IK again."""
    spec = task_spec(task_name)
    trajectory_path = Path(trajectory_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    task, _ = _registered_task(spec)
    with np.load(trajectory_path, allow_pickle=False) as loaded:
        required = (
            "source_time_s", "time_s", "time_retimed", "qpos",
            "collision", "failure_reason", "synchronous_success",
        )
        payload = {name: loaded[name] for name in required}
    execution_time_s = validate_saved_trajectory(task.time_s, payload)

    mount = json.loads(json.dumps(mount))
    separation = mount_separation_for_run(mount)
    mount["base_distance_m"] = separation
    if "mode" not in mount:
        mount["adapter_height_m"] = (
            float(mount["shared_base_z_m"]) - TABLE_HEIGHT_M)
    scene = output.with_suffix(".scene.xml")
    build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], separation, scene,
        table_height_m=TABLE_HEIGHT_M,
        mount_xy_m=mount["xy"], mount_yaw_deg=mount["yaw"],
        **(_scene_mount_kwargs(task, mount) if "mode" in mount else
           {"mount_adapter_height_m": mount["adapter_height_m"]}),
    )
    reasons = np.asarray(payload["failure_reason"]).astype(str)
    collision = np.asarray(payload["collision"], dtype=bool)
    synchronous = np.asarray(payload["synchronous_success"], dtype=bool)
    diagnostics = [FrameDiagnostics(
        row, float(timestamp), reasons[row], reasons[row],
        bool(collision[row]), False, "cached_fixed_source_time",
    ) for row, timestamp in enumerate(execution_time_s)]
    azimuth, elevation, distance = (
        _camera(task_name, mount) if camera_override is None
        else camera_override)
    rendered = render_mujoco_mp4(
        scene, output, execution_time_s,
        qpos=np.asarray(payload["qpos"]), diagnostics=diagnostics,
        left_targets=task.left_position_m,
        right_targets=task.right_position_m,
        follow_success=synchronous,
        failure_reasons=reasons.tolist(),
        config=VideoRenderConfig(
            width=1280, height=720, fps=60,
            trajectory_radius_m=0.008,
            interpolate_states=False,
            camera_azimuth_deg=azimuth,
            camera_elevation_deg=elevation,
            camera_distance_scale=distance,
            title=f"Dual official PiPER-X {task_name} | fixed time",
        ),
    )

    summary_path = _summary_for_trajectory(trajectory_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["mount"] = mount
    summary["decode"] = rendered.check.__dict__
    summary["rendered_from_trajectory_cache"] = str(
        trajectory_path.resolve())
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    target_trajectory = output.with_suffix(".trajectory.npz")
    if trajectory_path.resolve() != target_trajectory.resolve():
        shutil.copy2(trajectory_path, target_trajectory)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("fold_box", "seal_bag"))
    parser.add_argument("mount_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trajectory-cache", type=Path,
        help="render this completed fixed-time trajectory without rerunning IK")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--allow-legacy-mount", action="store_true")
    parser.add_argument(
        "--solver-method", choices=("xarm6-style", "paired"),
        default="xarm6-style")
    args = parser.parse_args(argv)
    payload = json.loads(args.mount_json.read_text(encoding="utf-8"))
    mount = payload.get("selected_mount", payload)
    if args.trajectory_cache is not None:
        if args.no_video:
            parser.error("--trajectory-cache cannot be combined with --no-video")
        if args.output is None:
            parser.error("--trajectory-cache requires --output")
        render_saved_run(args.task, mount, args.trajectory_cache, args.output)
        return
    run_task(
        args.task, mount, output=args.output,
        render_video=not args.no_video,
        allow_legacy_mount=args.allow_legacy_mount,
        solver_method=args.solver_method)


if __name__ == "__main__":
    main()
