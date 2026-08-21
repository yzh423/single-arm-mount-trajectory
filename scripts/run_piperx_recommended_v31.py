"""Run the PDF-recommended dual-PiperX mount and rescue-v3.1 protocol."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from factory_bimanual.artifacts import FrameDiagnostics
from factory_bimanual.complete_follow import CompleteFollowRunner
from factory_bimanual.piperx_recommended import (
    WorldMount,
    load_recommended_config,
    world_mount_for_family,
)
from factory_bimanual.mount_orientation import mount_quaternions
from factory_bimanual.recommended_follow import RecommendedFollowRunner
from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.source_data import load_factory_task
from factory_bimanual.task_family import TaskFamily
from factory_bimanual.tool_frame_calibration import CalibrationArtifact
from factory_bimanual.tool_frame_calibration import (
    apply_bounded_wrist_adaptation,
    apply_fixed_tool_rotation,
    apply_fixed_tool_translation,
)
from factory_bimanual.trajectory_conditioning import (
    ConditioningAudit,
    bounded_savgol_se3,
)
from factory_bimanual.video import VideoRenderConfig, render_mujoco_mp4
from scripts.render_factory_dual_piperx_fixed_time import (
    CALIBRATION_PATH,
    TABLE_HEIGHT_M,
)
from scripts.render_factory_dual_xarm6_fold_box import bounded_smooth_pose_series
from scripts.render_factory_dual_xarm6_se3_follow import prepare_follow_targets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/piperx_complete_follow"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="8-11/Fold_Box")
    parser.add_argument("--source-take", default="161044")
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-candidates", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--mount-mode",
        choices=("upright_table", "horizontal_wall", "horizontal_forward", "inverted"),
    )
    parser.add_argument(
        "--mount-coordinate-domain",
        choices=("registered_world", "source_frame"),
        default="registered_world",
    )
    parser.add_argument("--left-base-xyz-m", type=float, nargs=3)
    parser.add_argument("--right-base-xyz-m", type=float, nargs=3)
    parser.add_argument("--left-yaw-deg", type=float)
    parser.add_argument("--right-yaw-deg", type=float)
    parser.add_argument("--left-tool-offset-wxyz", type=float, nargs=4)
    parser.add_argument("--right-tool-offset-wxyz", type=float, nargs=4)
    parser.add_argument(
        "--condition-targets", action="store_true",
        help="apply the report's optional bounded 5 mm / 1 deg SE(3) conditioning",
    )
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args(argv)


def candidate_rank_key(metrics):
    """Return the audit-preserving lexicographic score for mount candidates."""

    return (
        float(metrics["strict_coverage"]),
        float(metrics["retimed_coverage"]),
        -float(metrics["maximum_normalized_error"]),
        float(metrics["collision_free_coverage"]),
        float(metrics["fixed_time_coverage"]),
        -float(metrics["execution_duration_s"]),
    )


def _resolve_mount(
        options, base_mount, *, registration_rotation,
        registration_translation_m):
    """Apply an explicit, auditable mount override to a configured mount."""

    left_override = options.left_base_xyz_m
    right_override = options.right_base_xyz_m
    if (left_override is None) != (right_override is None):
        raise ValueError("left and right base overrides must be provided together")
    left = np.asarray(base_mount.left_xyz_m, dtype=float).copy()
    right = np.asarray(base_mount.right_xyz_m, dtype=float).copy()
    selection_method = base_mount.selection_method
    coordinate_domain = base_mount.coordinate_domain
    if left_override is not None:
        left = np.asarray(left_override, dtype=float)
        right = np.asarray(right_override, dtype=float)
        coordinate_domain = str(options.mount_coordinate_domain)
        if coordinate_domain == "source_frame":
            rotation = np.asarray(registration_rotation, dtype=float)
            translation = np.asarray(registration_translation_m, dtype=float)
            left = rotation @ left + translation
            right = rotation @ right + translation
            coordinate_domain = "registered_world_from_source_frame_override"
        selection_method = "explicit CLI mount override"
    if not np.isclose(left[2], right[2], atol=1e-9, rtol=0.0):
        raise ValueError("mount override must use one shared base height")
    return WorldMount(
        family=base_mount.family,
        morphology=base_mount.morphology,
        mode=options.mount_mode or base_mount.mode,
        left_xyz_m=left,
        right_xyz_m=right,
        shared_base_z_m=float(0.5 * (left[2] + right[2])),
        source_take=base_mount.source_take,
        left_yaw_deg=(
            base_mount.left_yaw_deg if options.left_yaw_deg is None
            else float(options.left_yaw_deg)),
        right_yaw_deg=(
            base_mount.right_yaw_deg if options.right_yaw_deg is None
            else float(options.right_yaw_deg)),
        coordinate_domain=coordinate_domain,
        selection_method=selection_method,
    )


def _resolve_tool_offsets(options, mount_spec, calibration):
    overrides = {
        "left": options.left_tool_offset_wxyz,
        "right": options.right_tool_offset_wxyz,
    }
    if (overrides["left"] is None) != (overrides["right"] is None):
        raise ValueError("left and right tool offsets must be provided together")
    if overrides["left"] is not None:
        result = {}
        for side in ("left", "right"):
            value = np.asarray(overrides[side], dtype=float)
            norm = float(np.linalg.norm(value))
            if value.shape != (4,) or not np.isfinite(value).all() or norm < 1e-12:
                raise ValueError(f"{side} tool offset must be a finite quaternion")
            value = value / norm
            result[side] = tuple(float(item) for item in value)
        return result, "explicit CLI task-specific fixed R_tool"
    result = {
        "left": (
            mount_spec.left_tool_offset_quaternion_wxyz
            or calibration.left_offset_quaternion_wxyz),
        "right": (
            mount_spec.right_tool_offset_quaternion_wxyz
            or calibration.right_offset_quaternion_wxyz),
    }
    selection = (
        mount_spec.tool_offset_selection
        or "locked shared PiperX calibration artifact")
    return result, selection


def scene_mount_kwargs(mount, task, *, table_height_m):
    """Translate an audited world mount into scene-builder arguments."""

    yaw = mount.yaw_deg
    quaternion = None
    representation = "yaw_deg"
    if mount.mode not in ("upright_table", "baseline_frozen"):
        target_center = np.mean(np.vstack([
            np.asarray(task.left_position_m, dtype=float),
            np.asarray(task.right_position_m, dtype=float),
        ]), axis=0)
        quaternion = mount_quaternions(
            mount.mode, mount.xy, target_center, yaw)
        yaw = {"left": 0.0, "right": 0.0}
        representation = "quaternion_wxyz"
    kwargs = {
        "table_height_m": float(table_height_m),
        "mount_xy_m": mount.xy,
        "mount_yaw_deg": yaw,
        "mount_adapter_height_m": (
            float(mount.shared_base_z_m) - float(table_height_m)),
        "mount_quaternion_wxyz": quaternion,
        "mount_support_mode": (
            None if mount.mode == "baseline_frozen" else mount.mode),
    }
    evidence = {
        "coordinate_domain": mount.coordinate_domain,
        "orientation_representation": representation,
        "yaw_deg": mount.yaw_deg,
        "quaternion_wxyz": (
            None if quaternion is None
            else {side: list(quaternion[side]) for side in ("left", "right")}
        ),
    }
    return kwargs, evidence


def _slerp_wxyz(time_s, quaternion_wxyz, target_time_s):
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    rotations = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
    result = Slerp(np.asarray(time_s, dtype=float), rotations)(target_time_s)
    xyzw = result.as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def resample_task_60hz(task, *, rate_hz=60.0):
    """Resample a complete bimanual pose track onto an exact-rate timeline."""

    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("rate_hz must be finite and positive")
    source_time = np.asarray(task.time_s, dtype=float)
    relative = source_time - source_time[0]
    duration = float(relative[-1])
    target = np.arange(0.0, duration + 0.5 / rate_hz, 1.0 / rate_hz)
    if target[-1] > duration:
        target[-1] = duration
    elif duration - target[-1] > 1e-12:
        target = np.r_[target, duration]
    values = task.__dict__.copy()
    values["time_s"] = target
    values["source_row_index"] = np.searchsorted(
        relative, target, side="left").clip(0, len(relative) - 1)
    for side in ("left", "right"):
        position = np.asarray(getattr(task, f"{side}_position_m"), dtype=float)
        values[f"{side}_position_m"] = np.column_stack([
            np.interp(target, relative, position[:, axis])
            for axis in range(3)
        ])
        values[f"{side}_quaternion_wxyz"] = _slerp_wxyz(
            relative,
            getattr(task, f"{side}_quaternion_wxyz"),
            target,
        )
        valid = np.asarray(getattr(task, f"{side}_valid"), dtype=bool)
        values[f"{side}_valid"] = valid[values["source_row_index"]]
        gripper = getattr(task, f"{side}_gripper_angle_rad")
        if gripper is not None:
            values[f"{side}_gripper_angle_rad"] = np.interp(
                target, relative, np.asarray(gripper, dtype=float)
            )
    return SimpleNamespace(**values)


def smooth_follow_targets(task, mapped_quaternions):
    """Apply the report's bounded SE(3) target conditioning at 60 Hz."""

    values = task.__dict__.copy()
    mapped = {}
    for side in ("left", "right"):
        position, quaternion = bounded_smooth_pose_series(
            getattr(task, f"{side}_position_m"),
            mapped_quaternions[side],
            position_cap_m=0.003,
            orientation_cap_rad=np.deg2rad(1.0),
            sigma_frames=1.0,
        )
        values[f"{side}_position_m"] = position
        mapped[side] = quaternion
    return SimpleNamespace(**values), mapped


def condition_complete_follow_targets(
        task, tool_offsets, *, tool_translations=None,
        wrist_adaptation=None, apply_conditioning=True):
    """Apply a fixed R_tool and optionally the report's SE(3) conditioning."""
    values = task.__dict__.copy()
    translations = tool_translations or {
        side: (0.0, 0.0, 0.0) for side in ("left", "right")}
    wrist_angle_deg = {
        side: np.zeros(len(task.time_s), dtype=float)
        for side in ("left", "right")
    }
    for side in ("left", "right"):
        source_quaternion = getattr(task, f"{side}_quaternion_wxyz")
        values[f"{side}_position_m"] = apply_fixed_tool_translation(
            getattr(task, f"{side}_position_m"),
            source_quaternion,
            translations[side],
        )
        mapped = apply_fixed_tool_rotation(
            source_quaternion, tool_offsets[side])
        if wrist_adaptation is not None and wrist_adaptation.side == side:
            mapped, wrist_angle_deg[side] = apply_bounded_wrist_adaptation(
                mapped, task.time_s, wrist_adaptation)
        values[f"{side}_quaternion_wxyz"] = mapped
        values[f"{side}_wrist_adaptation_angle_deg"] = wrist_angle_deg[side]
    mapped_task = SimpleNamespace(**values)
    if not apply_conditioning:
        mapped = {
            side: getattr(mapped_task, f"{side}_quaternion_wxyz").copy()
            for side in ("left", "right")
        }
        return mapped_task, mapped, ConditioningAudit(0, 0, 0.0, 0.0)
    conditioned = bounded_savgol_se3(
        mapped_task,
        sides=("left", "right"),
        window=9,
        polyorder=3,
        maximum_position_deviation_m=0.005,
        maximum_orientation_deviation_rad=np.deg2rad(1.0),
    )
    mapped = {
        side: getattr(
            conditioned.task, f"{side}_quaternion_wxyz").copy()
        for side in ("left", "right")
    }
    return conditioned.task, mapped, conditioned.audit


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _fraction_true(values):
    mask = np.asarray(values, dtype=bool)
    return 1.0 if not mask.size else float(np.mean(mask))


def _dynamic_limits_passed(measured, limit):
    tolerance = max(1e-12, 1e-9*float(limit))
    return bool(float(measured) <= float(limit) + tolerance)


def build_summary(
    result,
    *,
    config,
    mount,
    source_path,
    source_frames,
    execution_frames,
    duration_s,
):
    states = Counter(str(value) for value in result.source_state)
    events = [_json_value(asdict(event)) for event in result.events]
    accepted = np.asarray(result.source_accepted, dtype=bool)

    def accepted_max(values, scale=1.0):
        selected = np.asarray(values, dtype=float)[accepted]
        return None if not len(selected) else float(scale * np.max(selected))

    return {
        "schema": config.schema,
        "robot": "piperx_dual",
        "source": {
            "path": str(Path(source_path).resolve()),
            "frames_60hz": int(source_frames),
            "duration_s": float(duration_s),
        },
        "mount": mount.as_scene_mount(),
        "acceptance": {
            "position_tolerance_mm": (
                1000.0 * config.accept.position_tolerance_m
            ),
            "orientation_tolerance_deg": float(
                np.rad2deg(config.accept.orientation_tolerance_rad)
            ),
            "per_joint_branch_guard_rad": config.accept.branch_guard_rad,
        },
        "protocol": {
            "anchor_restarts": int(result.anchor_restart_count),
            "later_frame_solver": "single DLS warm-start from last command",
            "failure_policy": "non-terminal HOLD; rescue_v3.1 only",
            "target_conditioning": (
                "60 Hz bounded SE(3) smoothing; 3 mm / 1 deg maximum target change"
            ),
            "execution_frames": int(execution_frames),
            "dropped_source_frames": int(result.dropped_source_frames),
            "cycle_delay_s": float(result.cycle_delay_s),
        },
        "metrics": {
            "strict_synchronous_coverage": float(
                np.mean(accepted)
            ),
            "strict_frames": int(np.count_nonzero(accepted)),
            "hold_or_recovery_frames": int(
                np.count_nonzero(~accepted)
            ),
            "collision_frames": int(np.count_nonzero(result.collision)),
            "state_counts": dict(sorted(states.items())),
            "left_position_error_mm": {
                "mean": float(1000.0 * np.mean(result.position_error_m["left"])),
                "max": float(1000.0 * np.max(result.position_error_m["left"])),
            },
            "right_position_error_mm": {
                "mean": float(1000.0 * np.mean(result.position_error_m["right"])),
                "max": float(1000.0 * np.max(result.position_error_m["right"])),
            },
            "left_orientation_error_deg": {
                "mean": float(np.rad2deg(np.mean(
                    result.orientation_error_rad["left"]
                ))),
                "max": float(np.rad2deg(np.max(
                    result.orientation_error_rad["left"]
                ))),
            },
            "right_orientation_error_deg": {
                "mean": float(np.rad2deg(np.mean(
                    result.orientation_error_rad["right"]
                ))),
                "max": float(np.rad2deg(np.max(
                    result.orientation_error_rad["right"]
                ))),
            },
            "rescue_event_count": len(events),
            "accepted_only": {
                "left_max_position_mm": accepted_max(
                    result.position_error_m["left"], 1000.0
                ),
                "right_max_position_mm": accepted_max(
                    result.position_error_m["right"], 1000.0
                ),
                "left_max_orientation_deg": accepted_max(
                    result.orientation_error_rad["left"], 180.0 / np.pi
                ),
                "right_max_orientation_deg": accepted_max(
                    result.orientation_error_rad["right"], 180.0 / np.pi
                ),
            },
        },
        "rescue_events": events,
    }


def build_complete_summary(
    result,
    *,
    config,
    mount,
    source_path,
    source_frames,
    execution_frames,
    duration_s,
    tool_offsets,
    tool_translations,
    wrist_adaptation,
    conditioning_audit,
    maximum_candidates_per_side,
):
    reached = np.asarray(result.source_reached, dtype=bool)
    fixed = np.asarray(result.fixed_time_accepted, dtype=bool)
    velocity_ok = (
        np.max(np.abs(result.source_velocity_rad_s), axis=1)
        <= config.execution.maximum_velocity_rad_s + 1e-12
    )
    acceleration_ok = (
        np.max(np.abs(result.source_acceleration_rad_s2), axis=1)
        <= config.execution.maximum_acceleration_rad_s2 + 1e-12
    )

    def reached_max(values, scale=1.0):
        selected = np.asarray(values, dtype=float)[reached]
        return None if not len(selected) else float(scale*np.max(selected))

    return {
        "schema": "piperx-complete-follow-v2",
        "robot": "piperx_dual",
        "source": {
            "path": str(Path(source_path).resolve()),
            "frames_60hz": int(source_frames),
            "duration_s": float(duration_s),
            "target_basis": (
                "registered_resampled_calibrated_tcp"
                if conditioning_audit.window == 0 else
                "registered_resampled_calibrated_tcp_conditioned"
            ),
            "raw_hand_trace_preserved": True,
            "tracking_reference": "task-level calibrated PiperX TCP",
        },
        "mount": mount.as_scene_mount(),
        "tool_frame": {
            "policy": "task-scoped strict representative-pose feasibility probe",
            "left_offset_quaternion_wxyz": np.asarray(
                tool_offsets["left"], dtype=float).tolist(),
            "right_offset_quaternion_wxyz": np.asarray(
                tool_offsets["right"], dtype=float).tolist(),
            "translation_coordinate_frame": "registered source-hand local",
            "left_translation_m": np.asarray(
                tool_translations["left"], dtype=float).tolist(),
            "right_translation_m": np.asarray(
                tool_translations["right"], dtype=float).tolist(),
            "wrist_adaptation": (
                None if wrist_adaptation is None else asdict(wrist_adaptation)
            ),
            "tracking_error_reference": "calibrated TCP target",
        },
        "acceptance": {
            "position_tolerance_mm": (
                1000.0*config.accept.position_tolerance_m),
            "orientation_tolerance_deg": float(np.rad2deg(
                config.accept.orientation_tolerance_rad)),
            "per_joint_branch_guard_rad": config.accept.branch_guard_rad,
        },
        "protocol": {
            "anchor_restarts": config.anchor_restarts,
            "warm_start_iterations": config.dls.max_iterations,
            "global_rescue": "failed side only; collision may expand both sides",
            "maximum_candidates_per_side": int(maximum_candidates_per_side),
            "failure_policy": "no source pose dropped; lossless retiming",
            "side_state_machines": "independent until paired collision coordination",
            "target_conditioning": {
                "method": (
                    "disabled; exact registered 60 Hz target"
                    if conditioning_audit.window == 0 else
                    "Savitzky-Golay SE(3), window 9, polyorder 3"
                ),
                "enabled": bool(conditioning_audit.window),
                "position_deviation_limit_mm": (
                    0.0 if conditioning_audit.window == 0 else 5.0),
                "orientation_deviation_limit_deg": (
                    0.0 if conditioning_audit.window == 0 else 1.0),
                "measured_max_position_deviation_mm": (
                    1000.0*conditioning_audit.maximum_position_deviation_m),
                "measured_max_orientation_deviation_deg": float(np.rad2deg(
                    conditioning_audit.maximum_orientation_deviation_rad)),
            },
            "dls": {
                "damping": config.dls.damping,
                "maximum_step_rad": config.dls.maximum_step_rad,
                "position_error_clip_m": config.dls.position_error_clip_m,
                "orientation_error_clip_rad": (
                    config.dls.orientation_error_clip_rad),
            },
            "execution_frames": int(execution_frames),
            "inserted_transition_frames": int(
                result.inserted_transition_frames),
            "cycle_delay_s": float(result.cycle_delay_s),
            "time_scale": float(result.time_scale),
            "retiming_method": (
                "lossless local segment stretching with explicit branch transitions"
            ),
            "retimed_duration_s": float(
                result.execution_time_s[-1]-result.execution_time_s[0]),
            "dynamic_limits": {
                "maximum_velocity_rad_s": (
                    config.execution.maximum_velocity_rad_s),
                "maximum_acceleration_rad_s2": (
                    config.execution.maximum_acceleration_rad_s2),
                "measured_execution_maximum_velocity_rad_s": (
                    result.maximum_velocity_rad_s),
                "measured_execution_maximum_acceleration_rad_s2": (
                    result.maximum_acceleration_rad_s2),
            },
        },
        "metrics": {
            "complete_source_pose_coverage": _fraction_true(reached),
            "complete_source_pose_frames": int(np.count_nonzero(reached)),
            "fixed_time_synchronous_coverage": _fraction_true(fixed),
            "fixed_time_synchronous_frames": int(np.count_nonzero(fixed)),
            "original_time_velocity_edge_coverage": _fraction_true(velocity_ok),
            "original_time_velocity_edges": int(np.count_nonzero(velocity_ok)),
            "original_time_acceleration_knot_coverage": _fraction_true(
                acceleration_ok),
            "original_time_acceleration_knots": int(np.count_nonzero(
                acceleration_ok)),
            "original_time_maximum_velocity_rad_s": float(np.max(
                np.abs(result.source_velocity_rad_s), initial=0.0)),
            "original_time_maximum_acceleration_rad_s2": float(np.max(
                np.abs(result.source_acceleration_rad_s2), initial=0.0)),
            "retimed_execution_dynamic_limits_passed": bool(
                _dynamic_limits_passed(
                    result.maximum_velocity_rad_s,
                    config.execution.maximum_velocity_rad_s,
                )
                and _dynamic_limits_passed(
                    result.maximum_acceleration_rad_s2,
                    config.execution.maximum_acceleration_rad_s2,
                )
            ),
            "unreached_source_pose_frames": int(np.count_nonzero(~reached)),
            "collision_frames": int(np.count_nonzero(
                result.source_collision)),
            "collision_free_strict_coverage": float(np.mean(
                reached & ~np.asarray(result.source_collision, dtype=bool))),
            "execution_collision_frames": int(np.count_nonzero(
                result.execution_collision)),
            "global_rescue_frames": {
                side: int(np.count_nonzero(
                    result.per_side_global_rescue[side]))
                for side in ("left", "right")
            },
            "left_position_error_mm": {
                "mean": float(1000.0*np.mean(
                    result.position_error_m["left"])),
                "max": float(1000.0*np.max(
                    result.position_error_m["left"])),
            },
            "right_position_error_mm": {
                "mean": float(1000.0*np.mean(
                    result.position_error_m["right"])),
                "max": float(1000.0*np.max(
                    result.position_error_m["right"])),
            },
            "left_orientation_error_deg": {
                "mean": float(np.rad2deg(np.mean(
                    result.orientation_error_rad["left"]))),
                "max": float(np.rad2deg(np.max(
                    result.orientation_error_rad["left"]))),
            },
            "right_orientation_error_deg": {
                "mean": float(np.rad2deg(np.mean(
                    result.orientation_error_rad["right"]))),
                "max": float(np.rad2deg(np.max(
                    result.orientation_error_rad["right"]))),
            },
            "reached_only": {
                "left_max_position_mm": reached_max(
                    result.position_error_m["left"], 1000.0),
                "right_max_position_mm": reached_max(
                    result.position_error_m["right"], 1000.0),
                "left_max_orientation_deg": reached_max(
                    result.orientation_error_rad["left"], 180.0/np.pi),
                "right_max_orientation_deg": reached_max(
                    result.orientation_error_rad["right"], 180.0/np.pi),
            },
        },
    }


def _source_path(family: TaskFamily, source_take: str) -> Path:
    directory = ROOT / "data/factory" / family.date / family.task
    matches = sorted(directory.glob(f"*_{source_take}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one take {source_take} under {directory}"
        )
    return matches[0]


def _registered_source(path: Path, family: TaskFamily):
    source = load_factory_task(
        path, family.key, max_translation_jump_m=0.07,
        repair_invalid_pose_rows=True,
    )
    points = np.vstack((source.left_position_m, source.right_position_m))
    translation = np.asarray((
        -points[:, 0].mean(),
        -points[:, 1].mean(),
        0.90 - points[:, 2].min(),
    ))
    registration = RigidTaskRegistration(np.eye(3), translation)
    return register_task(source, registration), registration


def _slice_task(task, count: int | None):
    if count is None or count >= len(task.time_s):
        return task
    if count < 2:
        raise ValueError("max-frames must be at least 2")
    values = {
        key: (value[:count] if isinstance(value, np.ndarray)
              and value.ndim and len(value) == len(task.time_s) else value)
        for key, value in task.__dict__.items()
    }
    return SimpleNamespace(**values)


def _write_qa_frames(video_path: Path, output_dir: Path):
    capture = cv2.VideoCapture(str(video_path))
    count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    outputs = []
    for label, frame in zip(("start", "middle", "end"), (0, count // 2, count - 1)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame))
        ok, image = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"cannot decode QA frame {frame}")
        path = output_dir / f"qa_{label}.png"
        cv2.imwrite(str(path), image)
        outputs.append(path)
    capture.release()
    return outputs


def run(options):
    config = load_recommended_config()
    family = TaskFamily.parse(options.family)
    source_path = _source_path(family, options.source_take)
    task, registration = _registered_source(source_path, family)
    task = _slice_task(
        resample_task_60hz(task, rate_hz=options.rate_hz),
        options.max_frames,
    )
    registered_task = task
    configured_mount = world_mount_for_family(
        config, family,
        registration.rotation_world_from_vr,
        registration.translation_world_m,
    )
    mount = _resolve_mount(
        options, configured_mount,
        registration_rotation=registration.rotation_world_from_vr,
        registration_translation_m=registration.translation_world_m,
    )
    output_dir = Path(options.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{family.date}_{family.task}_{options.source_take}_complete_follow"
    scene = output_dir / f"{stem}.scene.xml"
    scene_kwargs, mount_orientation_evidence = scene_mount_kwargs(
        mount, task, table_height_m=TABLE_HEIGHT_M)
    manifest = build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], mount.base_distance_m, scene,
        **scene_kwargs,
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    calibration = CalibrationArtifact.read(CALIBRATION_PATH)
    mount_spec = config.mounts[family.key]
    tool_offsets, tool_offset_selection = _resolve_tool_offsets(
        options, mount_spec, calibration)
    task, mapped_quaternions, conditioning_audit = (
        condition_complete_follow_targets(
            task, tool_offsets,
            tool_translations={
                "left": mount_spec.left_tool_translation_m,
                "right": mount_spec.right_tool_translation_m,
            },
            wrist_adaptation=mount_spec.wrist_adaptation,
            apply_conditioning=options.condition_targets,
        ))
    runner = CompleteFollowRunner(
        model, task, mapped_quaternions, config,
        maximum_candidates_per_side=options.maximum_candidates,
    )
    result = runner.run()
    summary = build_complete_summary(
        result,
        config=config,
        mount=mount,
        source_path=source_path,
        source_frames=len(task.time_s),
        execution_frames=len(result.execution_time_s),
        duration_s=float(task.time_s[-1] - task.time_s[0]),
        tool_offsets=tool_offsets,
        tool_translations={
            "left": mount_spec.left_tool_translation_m,
            "right": mount_spec.right_tool_translation_m,
        },
        wrist_adaptation=mount_spec.wrist_adaptation,
        conditioning_audit=conditioning_audit,
        maximum_candidates_per_side=options.maximum_candidates,
    )
    summary["source"]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    summary["mount"].update(mount_orientation_evidence)
    summary["tool_frame"]["selection"] = tool_offset_selection
    summary["scene_manifest"] = asdict(manifest)
    summary_path = output_dir / f"{stem}.summary.json"
    trajectory_path = output_dir / f"{stem}.trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        source_time_s=np.asarray(task.time_s),
        execution_time_s=result.execution_time_s,
        execution_qpos=result.execution_qpos,
        source_qpos=result.source_qpos,
        source_index=result.execution_source_index,
        execution_state=result.execution_state,
        source_execution_index=result.source_execution_index,
        source_reached=result.source_reached,
        fixed_time_accepted=result.fixed_time_accepted,
        collision=result.source_collision,
        execution_collision=result.execution_collision,
        execution_state_collision=result.execution_state_collision,
        execution_incoming_transition_collision=(
            result.execution_incoming_transition_collision),
        left_target_position_m=task.left_position_m,
        right_target_position_m=task.right_position_m,
        left_target_quaternion_wxyz=mapped_quaternions["left"],
        right_target_quaternion_wxyz=mapped_quaternions["right"],
        raw_left_hand_position_m=registered_task.left_position_m,
        raw_right_hand_position_m=registered_task.right_position_m,
        raw_left_hand_quaternion_wxyz=registered_task.left_quaternion_wxyz,
        raw_right_hand_quaternion_wxyz=registered_task.right_quaternion_wxyz,
        left_wrist_adaptation_angle_deg=(
            task.left_wrist_adaptation_angle_deg),
        right_wrist_adaptation_angle_deg=(
            task.right_wrist_adaptation_angle_deg),
        left_actual_tcp=result.actual_tcp["left"],
        right_actual_tcp=result.actual_tcp["right"],
        left_position_error_m=result.position_error_m["left"],
        right_position_error_m=result.position_error_m["right"],
        left_orientation_error_rad=result.orientation_error_rad["left"],
        right_orientation_error_rad=result.orientation_error_rad["right"],
        left_candidate_count=result.per_side_candidate_count["left"],
        right_candidate_count=result.per_side_candidate_count["right"],
        left_global_rescue=result.per_side_global_rescue["left"],
        right_global_rescue=result.per_side_global_rescue["right"],
        source_velocity_rad_s=result.source_velocity_rad_s,
        source_acceleration_rad_s2=result.source_acceleration_rad_s2,
    )
    if not options.no_video:
        execution_target_left = task.left_position_m[
            result.execution_source_index]
        execution_target_right = task.right_position_m[
            result.execution_source_index]
        reasons = [
            (f"{value}_COLLISION" if result.execution_collision[row]
             else str(value))
            for row, value in enumerate(result.execution_state)
        ]
        execution_success = np.ones(len(result.execution_state), dtype=bool)
        diagnostics = [
            FrameDiagnostics(
                int(source_index), float(task.time_s[source_index]),
                reasons[row], reasons[row],
                bool(result.execution_collision[row]), False,
                str(source_path.resolve()),
                execution_index=row,
                execution_time_s=float(result.execution_time_s[row]),
                execution_state=reasons[row],
                state_collision=bool(result.execution_state_collision[row]),
                incoming_transition_collision=bool(
                    result.execution_incoming_transition_collision[row]),
            )
            for row, source_index in enumerate(result.execution_source_index)
        ]
        video_path = output_dir / f"{stem}.mp4"
        rendered = render_mujoco_mp4(
            scene, video_path, result.execution_time_s,
            qpos=result.execution_qpos,
            diagnostics=diagnostics,
            left_targets=execution_target_left,
            right_targets=execution_target_right,
            follow_success=execution_success,
            failure_reasons=reasons,
            timeline_domain="execution",
            config=VideoRenderConfig(
                width=1280, height=720, fps=30.0,
                interpolate_states=True,
                camera_azimuth_deg=185.0,
                camera_elevation_deg=-18.0,
                camera_distance_scale=1.25,
                title="Dual PiperX - raw-pose complete follow (1 mm / 0.5 deg)",
            ),
        )
        qa_frames = _write_qa_frames(video_path, output_dir)
        summary["video"] = {
            "path": str(video_path),
            "decode_check": asdict(rendered.check),
            "qa_frames": [str(path) for path in qa_frames],
            "renderer": (
                f"MuJoCo {mujoco.__version__} offscreen, "
                "official PiperX meshes"),
            "timeline_domain": "execution",
            "state_sampling": "linear interpolation between retimed execution knots",
        }
    artifact_paths = {
        "trajectory_npz": trajectory_path,
        "scene_xml": scene,
    }
    if not options.no_video:
        artifact_paths.update({
            "video_mp4": video_path,
            "video_provenance_json": video_path.with_suffix(
                ".provenance.json"),
        })
    summary["artifacts"] = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in artifact_paths.items()
    }
    summary_path.write_text(
        json.dumps(_json_value(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "summary": summary_path,
        "trajectory": trajectory_path,
        "scene": scene,
        "video": None if options.no_video else video_path,
    }


def main(argv=None):
    outputs = run(parse_args(argv))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
