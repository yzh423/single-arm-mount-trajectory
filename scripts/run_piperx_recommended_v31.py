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
from factory_bimanual.piperx_recommended import (
    load_recommended_config,
    world_mount_for_family,
)
from factory_bimanual.recommended_follow import RecommendedFollowRunner
from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.source_data import load_factory_task
from factory_bimanual.task_family import TaskFamily
from factory_bimanual.tool_frame_calibration import CalibrationArtifact
from factory_bimanual.video import VideoRenderConfig, render_mujoco_mp4
from scripts.render_factory_dual_piperx_fixed_time import (
    CALIBRATION_PATH,
    TABLE_HEIGHT_M,
)
from scripts.render_factory_dual_xarm6_fold_box import bounded_smooth_pose_series
from scripts.render_factory_dual_xarm6_se3_follow import prepare_follow_targets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/piperx_recommended_v31"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="8-11/Fold_Box")
    parser.add_argument("--source-take", default="161044")
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-candidates", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args(argv)


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
    mount = world_mount_for_family(
        config, family,
        registration.rotation_world_from_vr,
        registration.translation_world_m,
    )
    if mount.mode != "upright_table":
        raise ValueError(
            "the evidence render currently targets the PDF's upright Fold_Box mount"
        )
    output_dir = Path(options.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{family.date}_{family.task}_{options.source_take}_recommended_v31"
    scene = output_dir / f"{stem}.scene.xml"
    manifest = build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], mount.base_distance_m, scene,
        table_height_m=TABLE_HEIGHT_M,
        mount_xy_m=mount.xy,
        mount_yaw_deg={"left": 0.0, "right": 0.0},
        mount_adapter_height_m=mount.shared_base_z_m - TABLE_HEIGHT_M,
        mount_support_mode=mount.mode,
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    calibration = CalibrationArtifact.read(CALIBRATION_PATH)
    task, mapped_quaternions = prepare_follow_targets(
        model, task, calibration=calibration
    )
    task, mapped_quaternions = smooth_follow_targets(task, mapped_quaternions)
    runner = RecommendedFollowRunner(
        model, task, mapped_quaternions, config,
        maximum_candidates_per_side=options.maximum_candidates,
    )
    result = runner.run()
    summary = build_summary(
        result,
        config=config,
        mount=mount,
        source_path=source_path,
        source_frames=len(task.time_s),
        execution_frames=len(result.execution_time_s),
        duration_s=float(task.time_s[-1] - task.time_s[0]),
    )
    summary["source"]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    summary["scene_manifest"] = asdict(manifest)
    summary_path = output_dir / f"{stem}.summary.json"
    trajectory_path = output_dir / f"{stem}.trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        source_time_s=np.asarray(task.time_s),
        execution_time_s=result.execution_time_s,
        execution_qpos=result.execution_qpos,
        source_qpos=result.source_qpos,
        source_index=result.source_index,
        source_state=result.source_state,
        source_accepted=result.source_accepted,
        collision=result.collision,
        left_target_position_m=task.left_position_m,
        right_target_position_m=task.right_position_m,
        left_actual_tcp=result.actual_tcp["left"],
        right_actual_tcp=result.actual_tcp["right"],
        left_position_error_m=result.position_error_m["left"],
        right_position_error_m=result.position_error_m["right"],
        left_orientation_error_rad=result.orientation_error_rad["left"],
        right_orientation_error_rad=result.orientation_error_rad["right"],
        candidate_pair_count=result.candidate_pair_count,
        warm_start_attempted=result.warm_start_attempted,
    )
    if not options.no_video:
        execution_target_left = task.left_position_m[result.source_index]
        execution_target_right = task.right_position_m[result.source_index]
        reasons = [str(value) for value in result.schedule.state]
        diagnostics = [
            FrameDiagnostics(
                int(source_index), float(result.execution_time_s[row]),
                reasons[row], reasons[row], False, False,
                "recommended_v3.1",
            )
            for row, source_index in enumerate(result.source_index)
        ]
        video_path = output_dir / f"{stem}.mp4"
        rendered = render_mujoco_mp4(
            scene, video_path, result.execution_time_s,
            qpos=result.execution_qpos,
            diagnostics=diagnostics,
            left_targets=execution_target_left,
            right_targets=execution_target_right,
            follow_success=result.schedule.accepted,
            failure_reasons=reasons,
            config=VideoRenderConfig(
                width=1280, height=720, fps=60.0,
                interpolate_states=False,
                camera_azimuth_deg=185.0,
                camera_elevation_deg=-18.0,
                camera_distance_scale=1.25,
                title="Dual PiperX - PDF recommended v3.1 (1 mm / 0.5 deg)",
            ),
        )
        qa_frames = _write_qa_frames(video_path, output_dir)
        summary["video"] = {
            "path": str(video_path),
            "decode_check": asdict(rendered.check),
            "qa_frames": [str(path) for path in qa_frames],
            "renderer": "MuJoCo 3.11 offscreen, official PiperX meshes",
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
