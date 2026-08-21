"""Build and validate true fixed-source-time PiperX evidence and videos."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from factory_bimanual.artifacts import FrameDiagnostics
from factory_bimanual.video import (
    VideoRenderConfig,
    decode_check_mp4,
    render_mujoco_mp4,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "reports/piperx_two_task_complete_follow"
DEFAULT_OUTPUT = ROOT / "reports/piperx_two_task_fixed_time"
TASKS = {
    "fold_box": ("Fold_Box", "8-11_Fold_Box_161044"),
    "seal_bag": ("Seal_Bag", "8-11_Seal_Bag_161504"),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_fixed_time_arrays(source):
    required = {
        "source_time_s", "source_qpos", "source_reached",
        "fixed_time_accepted", "collision", "source_velocity_rad_s",
        "source_acceleration_rad_s2",
    }
    missing = required.difference(source)
    if missing:
        raise ValueError(f"source evidence is missing: {sorted(missing)}")
    reached = np.asarray(source["source_reached"], dtype=bool)
    collision = np.asarray(source["collision"], dtype=bool)
    if not np.all(reached):
        raise ValueError("fixed-time bundle requires complete source reach")
    if np.any(collision):
        raise ValueError("fixed-time bundle requires zero collision source path")
    payload = {
        name: np.asarray(value).copy()
        for name, value in source.items()
        if not name.startswith("execution_")
        and name not in {"source_index", "source_execution_index"}
    }
    payload.update({
        "fixed_time_s": np.asarray(source["source_time_s"], dtype=float).copy(),
        "fixed_time_qpos": np.asarray(source["source_qpos"], dtype=float).copy(),
        "fixed_time_state": np.full(len(reached), "FOLLOW_FIXED_TIME"),
        "dynamic_limit_violation": ~np.asarray(
            source["fixed_time_accepted"], dtype=bool),
        "timing_mode": np.asarray("fixed_source_time"),
        "retiming_applied": np.asarray(False),
    })
    validate_fixed_time_arrays(payload)
    return payload


def validate_fixed_time_arrays(payload):
    source_time = np.asarray(payload["source_time_s"], dtype=float)
    fixed_time = np.asarray(payload["fixed_time_s"], dtype=float)
    source_qpos = np.asarray(payload["source_qpos"], dtype=float)
    fixed_qpos = np.asarray(payload["fixed_time_qpos"], dtype=float)
    if (source_time.ndim != 1 or len(source_time) == 0
            or np.any(np.diff(source_time) <= 0.0)):
        raise ValueError("source timestamps must be finite and increasing")
    if not np.array_equal(fixed_time, source_time):
        raise ValueError("fixed time must equal source timestamps exactly")
    if not np.array_equal(fixed_qpos, source_qpos):
        raise ValueError("fixed time must use source qpos exactly")
    if np.asarray(payload["timing_mode"]).item() != "fixed_source_time":
        raise ValueError("timing mode must be fixed_source_time")
    if bool(np.asarray(payload["retiming_applied"]).item()):
        raise ValueError("fixed-time bundle cannot be retimed")
    if np.any(np.asarray(payload["collision"], dtype=bool)):
        raise ValueError("fixed-time source path must retain zero collision")
    if not np.all(np.asarray(payload["source_reached"], dtype=bool)):
        raise ValueError("fixed-time source path must retain complete reach")
    return True


def _source_paths(task_key, stem):
    directory = SOURCE_ROOT / task_key
    return {
        "summary": directory / f"{stem}_complete_follow.summary.json",
        "trajectory": directory / f"{stem}_complete_follow.trajectory.npz",
        "scene": directory / f"{stem}_complete_follow.scene.xml",
    }


def _artifact(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def build_task(task_key, output_root=DEFAULT_OUTPUT, *, render_video=True):
    task_name, stem = TASKS[task_key]
    source_paths = _source_paths(task_key, stem)
    source_summary = json.loads(
        source_paths["summary"].read_text(encoding="utf-8"))
    with np.load(source_paths["trajectory"], allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}
    fixed = build_fixed_time_arrays(source)
    output_dir = Path(output_root) / task_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{stem}_fixed_time"
    trajectory_path = output_dir / f"{output_stem}.trajectory.npz"
    scene_path = output_dir / f"{output_stem}.scene.xml"
    summary_path = output_dir / f"{output_stem}.summary.json"
    np.savez_compressed(trajectory_path, **fixed)
    shutil.copy2(source_paths["scene"], scene_path)

    source_time = np.asarray(fixed["fixed_time_s"], dtype=float)
    collision = np.asarray(fixed["collision"], dtype=bool)
    dynamic_bad = np.asarray(fixed["dynamic_limit_violation"], dtype=bool)
    diagnostics = [
        FrameDiagnostics(
            row, float(timestamp), "FOLLOW_FIXED_TIME",
            "dynamic_limit_violation" if dynamic_bad[row] else "ok",
            bool(collision[row]), False, str(source_paths["trajectory"].resolve()),
            execution_index=row, execution_time_s=float(timestamp),
            execution_state="FOLLOW_FIXED_TIME",
            state_collision=bool(collision[row]),
            incoming_transition_collision=False,
        )
        for row, timestamp in enumerate(source_time)
    ]
    artifacts = {
        "trajectory_npz": _artifact(trajectory_path),
        "scene_xml": _artifact(scene_path),
    }
    video_record = None
    if render_video:
        video_path = output_dir / f"{output_stem}.mp4"
        rendered = render_mujoco_mp4(
            scene_path, video_path, source_time,
            qpos=np.asarray(fixed["fixed_time_qpos"], dtype=float),
            diagnostics=diagnostics,
            left_targets=np.asarray(fixed["left_target_position_m"], dtype=float),
            right_targets=np.asarray(fixed["right_target_position_m"], dtype=float),
            follow_success=np.asarray(fixed["source_reached"], dtype=bool),
            failure_reasons=np.where(
                dynamic_bad, "dynamic_limit_violation", "ok").tolist(),
            timeline_domain="fixed_source_time",
            config=VideoRenderConfig(
                width=1280, height=720, fps=30.0,
                interpolate_states=True,
                camera_azimuth_deg=185.0,
                camera_elevation_deg=-18.0,
                camera_distance_scale=1.25,
                title=(
                    "Dual PiperX - FIXED SOURCE TIME "
                    "(kinematic follow; dynamics exceeded)"),
            ),
        )
        artifacts["video_mp4"] = _artifact(video_path)
        artifacts["video_provenance_json"] = _artifact(
            video_path.with_suffix(".provenance.json"))
        video_record = {
            "path": str(video_path.resolve()),
            "decode_check": asdict(rendered.check),
            "timeline_domain": "fixed_source_time",
            "state_sampling": "linear interpolation between source-time knots",
        }

    fixed_ok = np.asarray(fixed["fixed_time_accepted"], dtype=bool)
    vmax = float(np.max(np.abs(fixed["source_velocity_rad_s"]), initial=0.0))
    amax = float(np.max(np.abs(fixed["source_acceleration_rad_s2"]), initial=0.0))
    summary = {
        "schema": "piperx-two-task-fixed-time-v1",
        "task": task_name,
        "family": source_summary["mount"]["family"],
        "timing": {
            "mode": "fixed_source_time",
            "retiming_applied": False,
            "source_frames": len(source_time),
            "fixed_time_frames": len(source_time),
            "source_duration_s": float(source_time[-1] - source_time[0]),
            "fixed_time_duration_s": float(source_time[-1] - source_time[0]),
            "inserted_frames": 0,
            "added_duration_s": 0.0,
            "timestamp_identity": True,
            "qpos_identity": True,
        },
        "tracking": {
            "target_basis": source_summary["source"]["target_basis"],
            "position_tolerance_mm": 1.0,
            "orientation_tolerance_deg": 0.5,
            "reached_frames": int(np.count_nonzero(fixed["source_reached"])),
            "coverage": float(np.mean(fixed["source_reached"])),
            "collision_frames": int(np.count_nonzero(collision)),
        },
        "dynamics": {
            "maximum_velocity_rad_s": vmax,
            "maximum_acceleration_rad_s2": amax,
            "limit_velocity_rad_s": 1.0,
            "limit_acceleration_rad_s2": 4.0,
            "accepted_frames": int(np.count_nonzero(fixed_ok)),
            "accepted_coverage": float(np.mean(fixed_ok)),
            "limits_passed": bool(np.all(fixed_ok)),
            "interpretation": (
                "kinematic fixed-time evidence only; not dynamically "
                "executable on the configured PiperX limits"),
        },
        "source_evidence": {
            "summary": _artifact(source_paths["summary"]),
            "trajectory": _artifact(source_paths["trajectory"]),
        },
        "video": video_record,
        "artifacts": artifacts,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return summary_path


def validate_task(summary_path, *, decode_video=True):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "piperx-two-task-fixed-time-v1":
        raise ValueError("unexpected fixed-time schema")
    timing = summary["timing"]
    if (timing["mode"] != "fixed_source_time"
            or timing["retiming_applied"] is not False
            or timing["inserted_frames"] != 0
            or timing["added_duration_s"] != 0.0
            or timing["timestamp_identity"] is not True
            or timing["qpos_identity"] is not True):
        raise ValueError("fixed-time identity contract is absent")
    trajectory_record = summary["artifacts"]["trajectory_npz"]
    trajectory_path = summary_path.parent / Path(trajectory_record["path"]).name
    if _artifact(trajectory_path)["sha256"] != trajectory_record["sha256"]:
        raise ValueError("fixed-time trajectory hash mismatch")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        validate_fixed_time_arrays(archive)
    if summary["tracking"]["coverage"] != 1.0:
        raise ValueError("fixed-time bundle lost complete tracking")
    if summary["tracking"]["collision_frames"] != 0:
        raise ValueError("fixed-time bundle is not collision-free")
    if summary["dynamics"]["limits_passed"] is not False:
        raise ValueError("fixed-time bundle must disclose dynamic violations")
    if decode_video and summary["video"] is not None:
        video = summary_path.parent / Path(summary["video"]["path"]).name
        recorded = summary["video"]["decode_check"]
        check = decode_check_mp4(
            video, expected_frames=int(recorded["frame_count"]),
            expected_resolution=(1280, 720),
            expected_duration_s=float(recorded["duration_s"]))
        if check.fps != 30.0:
            raise ValueError("fixed-time video must use 30 fps")
    return summary


def build_bundle(output_root=DEFAULT_OUTPUT, *, render_video=True):
    output_root = Path(output_root)
    records = {}
    for task_key in TASKS:
        summary_path = build_task(
            task_key, output_root, render_video=render_video)
        records[task_key] = validate_task(
            summary_path, decode_video=render_video)
    manifest = {
        "schema": "piperx-two-task-fixed-time-manifest-v1",
        "timing_mode": "fixed_source_time",
        "retiming_applied": False,
        "tasks": records,
    }
    manifest_path = output_root / "fixed_time_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        for task_key, (_, stem) in TASKS.items():
            validate_task(
                args.output_root / task_key / f"{stem}_fixed_time.summary.json")
        print(args.output_root / "fixed_time_manifest.json")
        return
    print(build_bundle(args.output_root, render_video=not args.no_video))


if __name__ == "__main__":
    main()
