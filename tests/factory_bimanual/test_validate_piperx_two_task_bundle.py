import hashlib
import json

import numpy as np
import pytest

from scripts.validate_piperx_two_task_bundle import validate_task_bundle


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path):
    stem = "8-11_Seal_Bag_161504_complete_follow"
    trajectory = tmp_path / f"{stem}.trajectory.npz"
    scene = tmp_path / f"{stem}.scene.xml"
    video = tmp_path / f"{stem}.mp4"
    provenance = tmp_path / f"{stem}.provenance.json"
    np.savez_compressed(
        trajectory,
        source_time_s=np.asarray([0.0, 1.0]),
        execution_time_s=np.asarray([0.0, 2.0]),
        source_reached=np.asarray([True, True]),
        fixed_time_accepted=np.asarray([True, False]),
        collision=np.asarray([False, True]),
        execution_collision=np.asarray([False, True]),
        left_position_error_m=np.asarray([0.0002, 0.0009]),
        right_position_error_m=np.asarray([0.0003, 0.0008]),
        left_orientation_error_rad=np.deg2rad([0.1, 0.4]),
        right_orientation_error_rad=np.deg2rad([0.2, 0.3]),
    )
    scene.write_text("<mujoco/>", encoding="utf-8")
    video.write_bytes(b"video fixture")
    provenance.write_text(json.dumps({
        "schema_version": 2,
        "timeline_domain": "execution",
        "decode_check": {"frame_count": 61, "fps": 30.0,
                         "duration_s": 2.033333333333333},
        "execution_knots": [{}, {}],
        "encoded_frames": [{}] * 61,
    }), encoding="utf-8")
    artifacts = {
        "trajectory_npz": trajectory,
        "scene_xml": scene,
        "video_mp4": video,
        "video_provenance_json": provenance,
    }
    summary = {
        "schema": "piperx-complete-follow-v2",
        "source": {"path": "handheld_20260811_161504.csv",
                   "frames_60hz": 2, "duration_s": 1.0,
                   "target_basis": "registered_resampled_raw"},
        "mount": {"family": "8-11/Seal_Bag"},
        "acceptance": {"position_tolerance_mm": 1.0,
                       "orientation_tolerance_deg": 0.5},
        "protocol": {"execution_frames": 2, "retimed_duration_s": 2.0,
                     "cycle_delay_s": 1.0,
                     "dynamic_limits": {
                         "measured_execution_maximum_velocity_rad_s": 1.0,
                         "measured_execution_maximum_acceleration_rad_s2": 4.0}},
        "metrics": {
            "complete_source_pose_frames": 2,
            "complete_source_pose_coverage": 1.0,
            "fixed_time_synchronous_frames": 1,
            "collision_frames": 1,
            "execution_collision_frames": 1,
            "left_position_error_mm": {"max": 0.9},
            "right_position_error_mm": {"max": 0.8},
            "left_orientation_error_deg": {"max": 0.4},
            "right_orientation_error_deg": {"max": 0.3},
        },
        "video": {"decode_check": {"frame_count": 61, "fps": 30.0,
                                   "duration_s": 2.033333333333333}},
        "artifacts": {
            key: {"path": str(path), "sha256": _sha(path),
                  "size_bytes": path.stat().st_size}
            for key, path in artifacts.items()
        },
    }
    summary_path = tmp_path / f"{stem}.summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_validator_recomputes_task_metrics_and_hashes(tmp_path):
    summary = _bundle(tmp_path)

    result = validate_task_bundle(
        summary, family="8-11/Seal_Bag", take="161504",
        frames=2, decode_video=False)

    assert result["pose_frames"] == 2
    assert result["collision_frames"] == 1
    assert result["video_frames"] == 61


def test_validator_rejects_tampered_trajectory(tmp_path):
    summary = _bundle(tmp_path)
    trajectory = next(tmp_path.glob("*.trajectory.npz"))
    trajectory.write_bytes(trajectory.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        validate_task_bundle(
            summary, family="8-11/Seal_Bag", take="161504",
            frames=2, decode_video=False)
