import hashlib
import json

import numpy as np
import pytest

from scripts.validate_piperx_two_task_bundle import validate_task_bundle


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path, *, collision=False):
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
        collision=np.asarray([False, collision]),
        execution_collision=np.asarray([False, collision]),
        raw_left_hand_position_m=np.zeros((2, 3)),
        raw_right_hand_position_m=np.zeros((2, 3)),
        raw_left_hand_quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        raw_right_hand_quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        left_target_position_m=np.tile([0.001, 0.0, 0.0], (2, 1)),
        right_target_position_m=np.tile([-0.001, 0.0, 0.0], (2, 1)),
        left_wrist_adaptation_angle_deg=np.zeros(2),
        right_wrist_adaptation_angle_deg=np.zeros(2),
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
                   "target_basis": "registered_resampled_calibrated_tcp",
                   "raw_hand_trace_preserved": True,
                   "tracking_reference": "task-level calibrated PiperX TCP"},
        "mount": {"family": "8-11/Seal_Bag"},
        "tool_frame": {
            "translation_coordinate_frame": "registered source-hand local",
            "left_translation_m": [0.001, 0.0, 0.0],
            "right_translation_m": [-0.001, 0.0, 0.0],
            "wrist_adaptation": None,
            "tracking_error_reference": "calibrated TCP target",
        },
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
            "retimed_execution_dynamic_limits_passed": True,
            "fixed_time_synchronous_frames": 1,
            "collision_frames": int(collision),
            "execution_collision_frames": int(collision),
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
    assert result["collision_frames"] == 0
    assert result["video_frames"] == 61


def test_validator_rejects_any_published_collision(tmp_path):
    summary = _bundle(tmp_path, collision=True)

    with pytest.raises(ValueError, match="zero collision"):
        validate_task_bundle(
            summary, family="8-11/Seal_Bag", take="161504",
            frames=2, decode_video=False)


def test_validator_requires_calibrated_tcp_target_basis(tmp_path):
    summary_path = _bundle(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source"]["target_basis"] = "registered_resampled_raw"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="calibrated TCP"):
        validate_task_bundle(
            summary_path, family="8-11/Seal_Bag", take="161504",
            frames=2, decode_video=False)


def test_validator_rejects_missing_raw_hand_evidence(tmp_path):
    summary_path = _bundle(tmp_path)
    trajectory = next(tmp_path.glob("*.trajectory.npz"))
    with np.load(trajectory, allow_pickle=False) as archive:
        payload = {
            name: archive[name] for name in archive.files
            if name != "raw_left_hand_position_m"
        }
    np.savez_compressed(trajectory, **payload)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = summary["artifacts"]["trajectory_npz"]
    record["sha256"] = _sha(trajectory)
    record["size_bytes"] = trajectory.stat().st_size
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="raw hand evidence"):
        validate_task_bundle(
            summary_path, family="8-11/Seal_Bag", take="161504",
            frames=2, decode_video=False)


def test_validator_rejects_same_length_tool_translation_in_wrong_direction(
        tmp_path):
    summary_path = _bundle(tmp_path)
    trajectory = next(tmp_path.glob("*.trajectory.npz"))
    with np.load(trajectory, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["left_target_position_m"] = np.tile(
        [0.0, 0.001, 0.0], (2, 1))
    np.savez_compressed(trajectory, **payload)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = summary["artifacts"]["trajectory_npz"]
    record["sha256"] = _sha(trajectory)
    record["size_bytes"] = trajectory.stat().st_size
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
            ValueError, match="calibrated TCP translation evidence"):
        validate_task_bundle(
            summary_path, family="8-11/Seal_Bag", take="161504",
            frames=2, decode_video=False)


def test_validator_rejects_tampered_trajectory(tmp_path):
    summary = _bundle(tmp_path)
    trajectory = next(tmp_path.glob("*.trajectory.npz"))
    trajectory.write_bytes(trajectory.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        validate_task_bundle(
            summary, family="8-11/Seal_Bag", take="161504",
            frames=2, decode_video=False)
