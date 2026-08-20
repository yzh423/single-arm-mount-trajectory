import json

import pytest

from scripts.build_piperx_two_task_report import load_report_data


def _manifest():
    task = {
        "task": "Fold_Box",
        "family": "8-11/Fold_Box",
        "take": "161044",
        "source_frames": 2,
        "pose_frames": 2,
        "pose_coverage": 1.0,
        "fixed_time_frames": 1,
        "fixed_time_coverage": 0.5,
        "collision_frames": 0,
        "collision_free_coverage": 1.0,
        "source_duration_s": 1.0,
        "execution_duration_s": 2.0,
        "cycle_delay_s": 1.0,
        "maximum_velocity_rad_s": 1.0,
        "maximum_acceleration_rad_s2": 4.0,
        "left_max_position_mm": 0.9,
        "right_max_position_mm": 0.8,
        "left_max_orientation_deg": 0.4,
        "right_max_orientation_deg": 0.3,
        "video_frames": 61,
        "video_fps": 30.0,
        "video_duration_s": 2.0,
        "summary": "fold.summary.json",
        "summary_sha256": "a" * 64,
        "artifacts": {},
    }
    seal = {**task, "task": "Seal_Bag", "family": "8-11/Seal_Bag",
            "take": "161504"}
    return {
        "schema": "piperx-two-task-complete-follow-v1",
        "acceptance": {"position_tolerance_mm": 1.0,
                       "orientation_tolerance_deg": 0.5},
        "tasks": [task, seal],
    }


def test_report_loader_requires_two_complete_raw_pose_tasks(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    data = load_report_data(path)

    assert set(data) == {"Fold_Box", "Seal_Bag"}


def test_report_loader_rejects_non_complete_pose_claim(tmp_path):
    payload = _manifest()
    payload["tasks"][1]["pose_frames"] = 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="complete raw-pose"):
        load_report_data(path)
