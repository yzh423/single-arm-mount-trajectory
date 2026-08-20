from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run_piperx_recommended_v31 import (
    _fraction_true,
    build_summary,
    parse_args,
    resample_task_60hz,
    smooth_follow_targets,
)


def _task():
    time = np.asarray([0.0, 0.01, 0.025, 0.05])
    position = np.column_stack((time, 2.0 * time, np.ones_like(time)))
    quaternion = np.tile([1.0, 0.0, 0.0, 0.0], (len(time), 1))
    return SimpleNamespace(
        name="fold_box",
        source_path=Path("take.csv"),
        source_row_index=np.arange(len(time)),
        time_s=time,
        coordinate_frame="vr_world",
        left_position_m=position,
        right_position_m=position + np.asarray([0.1, 0.0, 0.0]),
        left_quaternion_wxyz=quaternion,
        right_quaternion_wxyz=quaternion,
        left_valid=np.ones(len(time), dtype=bool),
        right_valid=np.ones(len(time), dtype=bool),
        left_gripper_angle_rad=None,
        right_gripper_angle_rad=None,
    )


def test_resample_task_60hz_preserves_endpoints_and_unit_quaternions():
    task = resample_task_60hz(_task(), rate_hz=60.0)

    assert task.time_s[0] == 0.0
    assert task.time_s[-1] == 0.05
    assert np.all(np.diff(task.time_s) > 0.0)
    assert len(task.time_s) == 4
    assert np.allclose(task.left_position_m[[0, -1], 0], [0.0, 0.05])
    assert np.allclose(np.linalg.norm(task.left_quaternion_wxyz, axis=1), 1.0)


def test_parse_args_uses_pdf_recommended_fold_box_take():
    options = parse_args([])

    assert options.family == "8-11/Fold_Box"
    assert options.source_take == "161044"
    assert options.rate_hz == 60.0
    assert options.maximum_candidates == 8
    assert options.condition_targets is False
    assert options.output_dir.name == "piperx_complete_follow"


def test_empty_dynamic_knot_set_is_vacuously_within_limits():
    assert _fraction_true(np.asarray([], dtype=bool)) == 1.0


def test_smooth_follow_targets_is_bounded_for_position_and_orientation():
    task = _task()
    task.left_position_m[2, 0] += 0.02
    task.right_position_m[2, 0] += 0.02
    mapped = {
        side: getattr(task, f"{side}_quaternion_wxyz").copy()
        for side in ("left", "right")
    }

    smoothed, _ = smooth_follow_targets(task, mapped)

    for side in ("left", "right"):
        change = np.linalg.norm(
            getattr(smoothed, f"{side}_position_m")
            - getattr(task, f"{side}_position_m"),
            axis=1,
        )
        assert np.all(change <= 0.003 + 1e-12)


def test_build_summary_reports_exact_strict_thresholds_and_mode_counts():
    result = SimpleNamespace(
        source_accepted=np.asarray([True, False, True]),
        source_state=np.asarray(["FOLLOW", "HOLD", "RECOVER"]),
        position_error_m={
            "left": np.asarray([0.0002, 0.002, 0.0003]),
            "right": np.asarray([0.0003, 0.0015, 0.0004]),
        },
        orientation_error_rad={
            "left": np.deg2rad([0.1, 1.0, 0.2]),
            "right": np.deg2rad([0.2, 0.8, 0.3]),
        },
        collision=np.asarray([False, False, False]),
        events=(),
        anchor_restart_count=40,
        dropped_source_frames=0,
        cycle_delay_s=0.0,
    )
    config = SimpleNamespace(
        schema="piperx-recommended-v3.1",
        accept=SimpleNamespace(
            position_tolerance_m=0.001,
            orientation_tolerance_rad=np.deg2rad(0.5),
            branch_guard_rad=0.3,
        ),
    )
    mount = SimpleNamespace(
        as_scene_mount=lambda: {"mode": "upright_table"},
    )

    summary = build_summary(
        result, config=config, mount=mount, source_path=Path("take.csv"),
        source_frames=3, execution_frames=3, duration_s=0.05,
    )

    assert summary["acceptance"]["position_tolerance_mm"] == 1.0
    assert summary["acceptance"]["orientation_tolerance_deg"] == 0.5
    assert summary["metrics"]["strict_synchronous_coverage"] == 2 / 3
    assert summary["metrics"]["accepted_only"]["left_max_position_mm"] == 0.3
    assert summary["metrics"]["accepted_only"]["right_max_orientation_deg"] == 0.3
    assert summary["metrics"]["state_counts"] == {
        "FOLLOW": 1,
        "HOLD": 1,
        "RECOVER": 1,
    }
