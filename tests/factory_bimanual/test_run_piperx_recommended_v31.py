from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run_piperx_recommended_v31 import (
    _fraction_true,
    _resolve_mount,
    _resolve_tool_offsets,
    build_summary,
    candidate_rank_key,
    condition_complete_follow_targets,
    parse_args,
    resample_task_60hz,
    scene_mount_kwargs,
    smooth_follow_targets,
)
from factory_bimanual.piperx_recommended import WorldMount
from factory_bimanual.task_family import TaskFamily


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


def _mount(mode):
    return WorldMount(
        family=TaskFamily.parse("8-11/Seal_Bag"),
        morphology="humanoid_pole",
        mode=mode,
        left_xyz_m=np.asarray([-0.3, 0.2, 1.1]),
        right_xyz_m=np.asarray([-0.2, -0.4, 1.1]),
        shared_base_z_m=1.1,
        source_take="161504",
        left_yaw_deg=-15.0,
        right_yaw_deg=20.0,
    )


def test_horizontal_forward_scene_mount_uses_explicit_base_quaternions():
    task = _task()

    kwargs, evidence = scene_mount_kwargs(
        _mount("horizontal_forward"), task, table_height_m=0.75)

    assert set(kwargs["mount_quaternion_wxyz"]) == {"left", "right"}
    assert kwargs["mount_yaw_deg"] == {"left": 0.0, "right": 0.0}
    assert kwargs["mount_support_mode"] == "horizontal_forward"
    assert evidence["orientation_representation"] == "quaternion_wxyz"
    assert evidence["coordinate_domain"] == "registered_world"


def test_upright_scene_mount_retains_explicit_yaw():
    kwargs, evidence = scene_mount_kwargs(
        _mount("upright_table"), _task(), table_height_m=0.75)

    assert kwargs["mount_quaternion_wxyz"] is None
    assert kwargs["mount_yaw_deg"] == {"left": -15.0, "right": 20.0}
    assert evidence["orientation_representation"] == "yaw_deg"


def test_candidate_ranking_never_trades_strict_coverage_for_collision():
    lower_coverage_collision_free = {
        "strict_coverage": 0.99,
        "retimed_coverage": 1.0,
        "maximum_normalized_error": 0.5,
        "collision_free_coverage": 1.0,
        "fixed_time_coverage": 1.0,
        "execution_duration_s": 20.0,
    }
    full_coverage_with_collisions = {
        **lower_coverage_collision_free,
        "strict_coverage": 1.0,
        "collision_free_coverage": 0.8,
    }

    assert candidate_rank_key(full_coverage_with_collisions) > candidate_rank_key(
        lower_coverage_collision_free)


def test_world_mount_override_is_explicit_and_preserves_yaw():
    options = parse_args([
        "--family", "8-11/Seal_Bag",
        "--mount-mode", "upright_table",
        "--mount-coordinate-domain", "registered_world",
        "--left-base-xyz-m", "-0.37", "0.18", "0.85",
        "--right-base-xyz-m", "-0.20", "-0.47", "0.85",
        "--left-yaw-deg", "0",
        "--right-yaw-deg", "30",
    ])
    base = _mount("horizontal_forward")

    mount = _resolve_mount(
        options, base,
        registration_rotation=np.eye(3),
        registration_translation_m=np.zeros(3),
    )

    assert mount.mode == "upright_table"
    assert np.allclose(mount.left_xyz_m, [-0.37, 0.18, 0.85])
    assert np.allclose(mount.right_xyz_m, [-0.20, -0.47, 0.85])
    assert mount.yaw_deg == {"left": 0.0, "right": 30.0}
    assert mount.selection_method == "explicit CLI mount override"


def test_cli_tool_offsets_override_shared_calibration_per_task():
    options = parse_args([
        "--left-tool-offset-wxyz", "1", "0", "0", "0",
        "--right-tool-offset-wxyz", "0", "1", "0", "0",
    ])
    spec = SimpleNamespace(
        left_tool_offset_quaternion_wxyz=None,
        right_tool_offset_quaternion_wxyz=None,
    )
    calibration = SimpleNamespace(
        left_offset_quaternion_wxyz=(0.5, 0.5, 0.5, 0.5),
        right_offset_quaternion_wxyz=(0.5, -0.5, 0.5, -0.5),
    )

    offsets, selection = _resolve_tool_offsets(options, spec, calibration)

    assert offsets == {
        "left": (1.0, 0.0, 0.0, 0.0),
        "right": (0.0, 1.0, 0.0, 0.0),
    }
    assert selection == "explicit CLI task-specific fixed R_tool"


def test_complete_follow_conditioning_applies_fixed_translation_and_wrist_schedule():
    task = _task()
    identity = (1.0, 0.0, 0.0, 0.0)
    spec = SimpleNamespace(
        side="right", axis="x", angle_deg=-10.0,
        hold_until_s=0.01, return_until_s=0.05,
    )

    prepared, mapped, audit = condition_complete_follow_targets(
        task,
        {"left": identity, "right": identity},
        tool_translations={
            "left": (0.01, 0.0, 0.0),
            "right": (0.0, 0.0, 0.0),
        },
        wrist_adaptation=spec,
        apply_conditioning=False,
    )

    np.testing.assert_allclose(
        prepared.left_position_m,
        task.left_position_m + np.asarray([0.01, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(prepared.right_position_m, task.right_position_m)
    assert audit.window == 0
    assert not np.allclose(mapped["right"][0], task.right_quaternion_wxyz[0])
    np.testing.assert_allclose(mapped["right"][-1], task.right_quaternion_wxyz[-1])
