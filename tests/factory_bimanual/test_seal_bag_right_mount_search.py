import numpy as np

from scripts.search_seal_bag_right_mount import (
    generate_right_mount_candidates,
    longest_failure_run,
    right_mount_score,
    whole_trajectory_window_indices,
    exhaustive_tabletop_right_mounts,
    visual_wrist_flip_mask,
)
from scripts.render_factory_dual_xarm6_se3_follow import SELECTED_MOUNT
from scripts.render_factory_dual_xarm6_se3_follow import output_path_for_robot
from scripts.render_factory_dual_xarm6_se3_follow import parse_run_options
from scripts.render_factory_dual_xarm6_se3_follow import mount_selection_method


def test_mount_candidates_change_only_right_upright_pose():
    candidates = generate_right_mount_candidates(
        left_xy=(-0.31, 0.26), right_xy=(-0.22, -0.34),
        left_yaw_deg=315.0, right_yaw_deg=45.0,
        xy_offsets_m=(-0.05, 0.0, 0.05), yaw_offsets_deg=(-15.0, 0.0, 15.0),
        shared_base_z_m=0.87,
    )
    assert len(candidates) == 27
    assert candidates[0]["xy"]["right"] == (-0.22, -0.34)
    assert candidates[0]["yaw"]["right"] == 45.0
    for candidate in candidates:
        assert candidate["xy"]["left"] == (-0.31, 0.26)
        assert candidate["yaw"]["left"] == 315.0
        assert candidate["shared_base_z_m"] == 0.87
        assert candidate["tilt_deg"] == {"left": 0.0, "right": 0.0}
        assert candidate["roll_deg"] == {"left": 0.0, "right": 0.0}


def test_mount_score_prioritizes_fixed_time_tracking_then_joint_margin():
    improved = right_mount_score(
        strict_success=np.ones(183, dtype=bool), collision_free=np.ones(183, dtype=bool),
        selected_joint_margin_rad=np.full(183, 0.20), candidate_counts=np.ones(183),
        mean_pose_error=0.001, displacement_m=0.08, yaw_displacement_deg=15.0,
    )
    incumbent = right_mount_score(
        strict_success=np.r_[np.ones(76, dtype=bool), np.zeros(107, dtype=bool)],
        collision_free=np.ones(183, dtype=bool),
        selected_joint_margin_rad=np.full(183, 0.80), candidate_counts=np.ones(183),
        mean_pose_error=0.0001, displacement_m=0.0, yaw_displacement_deg=0.0,
    )
    assert improved < incumbent


def test_global_score_prioritizes_total_and_longest_failure_before_margin():
    fragmented = np.zeros(20, dtype=bool)
    fragmented[[1, 3, 5, 7]] = True
    contiguous = np.zeros(20, dtype=bool)
    contiguous[1:4] = True
    assert longest_failure_run(fragmented) == 1
    assert longest_failure_run(contiguous) == 3
    fragmented_score = right_mount_score(
        strict_success=~fragmented, collision_free=np.ones(20, bool),
        selected_joint_margin_rad=np.full(20, .1), candidate_counts=np.ones(20),
        mean_pose_error=.01, displacement_m=.1, yaw_displacement_deg=0,
    )
    contiguous_score = right_mount_score(
        strict_success=~contiguous, collision_free=np.ones(20, bool),
        selected_joint_margin_rad=np.full(20, .9), candidate_counts=np.ones(20),
        mean_pose_error=.001, displacement_m=0, yaw_displacement_deg=0,
    )
    assert contiguous_score < fragmented_score


def test_screening_windows_cover_the_whole_timeline_and_preserve_adjacency():
    windows = whole_trajectory_window_indices(
        frame_count=100, window_count=5, window_length=4)
    assert len(windows) == 5
    np.testing.assert_array_equal(windows[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(windows[-1], [96, 97, 98, 99])
    assert all(np.all(np.diff(window) == 1) for window in windows)


def test_exhaustive_grid_includes_boundaries_yaws_and_excludes_left_overlap():
    mounts = exhaustive_tabletop_right_mounts(
        table_half_size_m=(.10, .10), mount_radius_m=.05,
        xy_step_m=.05, yaw_step_deg=90,
        left_xy=(0.0, 0.0), minimum_center_distance_m=.01,
    )
    assert len(mounts) == 32
    assert {item[2] for item in mounts} == {0.0, 90.0, 180.0, 270.0}
    assert all(abs(x) <= .05 and abs(y) <= .05 for x, y, _ in mounts)
    assert all(np.hypot(x, y) >= .01 for x, y, _ in mounts)


def test_visual_flip_gate_rejects_retimed_branch_switch_not_normal_motion():
    q = np.zeros((4, 6))
    q[1, 3] = np.deg2rad(10)
    q[2, 3] = np.deg2rad(179)
    q[2, 4] = np.deg2rad(100)
    q[3] = q[2] + np.deg2rad([1, 1, 1, 2, 2, 2])
    targets = np.zeros((4, 3))
    targets[1:, 0] = [.001, .002, .003]
    mask = visual_wrist_flip_mask(q, targets)
    np.testing.assert_array_equal(mask, [False, False, True, False])


def test_selected_mount_keeps_best_full_episode_baseline_until_global_search_wins():
    assert SELECTED_MOUNT["xy"]["right"] == (
        -0.21721384612283592, -0.34262402935409875)
    assert SELECTED_MOUNT["yaw"]["right"] == 45.0
    assert SELECTED_MOUNT["shared_base_z_m"] == 0.87


def test_dense_validation_cli_overrides_right_mount_and_uses_isolated_output():
    options = parse_run_options([
        "--right-x", "-0.267", "--right-y", "-0.343",
        "--right-yaw", "45", "--output-stem", "finalist_left_5cm",
        "--no-video",
        "--global-retimed-no-flip",
    ])
    assert options.right_xy == (-0.267, -0.343)
    assert options.right_yaw_deg == 45.0
    assert options.output_stem == "finalist_left_5cm"
    assert not options.render_video
    assert options.global_retimed_no_flip


def test_seal_bag_runner_selects_i2rt_yam_and_isolates_its_outputs():
    options = parse_run_options([
        "--robot", "i2rt_yam", "--output-stem", "seal_bag_i2rt_trial",
        "--no-video", "--global-retimed-no-flip",
    ])
    assert options.robot_name == "i2rt_yam"
    output = output_path_for_robot(options.robot_name, options.output_stem)
    assert output.name == "seal_bag_i2rt_trial.mp4"
    assert output.parent.name == "seal_bag_dual_i2rt_yam"


def test_seal_bag_runner_selects_piperx_and_isolates_its_outputs():
    options = parse_run_options([
        "--robot", "piperx", "--output-stem", "trial",
        "--left-x", "-.30", "--left-y", ".30", "--left-yaw", "0",
        "--right-x", "-.20", "--right-y", "-.40", "--right-yaw", "0",
        "--shared-base-z", ".81",
    ])
    assert options.robot_name == "piperx"
    assert options.left_xy == (-.30, .30)
    assert options.left_yaw_deg == 0
    assert options.right_xy == (-.20, -.40)
    assert options.shared_base_z_m == .81
    assert output_path_for_robot("piperx", "trial").parent.name == "seal_bag_dual_piperx"
    assert "PiperX" in mount_selection_method("piperx")
    assert "xArm6" not in mount_selection_method("piperx")
