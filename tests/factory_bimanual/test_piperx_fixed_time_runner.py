import numpy as np
import pytest
import inspect
from types import SimpleNamespace
from pathlib import Path

import scripts.render_factory_dual_piperx_fixed_time as runner
from scripts.render_factory_dual_xarm6_se3_follow import (
    solve_collision_safe_bimanual_method,
    solve_multibranch_single_arm_method,
)
from scripts.render_factory_dual_piperx_fixed_time import (
    fixed_time_timing_audit,
    mount_separation_for_run,
    load_locked_piperx_calibration,
    render_saved_run,
    comparison_planning_indices,
    interpolate_joint_path,
    solve_fixed_time_motion,
    task_spec,
    validate_saved_trajectory,
)
from factory_bimanual.tool_frame_calibration import (
    CalibrationArtifact,
    source_file_fingerprint,
)


def test_comparison_planning_indices_keep_source_endpoints():
    np.testing.assert_array_equal(
        comparison_planning_indices(10, stride=4), [0, 4, 8, 9])


def test_joint_interpolation_uses_shortest_periodic_delta():
    source_t = np.asarray([0., 1.])
    q = np.deg2rad(np.asarray([[179., 0.], [-179., 2.]]))
    full = interpolate_joint_path(
        source_t, q, np.asarray([0., .5, 1.]),
        periodic=np.asarray([True, False]))
    assert np.rad2deg(full[1, 0]) == pytest.approx(180.)
    assert np.rad2deg(full[1, 1]) == pytest.approx(1.)


def test_fixed_time_runner_has_separate_task_specific_outputs():
    fold = task_spec("fold_box")
    seal = task_spec("seal_bag")

    assert fold.output_stem == "fold_box_piperx_xarm6_style_fixed_time_front_720p"
    assert seal.output_stem == "seal_bag_piperx_xarm6_style_fixed_time_front_720p"
    assert fold.report_directory.name == "fold_box_dual_piperx"
    assert seal.report_directory.name == "seal_bag_dual_piperx"


def test_fixed_time_runner_rejects_unknown_task():
    with pytest.raises(ValueError, match="fold_box or seal_bag"):
        task_spec("unknown")


def test_fixed_time_timing_audit_requires_exact_source_schedule():
    source = np.asarray([5.0, 5.1, 5.35])
    audit = fixed_time_timing_audit(source, np.asarray([0.0, 0.1, 0.35]))

    assert audit == {
        "timing_mode": "fixed_source_time",
        "source_duration_s": pytest.approx(0.35),
        "execution_duration_s": pytest.approx(0.35),
        "retimed_frame_count": 0,
        "added_duration_s": 0.0,
    }

    with pytest.raises(ValueError, match="must equal source timestamps"):
        fixed_time_timing_audit(source, np.asarray([0.0, 0.2, 0.5]))


def test_recommended_search_allows_original_4387mm_mount_without_override():
    mount = {
        "xy": {"left": [-0.2312103678324848, -0.0130977167988798],
               "right": [0.0412103678324847, -0.3569022832011202]},
        "yaw": {"left": 15.0, "right": 45.0},
        "shared_base_z_m": 0.81,
    }

    assert mount_separation_for_run(
        mount, allow_legacy_mount=False) == pytest.approx(0.438650928528911)

    mount["xy"]["right"] = [mount["xy"]["left"][0],
                              mount["xy"]["left"][1] - .50]
    assert mount_separation_for_run(mount) == pytest.approx(.50)


def test_orientation_mount_requires_forty_centimetres_without_override():
    mount = {
        "mode": "upright_table",
        "xy": {"left": [0.0, 0.0], "right": [.39, 0.0]},
        "yaw": {"left": 0.0, "right": 180.0},
        "shared_base_z_m": .96,
    }
    with pytest.raises(ValueError, match="below 0.400000"):
        mount_separation_for_run(mount)
    mount["xy"]["right"] = [.40, 0.0]
    assert mount_separation_for_run(mount) == pytest.approx(.40)


def test_fixed_time_motion_uses_xarm6_style_without_retiming(monkeypatch):
    sentinel = object()
    calls = []

    def fake_solver(model, task, mapped_quat, **kwargs):
        calls.append((model, task, mapped_quat, kwargs))
        return sentinel

    monkeypatch.setattr(
        runner, "solve_multibranch_single_arm_method", fake_solver)

    assert solve_fixed_time_motion(
        "model", "task", "mapped", solver_method="xarm6-style") is sentinel
    assert calls == [("model", "task", "mapped", {
        "horizon": 12,
        "beam_width": 8,
        "candidate_iterations": 60,
        "velocity_limit_rad_s": 3.14,
        "global_retimed_no_flip": False,
        "robot_name": "piperx",
    })]


def test_fixed_time_motion_rejects_unknown_solver():
    with pytest.raises(ValueError, match="xarm6-style or paired"):
        solve_fixed_time_motion(
            "model", "task", "mapped", solver_method="unknown")


def test_paired_fixed_time_motion_enables_bounded_orientation_adaptation(monkeypatch):
    sentinel = object()
    calls = []

    def fake_solver(model, task, mapped_quat, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        runner, "solve_collision_safe_bimanual_method", fake_solver)

    assert solve_fixed_time_motion(
        "model", "task", "mapped", solver_method="paired") is sentinel
    assert calls[0]["strict_pose_only"] is True
    assert calls[0]["reference_aware_enabled"] is True
    assert calls[0]["orientation_adaptation_enabled"] is True


def test_saved_trajectory_must_match_source_time_and_not_be_retimed():
    source = np.asarray([5.0, 5.1, 5.3])
    payload = {
        "source_time_s": source,
        "time_s": np.asarray([0.0, 0.1, 0.3]),
        "time_retimed": np.zeros(3, dtype=bool),
        "qpos": np.zeros((3, 12)),
        "collision": np.zeros(3, dtype=bool),
        "failure_reason": np.asarray(["ok", "ok", "ok"]),
        "synchronous_success": np.ones(3, dtype=bool),
    }

    validate_saved_trajectory(source, payload)
    payload["time_retimed"][1] = True
    with pytest.raises(ValueError, match="retimed"):
        validate_saved_trajectory(source, payload)
    payload["time_retimed"][1] = False
    payload["time_s"][2] = .4
    with pytest.raises(ValueError, match="source timeline"):
        validate_saved_trajectory(source, payload)


def test_xarm6_style_solver_exposes_screening_budget_without_changing_defaults():
    parameters = inspect.signature(
        solve_multibranch_single_arm_method).parameters
    assert parameters["global_seed_count"].default == 12
    assert parameters["maximum_candidates"].default == 8
    assert parameters["constrained_fallback_enabled"].default is True


def test_paired_solver_exposes_candidate_budget_without_changing_defaults():
    parameters = inspect.signature(
        solve_collision_safe_bimanual_method).parameters
    assert parameters["candidate_iterations"].default == 60
    assert parameters["global_seed_count"].default == 2
    assert parameters["maximum_candidates_per_tier"].default == 8
    assert parameters["constrained_fallback_enabled"].default is True


def test_saved_run_renders_cache_without_resolving(monkeypatch, tmp_path):
    source = np.asarray([5.0, 5.1, 5.3])
    cache = tmp_path / "candidate.trajectory.npz"
    np.savez_compressed(
        cache,
        source_time_s=source,
        time_s=np.asarray([0.0, 0.1, 0.3]),
        time_retimed=np.zeros(3, dtype=bool),
        qpos=np.zeros((3, 12)),
        collision=np.asarray([False, True, False]),
        failure_reason=np.asarray(["ok", "COLLISION", "ok"]),
        synchronous_success=np.asarray([True, False, True]),
    )
    cache.with_name("candidate.summary.json").write_text(
        '{"task":"fold_box","decode":null}', encoding="utf-8")
    task = SimpleNamespace(
        time_s=source,
        left_position_m=np.zeros((3, 3)),
        right_position_m=np.ones((3, 3)),
    )
    monkeypatch.setattr(runner, "_registered_task", lambda spec: (task, np.zeros(3)))
    monkeypatch.setattr(runner, "build_same_model_scene", lambda *a, **k: None)
    monkeypatch.setattr(
        runner, "render_mujoco_mp4",
        lambda *a, **k: SimpleNamespace(check=SimpleNamespace(frame_count=19)))
    monkeypatch.setattr(
        runner, "solve_fixed_time_motion",
        lambda *a, **k: pytest.fail("cached render must not rerun IK"))
    mount = {
        "xy": {"left": [-0.45, 0.3], "right": [-0.35, -0.5]},
        "yaw": {"left": -15.0, "right": 15.0},
        "shared_base_z_m": 0.81,
    }
    output = tmp_path / "winner.mp4"

    summary = render_saved_run("fold_box", mount, cache, output)

    assert summary["decode"] == {"frame_count": 19}
    assert output.with_suffix(".trajectory.npz").exists()
    assert output.with_suffix(".summary.json").exists()


def test_locked_calibration_validates_both_task_sources(monkeypatch, tmp_path):
    sources = {}
    for task in ("fold_box", "seal_bag"):
        path = tmp_path / f"{task}.csv"
        path.write_text(f"source-{task}", encoding="utf-8")
        sources[task] = path
    artifact = CalibrationArtifact(
        version=1, robot="piperx", tasks=("fold_box", "seal_bag"),
        source_fingerprints={
            task: source_file_fingerprint(path)
            for task, path in sources.items()},
        left_offset_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        right_offset_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        left_axis_rotation_index=0, right_axis_rotation_index=0,
        left_local_xyz_deg=(0.0, 0.0, 0.0),
        right_local_xyz_deg=(0.0, 0.0, 0.0), metrics={},
    )
    path = tmp_path / "calibration.json"
    artifact.write(path)
    monkeypatch.setattr(
        runner, "task_spec",
        lambda task: SimpleNamespace(csv=sources[task]))

    assert load_locked_piperx_calibration(path) == artifact
    sources["fold_box"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_locked_piperx_calibration(path)
