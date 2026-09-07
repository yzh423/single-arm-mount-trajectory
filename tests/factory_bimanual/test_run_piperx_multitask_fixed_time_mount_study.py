from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scripts.run_piperx_multitask_fixed_time_mount_study as study

from factory_bimanual.multitask_fixed_time_study import (
    STUDY_MODES,
    TrajectorySpec,
)
from factory_bimanual.task_family import TaskFamily
from factory_bimanual.tool_frame_calibration import apply_fixed_tool_translation
from factory_bimanual.piperx_recommended import load_recommended_config
from scripts.run_piperx_multitask_fixed_time_mount_study import (
    _best_observed_mount,
    _best_current_funnel_mount,
    _candidate_fingerprint,
    _normalize_baseline_mount_payload,
    STUDY_ORIENTATION_TOLERANCE_RAD,
    STUDY_POSITION_TOLERANCE_M,
    family_representative_spec,
    _exploration_frontier,
    plan_jobs,
    prepare_family_follow_targets,
    rank_mount_result,
    study_status_path,
)


def _spec(index):
    return TrajectorySpec(
        family=TaskFamily("8-11", f"Task{index}"),
        take=f"{index:06d}",
        path=Path(f"task-{index}.csv"),
        source_sha256=f"{index:064x}",
        row_count=100,
    )


def test_job_matrix_contains_every_trajectory_mount_pair():
    specs = tuple(_spec(index) for index in range(27))
    jobs = plan_jobs(specs, STUDY_MODES)

    assert len(jobs) == 108
    assert len({(job.spec.key, job.mode) for job in jobs}) == 108
    assert {job.mode for job in jobs} == set(STUDY_MODES)


def test_mode_workers_write_isolated_status_files(tmp_path):
    assert study_status_path(tmp_path, None).name == "study_status.json"
    assert (study_status_path(tmp_path, "upright_table").name
            == "study_status_upright_table.json")
    assert (study_status_path(tmp_path, "horizontal_wall")
            != study_status_path(tmp_path, "inverted"))


def test_mount_search_uses_requested_strict_pose_gate():
    assert STUDY_POSITION_TOLERANCE_M == pytest.approx(.001)
    assert np.rad2deg(STUDY_ORIENTATION_TOLERANCE_RAD) == pytest.approx(.5)


def test_baseline_mount_preserves_mode_and_has_physical_table_adapter():
    payload = {
        "mode": "upright_table",
        "shared_base_z_m": .732,
        "base_z_m": {"left": .732, "right": .732},
        "selection_method": "configured",
    }

    normalized = _normalize_baseline_mount_payload(payload)

    assert normalized["mode"] == "upright_table"
    assert normalized["shared_base_z_m"] == pytest.approx(.751)
    assert normalized["base_z_m"] == {"left": .751, "right": .751}
    assert "physical support clamp" in normalized["selection_method"]


def test_horizontal_baseline_mode_is_not_rewritten_as_upright():
    payload = {
        "mode": "horizontal_forward",
        "shared_base_z_m": 1.2,
        "base_z_m": {"left": 1.2, "right": 1.2},
        "selection_method": "configured",
    }

    normalized = _normalize_baseline_mount_payload(payload)

    assert normalized["mode"] == "horizontal_forward"
    assert normalized["shared_base_z_m"] == pytest.approx(1.2)


def test_positive_low_profile_recommended_adapter_is_not_raised():
    payload = {
        "mode": "upright_table",
        "shared_base_z_m": .75709,
        "base_z_m": {"left": .75709, "right": .75709},
        "selection_method": "configured",
    }

    normalized = _normalize_baseline_mount_payload(payload)

    assert normalized["shared_base_z_m"] == pytest.approx(.75709)
    assert normalized["selection_method"] == "configured"


def test_family_target_preparation_applies_configured_tcp_translation():
    from factory_bimanual.multitask_fixed_time_study import (
        discover_dual_hand_trajectories,
    )
    from scripts.run_piperx_multitask_fixed_time_mount_study import (
        ROOT,
        _load_registered_spec,
    )

    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    spec = next(item for item in specs
                if item.key == "8-11/Fold_Box/161044")
    raw, _registration = _load_registered_spec(spec)
    prepared, _mapped, _audit = prepare_family_follow_targets(
        spec, raw, apply_conditioning=False)
    mount_spec = load_recommended_config().mounts[spec.family.key]
    expected_left = apply_fixed_tool_translation(
        raw.left_position_m, raw.left_quaternion_wxyz,
        mount_spec.left_tool_translation_m)

    np.testing.assert_allclose(prepared.left_position_m, expected_left)
    assert not np.allclose(prepared.left_position_m, raw.left_position_m)
    assert hasattr(prepared, "right_wrist_adaptation_angle_deg")


def test_raw_target_contract_disables_time_varying_wrist_adaptation():
    from factory_bimanual.multitask_fixed_time_study import (
        discover_dual_hand_trajectories,
    )
    from scripts.run_piperx_multitask_fixed_time_mount_study import (
        ROOT,
        _load_registered_spec,
    )

    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    spec = next(item for item in specs
                if item.key == "8-11/Fold_Box/161044")
    raw, _registration = _load_registered_spec(
        spec, timing_mode="controller_updates")

    prepared, _mapped, audit = prepare_family_follow_targets(
        spec, raw, apply_conditioning=False,
        apply_wrist_adaptation=False)

    assert prepared.timing_source == "paired_controller_receive"
    assert len(prepared.time_s) < spec.row_count
    np.testing.assert_allclose(
        prepared.right_wrist_adaptation_angle_deg, 0.0)
    assert audit.maximum_position_deviation_m == 0.0
    assert audit.maximum_orientation_deviation_rad == 0.0


def test_family_mount_search_uses_configured_representative_take():
    base = _spec(0)
    specs = tuple(replace(base, take=f"{index:06d}") for index in range(3))

    representative = family_representative_spec(
        specs, specs[2], representative_take="000001")

    assert representative == specs[1]


def test_family_representative_must_exist_in_discovered_dual_hand_data():
    base = _spec(0)
    specs = tuple(replace(base, take=f"{index:06d}") for index in range(3))

    try:
        family_representative_spec(
            specs, specs[0], representative_take="999999")
    except ValueError as error:
        assert "configured representative take" in str(error)
    else:
        raise AssertionError("missing representative take was accepted")


def test_collision_free_full_coverage_outranks_unsafe_and_partial_mounts():
    safe_full = {
        "both_accept_coverage": 1.0, "collision_frames": 0,
        "edge_collision_frames": 0, "topology_invalid_frames": 0,
        "longest_hold_frames": 0, "maximum_normalized_error": 1.0,
    }
    colliding_full = safe_full | {"collision_frames": 1}
    partial = safe_full | {"both_accept_coverage": 0.99}

    assert rank_mount_result(safe_full) < rank_mount_result(colliding_full)
    assert rank_mount_result(colliding_full) < rank_mount_result(partial)


def test_mount_selection_prefers_more_safe_pair_coverage_over_zero_coverage():
    partially_followable = {
        "mount": {"name": "partial"},
        "continuous_pair_coverage": 0.9,
        "pair_collision_frames": 1,
        "pair_edge_collision_frames": 0,
    }
    sparse_safe = {
        "mount": {"name": "safe"},
        "continuous_pair_coverage": 0.2,
        "pair_collision_frames": 0,
        "pair_edge_collision_frames": 0,
    }

    mount, _result = _best_observed_mount([
        partially_followable, sparse_safe])

    assert mount == partially_followable["mount"]


def test_exploration_frontier_keeps_coverage_and_safety_elites():
    high_coverage = {
        "mount": {"name": "coverage"},
        "continuous_pair_coverage": .8,
        "pair_collision_frames": 2,
        "pair_edge_collision_frames": 0,
    }
    safe = {
        "mount": {"name": "safe"},
        "continuous_pair_coverage": 0.0,
        "pair_collision_frames": 0,
        "pair_edge_collision_frames": 0,
    }

    selected = _exploration_frontier([safe, high_coverage], maximum=2)

    assert {row["mount"]["name"] for row in selected} == {
        "coverage", "safe"}


def test_local_search_includes_symmetric_collision_avoidance_candidates():
    assert hasattr(study, "_local_refinement_mounts")
    mount = {
        "xy": {"left": [-.3, .1], "right": [.3, -.1]},
        "yaw": {"left": -30., "right": 150.},
        "shared_base_z_m": .81,
    }
    original_distance = np.linalg.norm(np.subtract(
        mount["xy"]["left"], mount["xy"]["right"]))

    candidates = study._local_refinement_mounts(
        mount, mode="upright_table")

    assert max(np.linalg.norm(np.subtract(
        candidate["xy"]["left"], candidate["xy"]["right"]))
        for candidate in candidates) > original_distance


def test_local_budget_gives_each_frontier_seed_an_outward_candidate():
    assert hasattr(study, "_budgeted_local_refinement_mounts")
    mounts = [
        {"xy": {"left": [-.3, .1], "right": [.3, -.1]},
         "yaw": {"left": -30., "right": 150.},
         "shared_base_z_m": .81},
        {"xy": {"left": [-.35, .35], "right": [.35, .05]},
         "yaw": {"left": -20., "right": 160.},
         "shared_base_z_m": .81},
    ]

    selected = study._budgeted_local_refinement_mounts(
        [{"mount": mount} for mount in mounts],
        mode="upright_table", maximum=6)

    assert len(selected) == 6
    for mount in mounts:
        original = np.linalg.norm(np.subtract(
            mount["xy"]["left"], mount["xy"]["right"]))
        midpoint = np.mean([
            mount["xy"]["left"], mount["xy"]["right"]], axis=0)
        assert any(np.linalg.norm(np.subtract(
            candidate["xy"]["left"], candidate["xy"]["right"]))
            > original + 1e-12 and np.allclose(np.mean([
                candidate["xy"]["left"], candidate["xy"]["right"]],
                axis=0), midpoint)
            for candidate in selected)


def test_mount_search_schema_is_per_trajectory_not_family_shared():
    from scripts.run_piperx_multitask_fixed_time_mount_study import (
        STUDY_SEARCH_CONFIG,
    )

    assert "per-trajectory" in STUDY_SEARCH_CONFIG.schema


def test_candidate_cache_changes_with_tool_frame_contract():
    spec = _spec(0)
    common = (spec, "upright_table", "coarse", {"xy": [0, 0]}, {})

    first = _candidate_fingerprint(
        *common, target_contract={"left_tool": [1, 0, 0, 0]})
    second = _candidate_fingerprint(
        *common, target_contract={"left_tool": [0, 1, 0, 0]})

    assert first != second


def test_terminal_checkpoint_requires_exact_job_fingerprint():
    spec = _spec(0)
    target_contract = {"left_tool": [1, 0, 0, 0]}
    expected = study._job_fingerprint(
        spec, "upright_table", study.STUDY_SEARCH_CONFIG, target_contract)
    state = {
        "status": "complete",
        "selected_mount": {"mode": "upright_table"},
        "source_sha256": spec.source_sha256,
        "job_fingerprint": expected,
    }

    assert study._terminal_checkpoint_reusable(state, expected)
    assert not study._terminal_checkpoint_reusable(
        state | {"job_fingerprint": "stale"}, expected)
    assert not study._terminal_checkpoint_reusable(
        state | {"selected_mount": None}, expected)


def test_search_job_fingerprint_changes_with_tool_contract_and_budget():
    spec = _spec(0)
    first = study._job_fingerprint(
        spec, "upright_table", study.STUDY_SEARCH_CONFIG,
        {"left_tool": [1, 0, 0, 0]})
    second = study._job_fingerprint(
        spec, "upright_table", study.STUDY_SEARCH_CONFIG,
        {"left_tool": [0, 1, 0, 0]})
    changed_budget = replace(
        study.STUDY_SEARCH_CONFIG,
        coarse_budget=study.STUDY_SEARCH_CONFIG.coarse_budget + 1)
    third = study._job_fingerprint(
        spec, "upright_table", changed_budget,
        {"left_tool": [1, 0, 0, 0]})

    assert len({first, second, third}) == 3


def test_funnel_fallback_uses_only_explicit_current_stage_rows():
    old_historical = {
        "mount": {"name": "old-1.5-degree"},
        "continuous_pair_coverage": 1.0,
        "pair_collision_frames": 0,
        "pair_edge_collision_frames": 0,
    }
    current = {
        "mount": {"name": "current-0.5-degree"},
        "continuous_pair_coverage": 0.2,
        "pair_collision_frames": 0,
        "pair_edge_collision_frames": 0,
    }
    state_records = [old_historical, current]

    mount, _result = _best_current_funnel_mount([state_records[-1]])

    assert mount == current["mount"]


def test_ranking_uses_hold_and_error_only_after_safety_and_coverage():
    base = {
        "both_accept_coverage": 0.95, "collision_frames": 0,
        "edge_collision_frames": 0, "topology_invalid_frames": 0,
        "longest_hold_frames": 10, "maximum_normalized_error": 2.0,
    }
    shorter_hold = base | {"longest_hold_frames": 2}
    lower_error = shorter_hold | {"maximum_normalized_error": 1.0}

    assert rank_mount_result(lower_error) < rank_mount_result(shorter_hold)
    assert rank_mount_result(shorter_hold) < rank_mount_result(base)
