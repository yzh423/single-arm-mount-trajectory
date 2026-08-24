from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.multitask_fixed_time_study import (
    STUDY_MODES,
    TrajectorySpec,
)
from factory_bimanual.task_family import TaskFamily
from scripts.run_piperx_multitask_fixed_time_mount_study import (
    _best_observed_mount,
    _normalize_baseline_mount_payload,
    STUDY_ORIENTATION_TOLERANCE_RAD,
    STUDY_POSITION_TOLERANCE_M,
    family_representative_spec,
    plan_jobs,
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
    assert normalized["shared_base_z_m"] == pytest.approx(.83)
    assert normalized["base_z_m"] == {"left": .83, "right": .83}
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


def test_infeasible_fallback_prefers_any_collision_free_observation():
    colliding = {
        "mount": {"name": "colliding"},
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

    mount, _result = _best_observed_mount([colliding, sparse_safe])

    assert mount == sparse_safe["mount"]


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
