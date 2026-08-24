from pathlib import Path

from factory_bimanual.multitask_fixed_time_study import (
    STUDY_MODES,
    TrajectorySpec,
)
from factory_bimanual.task_family import TaskFamily
from scripts.run_piperx_multitask_fixed_time_mount_study import (
    plan_jobs,
    rank_mount_result,
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
