from pathlib import Path

from factory_bimanual.factory_task_catalog import TaskRepresentative
from factory_bimanual.per_task_mount_search import PerTaskSearchConfig
from scripts.run_piperx_factory_per_task_mount_search import (
    evaluate_stage_resumable, plan_mode_jobs,
)
from scripts.search_fold_box_piperx_paired_mount import bounded_full_planning_indices


def _representative(name):
    return TaskRepresentative(name, Path(f"/{name}.csv"), f"{name}.csv", 10,
                              name * 8)


def test_plan_schedules_each_representative_in_each_mode():
    representatives = tuple(_representative(f"task{i}") for i in range(11))
    jobs = plan_mode_jobs(representatives, PerTaskSearchConfig())
    assert len(jobs) == 33
    assert len({(job.task_name, job.mode) for job in jobs}) == 33


def test_stage_resume_skips_matching_fingerprint(tmp_path):
    calls = []
    mounts = [
        {"xy": {"left": [-.4, 0], "right": [.4, 0]},
         "yaw": {"left": 0, "right": 180}, "shared_base_z_m": .81},
    ]
    state = {"records": []}
    kwargs = dict(state=state, checkpoint=tmp_path / "state.json",
                  source_sha256="a" * 64, mode="upright_table",
                  stage="coarse", mounts=mounts,
                  config=PerTaskSearchConfig(), settings={"uniform_count": 8})
    first = evaluate_stage_resumable(
        **kwargs, evaluator=lambda mount, serial, settings: (
            calls.append(serial) or {"mount": mount, "coverage": .5}))
    second = evaluate_stage_resumable(
        **kwargs, evaluator=lambda mount, serial, settings: (
            calls.append(serial) or {"mount": mount, "coverage": .5}))
    assert first == second
    assert len(calls) == 1
    assert (tmp_path / "state.json").exists()


def test_full_planning_is_bounded_and_keeps_source_endpoints():
    indices = bounded_full_planning_indices(11030, maximum=512)
    assert len(indices) <= 512
    assert indices[0] == 0 and indices[-1] == 11029
