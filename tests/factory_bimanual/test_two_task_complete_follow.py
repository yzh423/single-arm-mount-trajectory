import json

from scripts.optimize_piperx_two_task_follow import (
    MountCandidate,
    append_experiment_record,
    local_mount_candidates,
    pending_candidates,
)


def _candidate():
    return MountCandidate(
        name="incumbent",
        mode="upright_table",
        left_xyz_m=(-0.35, 0.25, 0.81),
        right_xyz_m=(-0.30, -0.45, 0.81),
        left_yaw_deg=15.0,
        right_yaw_deg=15.0,
        origin="historical_v4",
    )


def test_local_mount_candidates_are_deterministic_and_share_height():
    first = local_mount_candidates(
        _candidate(), xy_step_m=0.025, z_step_m=0.02,
        yaw_step_deg=7.5, round_index=1)
    second = local_mount_candidates(
        _candidate(), xy_step_m=0.025, z_step_m=0.02,
        yaw_step_deg=7.5, round_index=1)

    assert [item.key for item in first] == [item.key for item in second]
    assert len(first) == len(set(item.key for item in first))
    assert all(item.left_xyz_m[2] == item.right_xyz_m[2] for item in first)
    assert any(item.right_xyz_m[0] == -0.275 for item in first)
    assert any(item.right_yaw_deg == 22.5 for item in first)


def test_completed_candidate_is_not_repeated_after_resume(tmp_path):
    log_path = tmp_path / "experiment.json"
    candidates = local_mount_candidates(
        _candidate(), xy_step_m=0.025, z_step_m=0.02,
        yaw_step_deg=7.5, round_index=1)
    append_experiment_record(log_path, {
        "candidate_key": candidates[0].key,
        "status": "probe_complete",
        "strict_hits": 3,
        "strict_total": 4,
    })

    remaining = pending_candidates(candidates, log_path, resume=True)

    assert candidates[0].key not in {item.key for item in remaining}
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "piperx-two-task-search-v1"
    assert payload["records"][0]["strict_hits"] == 3
