import pytest

from factory_bimanual.mount_comparison_visuals import STUDY_PANEL_ORDER
from scripts.build_piperx_multitask_mount_report import (
    build_report_claims,
    validate_report_manifest,
)


def _manifest(count=108):
    shards = []
    for index in range(count):
        trajectory = f"task-{index // 4:02d}"
        mode = STUDY_PANEL_ORDER[index % 4]
        shards.append({
            "trajectory": trajectory,
            "family": f"family-{index // 12}",
            "mode": mode,
            "metrics": {
                "both_accept_coverage": (
                    0.9 if mode == "upright_table" else 0.5),
                "collision_frames": 0,
                "edge_collision_frames": 0,
                "topology_invalid_frames": 0,
            },
        })
    return {
        "schema": "piperx-multitask-fixed-time-bundle-v1",
        "status": "complete",
        "trajectory_count": 27,
        "family_count": 12,
        "mode_count": 4,
        "retiming_applied": False,
        "shards": shards,
    }


def test_report_rejects_missing_trajectory_or_mount():
    validate_report_manifest(_manifest())

    with pytest.raises(ValueError, match="complete 27 x 4 matrix"):
        validate_report_manifest(_manifest(107))


def test_report_claims_count_one_winner_per_trajectory():
    claims = build_report_claims(_manifest())

    assert claims["total_trajectories"] == 27
    assert claims["total_experiments"] == 108
    assert claims["winner_counts"]["upright_table"] == 27
    assert sum(claims["winner_counts"].values()) == 27
