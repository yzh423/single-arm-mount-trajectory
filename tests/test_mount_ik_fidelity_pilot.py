import json
from pathlib import Path

import numpy as np

from scripts.run_mount_ik_fidelity_pilot import (
    DEFAULT_ROBOTS,
    build_parser,
    coarse_model_contract,
    coarse_fingerprint,
    coarse_algorithm_fingerprint,
    load_completed_robot_result,
    full_episode_optimism_metrics,
    official_model_contract,
    pilot_fingerprint,
    load_stage_records,
    stage_input_fingerprint,
    stage_counts,
)
from scripts.run_thirteen_arm_dense_search import (
    build_parser as build_coarse_parser,
    select_proxy_shortlist,
)
from scripts.build_mount_ik_fidelity_pilot_report import (
    build_report_payload,
    validate_pilot_completeness,
    write_report,
)


def test_pilot_defaults_match_approved_funnel() -> None:
    args = build_parser().parse_args([])

    assert args.robots == list(DEFAULT_ROBOTS)
    assert stage_counts(args) == {
        "coarse": 4096,
        "proxy": 256,
        "window": 128,
        "full": 24,
        "final": 8,
    }


def test_smoke_profile_is_small_but_preserves_all_stages() -> None:
    args = build_parser().parse_args(["--smoke"])

    assert stage_counts(args) == {
        "coarse": 32,
        "proxy": 12,
        "window": 6,
        "full": 2,
        "final": 2,
    }


def test_pilot_fingerprint_changes_with_tcp_and_episode_content() -> None:
    base = pilot_fingerprint(
        robot="xarm6", episode_sha256="episode-a", model_sha256="model-a",
        tcp_translation=(0.0, 0.0, 0.0), counts={"coarse": 32})
    changed_tcp = pilot_fingerprint(
        robot="xarm6", episode_sha256="episode-a", model_sha256="model-a",
        tcp_translation=(0.0, 0.0, 0.13), counts={"coarse": 32})
    changed_episode = pilot_fingerprint(
        robot="xarm6", episode_sha256="episode-b", model_sha256="model-a",
        tcp_translation=(0.0, 0.0, 0.0), counts={"coarse": 32})

    assert len(base) == 64
    assert len({base, changed_tcp, changed_episode}) == 3


def test_gpu_coarse_cli_can_emit_diverse_candidate_shortlist() -> None:
    args = build_coarse_parser().parse_args(["--shortlist-size", "17"])

    assert args.shortlist_size == 17


def test_gpu_proxy_shortlist_persists_candidate_ids_and_distant_basins() -> None:
    mounts = np.asarray(((0.0,), (.01,), (.95,), (1.0,)))
    ranks = [(4.0,), (3.0,), (2.0,), (1.0,)]

    rows = select_proxy_shortlist(
        mounts, ranks, lower=np.zeros(1), upper=np.ones(1), size=3,
        record=lambda index: {"proxy": float(ranks[index][0])})

    assert [row["candidate_id"] for row in rows] == [0, 3, 2]
    assert all("mount" in row and "proxy" in row for row in rows)


def test_gpu_proxy_shortlist_diversifies_over_the_entire_coarse_population() -> None:
    mounts = np.asarray([[0.01 * index] for index in range(9)] + [[1.0]])
    ranks = [(float(10 - index),) for index in range(10)]

    rows = select_proxy_shortlist(
        mounts, ranks, lower=np.zeros(1), upper=np.ones(1), size=2,
        record=lambda index: {"proxy": float(ranks[index][0])})

    assert [row["candidate_id"] for row in rows] == [0, 9]


def _pilot_row(robot: str, cache: Path) -> dict:
    return {
        "robot": robot,
        "cross_fidelity": {
            "spearman_rank_correlation": 0.5,
            "top_k_recall": 0.75,
            "optimistic_episode_pass_rate": 0.25,
        },
        "stage_counts": {"proxy": 12, "window": 6, "full_and_local": 2, "final": 2},
        "winner": {
            "episode_success": False,
            "frame_coverage": 0.8,
            "mount": [0.1, -0.2, 0.3, 0.0, 45.0, 0.0],
            "position_error_m_p95": 0.02,
            "orientation_error_rad_p95": 0.1,
        },
        "native_winner": {
            "episode_success": False,
            "frame_coverage": 0.9,
            "mount": [0.1, -0.2, 0.3, 0.0, 45.0, 0.0],
            "position_error_m_p95": 0.015,
            "orientation_error_rad_p95": 0.08,
        },
        "cache": str(cache),
        "elapsed_s": 12.0,
    }


def test_report_payload_contains_failures_mounts_and_video_paths(tmp_path: Path) -> None:
    rows = []
    videos = {}
    for robot in ("xarm6", "openarm"):
        cache = tmp_path / f"{robot}.npz"
        np.savez_compressed(
            cache,
            success=np.asarray([True, False]),
            failure_reason=np.asarray(["none", "position"]),
            table_collision=np.asarray([False, False]),
            self_collision=np.asarray([False, False]),
            edge_collision=np.asarray([False, False]),
            joint_discontinuity=np.asarray([False, False]),
            recovery_mode=np.asarray(["none", "hold_no_pose_candidate"]),
        )
        video = tmp_path / f"{robot}.mp4"
        video.write_bytes(b"nonempty-video-placeholder")
        rows.append(_pilot_row(robot, cache))
        videos[robot] = video

    payload = build_report_payload(rows, videos=videos)

    assert payload["complete"] is True
    assert payload["robots"][0]["mount"][4] == 45.0
    assert payload["robots"][0]["failure_counts"]["position"] == 1
    assert payload["robots"][0]["failure_counts"]["solver:hold_no_pose_candidate"] == 1
    assert payload["robots"][0]["native_frame_coverage"] == 0.9
    assert any("xarm6" in conclusion for conclusion in payload["conclusions"])
    assert payload["robots"][1]["video"].endswith("openarm.mp4")


def test_completeness_rejects_missing_robot_or_video(tmp_path: Path) -> None:
    cache = tmp_path / "xarm6.npz"
    np.savez_compressed(
        cache,
        success=np.asarray([False]),
        failure_reason=np.asarray(["position"]),
        table_collision=np.asarray([False]), self_collision=np.asarray([False]),
        edge_collision=np.asarray([False]), joint_discontinuity=np.asarray([False]),
        recovery_mode=np.asarray(["hold_no_pose_candidate"]),
    )

    errors = validate_pilot_completeness(
        [_pilot_row("xarm6", cache)], videos={},
        expected_robots=("xarm6", "openarm"), verify_video_decode=False)

    assert any("openarm" in error for error in errors)
    assert any("video" in error for error in errors)


def test_completed_robot_result_resumes_only_matching_fingerprint(tmp_path: Path) -> None:
    cache = tmp_path / "winner.npz"
    np.savez_compressed(cache, q=np.zeros((2, 1)))
    result_path = tmp_path / "robots" / "xarm6.json"
    result_path.parent.mkdir()
    result_path.write_text(json.dumps({
        "robot": "xarm6", "fingerprint": "expected", "cache": str(cache),
        "winner": {"frame_coverage": 1.0},
    }), encoding="utf-8")

    assert load_completed_robot_result(tmp_path, "xarm6", "expected") is not None
    assert load_completed_robot_result(tmp_path, "xarm6", "stale") is None


def test_coarse_fingerprint_invalidates_when_official_tcp_changes() -> None:
    common = dict(
        robots=("openarm",), episode_sha256="episode", candidate_count=4096)
    old = coarse_fingerprint(
        **common, model_contracts={"openarm": {"sha256": "model", "tcp": [0, 0, .13]}})
    corrected = coarse_fingerprint(
        **common, model_contracts={"openarm": {"sha256": "model", "tcp": [0, 0, 0]}})

    assert old != corrected
    assert len(coarse_algorithm_fingerprint()) == 64


def test_official_contract_fingerprints_meshes_tcp_and_collision_policy() -> None:
    contract = official_model_contract("openarm")

    assert len(contract["contract_sha256"]) == 64
    assert contract["tcp_authority"] == "official_xacro_default_83p5mm"
    assert contract["asset_sha256"]
    assert "missing" not in contract["asset_sha256"].values()
    assert any("collision_proxy_profiles" in key
               for key in contract["collision_sha256"])
    assert any("strict_mujoco_ik" in key
               for key in contract["collision_sha256"])
    coarse = coarse_model_contract("openarm")
    assert not any("strict_mujoco_ik" in key
                   for key in coarse["collision_sha256"])
    assert coarse["contract_sha256"] != contract["contract_sha256"]


def test_stage_checkpoint_rejects_changed_candidate_inputs(tmp_path: Path) -> None:
    path = tmp_path / "window.partial.json"
    signature = stage_input_fingerprint(
        robot_fingerprint="robot", stage="window",
        candidates=[{"candidate_id": 1, "mount": [0, 0, .2, 0]}])
    path.write_text(json.dumps({
        "stage_input_fingerprint": signature,
        "records": [{"candidate_id": 1, "frame_coverage": .5}],
    }), encoding="utf-8")

    assert len(load_stage_records(path, signature)) == 1
    changed = stage_input_fingerprint(
        robot_fingerprint="robot", stage="window",
        candidates=[{"candidate_id": 1, "mount": [0, 0, .3, 0]}])
    assert load_stage_records(path, changed) == []


def test_full_episode_optimism_uses_only_strict_full_candidates() -> None:
    metrics = full_episode_optimism_metrics(
        np.asarray((10, 11, 12)), np.asarray((True, True, False)),
        np.asarray((10, 12)), np.asarray((False, True)))

    assert metrics == {
        "full_episode_compared_candidate_count": 2,
        "optimistic_episode_pass_count": 1,
        "optimistic_episode_pass_rate": 0.5,
    }


def test_write_report_outputs_markdown_json_figures_and_pdf(tmp_path: Path) -> None:
    cache = tmp_path / "cache.npz"
    np.savez_compressed(
        cache, success=np.asarray([False]), failure_reason=np.asarray(["position"]),
        table_collision=np.asarray([False]), self_collision=np.asarray([False]),
        edge_collision=np.asarray([False]), joint_discontinuity=np.asarray([False]),
        recovery_mode=np.asarray(["hold_no_pose_candidate"]),
    )
    rows = [_pilot_row("xarm6", cache)]
    (tmp_path / "matrix.json").write_text(json.dumps(rows), encoding="utf-8")
    video = tmp_path / "videos" / "xarm6.mp4"
    video.parent.mkdir(); video.write_bytes(b"placeholder")

    payload = write_report(
        tmp_path / "matrix.json", tmp_path, expected_robots=("xarm6",),
        verify_video_decode=False)

    assert payload["complete"] is True
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "figures" / "failure_reason_heatmap.png").is_file()
    assert (tmp_path / "pdf" / "mount_ik_fidelity_pilot_report.pdf").is_file()
