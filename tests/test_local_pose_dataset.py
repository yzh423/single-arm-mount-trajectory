from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from design_optimization.local_pose_dataset import (
    assign_stratified_splits,
    clean_pose_frame,
    content_split,
    infer_hands,
    prepare_local_pose_benchmark,
    trajectory_has_motion,
)


def _pose_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [0.0, 0.01, 0.02, 0.03],
            "right_tcp_valid": [1, 1, 1, 1],
            "right_tcp_pos_x": [0.0, 0.01, np.nan, 0.03],
            "right_tcp_pos_y": [0.0, 0.0, np.nan, 0.0],
            "right_tcp_pos_z": [0.2, 0.2, np.nan, 0.2],
            "right_tcp_quat_w": [1.0, -1.0, np.nan, -1.0],
            "right_tcp_quat_x": [0.0, 0.0, np.nan, 0.0],
            "right_tcp_quat_y": [0.0, 0.0, np.nan, 0.0],
            "right_tcp_quat_z": [0.0, 0.0, np.nan, 0.0],
        }
    )


def test_batch_hand_routing_matches_approved_source_rules():
    assert infer_hands("7-27", "open-box") == ("right",)
    assert infer_hands("7-28", "open-box-3") == ("right",)
    assert infer_hands("8-05", "cap-left") == ("left",)
    assert infer_hands("8-05", "fold-towel-dual") == ("left", "right")
    assert infer_hands("8-06", "pour-water") == ("left", "right")


def test_content_split_is_deterministic_and_keeps_duplicates_together():
    digest = hashlib.sha256(b"same episode").hexdigest()
    first = content_split(digest)
    second = content_split(digest)
    assert first == second
    assert first in {"train", "validation", "test"}


def test_clean_pose_frame_repairs_bounded_gap_and_canonicalizes_quaternion():
    cleaned, record = clean_pose_frame(
        _pose_frame(),
        hand="right",
        maximum_internal_repair_samples=1,
        exclude_invalid_fraction=0.5,
    )
    assert record.disposition == "eligible"
    np.testing.assert_array_equal(cleaned.source_rows, np.arange(4))
    np.testing.assert_allclose(cleaned.position_m[2], [0.02, 0.0, 0.2])
    assert np.all(np.sum(cleaned.quaternion_wxyz[1:] * cleaned.quaternion_wxyz[:-1], axis=1) >= 0)
    np.testing.assert_allclose(np.linalg.norm(cleaned.quaternion_wxyz, axis=1), 1.0)


def test_clean_pose_frame_rejects_time_reversal():
    frame = _pose_frame()
    frame.loc[2, "t"] = -1.0
    cleaned, record = clean_pose_frame(frame, hand="right")
    assert cleaned is None
    assert record.disposition == "excluded_structural"
    assert record.reason_counts["time_reversal"] == 1


def test_static_episode_is_not_trajectory_eligible_but_rotation_only_is():
    position = np.zeros((4, 3))
    static_q = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    assert not trajectory_has_motion(position, static_q)
    angle = np.deg2rad(3.0)
    rotating_q = static_q.copy()
    rotating_q[-1] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    assert trajectory_has_motion(position, rotating_q)


def test_prepare_benchmark_writes_terminal_records_and_split_safe_episodes(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "processed"
    right = source / "7-27" / "open-box"
    dual = source / "8-06" / "pour-water"
    right.mkdir(parents=True)
    dual.mkdir(parents=True)
    frame = _pose_frame()
    for hand in ("left", "right"):
        for name in ("valid",):
            frame[f"{hand}_tcp_{name}"] = frame[f"right_tcp_{name}"]
        for channel in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"):
            frame[f"{hand}_tcp_{channel}"] = frame[f"right_tcp_{channel}"]
    frame.to_csv(right / "episode.csv", index=False)
    frame.to_csv(dual / "duplicate.csv", index=False)

    manifest = prepare_local_pose_benchmark(source, output)

    assert manifest["source_file_count"] == 2
    assert len(manifest["source_records"]) == 2
    assert len(manifest["episodes"]) == 3
    assert {row["hand"] for row in manifest["episodes"]} == {"left", "right"}
    assert all((output / row["artifact"]).exists() for row in manifest["episodes"])
    split_by_hash = {}
    for row in manifest["episodes"]:
        split_by_hash.setdefault(row["content_sha256"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_hash.values())
    assert (output / "manifest.json").exists()


def test_prepare_cli_run_consumes_versioned_yaml(tmp_path):
    from scripts.prepare_local_pose_benchmark import run

    source = tmp_path / "source"
    output = tmp_path / "output"
    task = source / "7-27" / "open-box"
    task.mkdir(parents=True)
    _pose_frame().to_csv(task / "episode.csv", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "source_root": str(source),
                "output_root": str(output),
                "cleaning": {
                    "maximum_internal_repair_samples": 1,
                    "exclude_invalid_fraction": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = run(config)
    assert manifest["source_file_count"] == 1
    assert len(manifest["episodes"]) == 1


def test_prepare_cli_is_directly_executable():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/prepare_local_pose_benchmark.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_task_stratified_split_keeps_every_nontrivial_task_evaluable():
    rows = [
        {
            "task": "task-a",
            "content_sha256": hashlib.sha256(f"episode-{index}".encode()).hexdigest(),
        }
        for index in range(10)
    ]
    rows.append(dict(rows[0]))
    assigned = assign_stratified_splits(rows)
    assert {row["split"] for row in assigned} == {"train", "validation", "test"}
    duplicate_splits = {
        row["split"] for row in assigned
        if row["content_sha256"] == rows[0]["content_sha256"]
    }
    assert len(duplicate_splits) == 1


def test_dual_hands_from_same_source_episode_share_split():
    rows = []
    for index in range(10):
        source = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        for hand in ("left", "right"):
            rows.append({"task": "dual-task", "hand": hand, "source_sha256": source,
                         "content_sha256": hashlib.sha256(f"{source}-{hand}".encode()).hexdigest()})
    assigned = assign_stratified_splits(rows)
    for source in {row["source_sha256"] for row in assigned}:
        assert len({row["split"] for row in assigned if row["source_sha256"] == source}) == 1
