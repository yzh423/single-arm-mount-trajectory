import json
from pathlib import Path

import numpy as np
import pytest

from design_optimization.local_pose_sampling import relative_pose_sample
from scripts.strict_trajectory_sources import (
    RelativeTrajectory, _relative, load_relative_task_trajectory,
    place_relative_positions, resample_trajectory,
    minimum_safe_anchor_z,
)


def test_local_explicit_episode_artifact_is_not_replaced_by_split_selection():
    root = ROOT / "data/processed/local_pose_benchmark"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    episodes = [row for row in manifest["episodes"] if row["task"] == "pick-right-left"][:3]
    trajectories = [load_relative_task_trajectory(
        "local", "pick-right-left", episode_artifact=row["artifact"])
        for row in episodes]
    assert [Path(item.source).name for item in trajectories] == [Path(row["artifact"]).name for row in episodes]
    assert len({len(item.time_s) for item in trajectories}) == 3


ROOT = Path(__file__).resolve().parents[1]


def test_local_uses_full_cleaned_episode_not_twenty_point_sample():
    sample = next(row for row in json.loads((ROOT / "data/processed/local_pose_benchmark/test_samples.json").read_text())
                  if row["task"] == "open-box-2")
    trajectory = load_relative_task_trajectory("local", "open-box-2")
    assert len(trajectory.position_m) > len(sample["relative_position_m"])
    assert np.linalg.norm(np.diff(trajectory.position_m, axis=0), axis=1).sum() > 0.1
    assert trajectory.source.endswith(".npz")


def test_local_validation_and_test_trajectories_are_split_isolated():
    validation = load_relative_task_trajectory("local", "open-box", split="validation")
    test = load_relative_task_trajectory("local", "open-box", split="test")
    assert validation.source != test.source


def test_external_sources_use_contiguous_original_frames():
    required = (
        ROOT / "data/processed/external_domains/droid_samples.json",
        ROOT / "data/processed/external_domains/egodex_samples.json",
        ROOT / "data/DROID",
        ROOT / "data/EgoDex/pose_only_test/egodex_pose_only_test.npz",
    )
    if not all(path.exists() for path in required):
        pytest.skip("optional DROID/EgoDex corpora are excluded from this project bundle")
    for domain in ("droid", "egodex"):
        trajectory = load_relative_task_trajectory(domain, "droid_01" if domain == "droid" else "add_remove_lid")
        assert len(trajectory.position_m) > 8
        assert len(trajectory.time_s) == len(trajectory.quaternion_wxyz)


def test_relative_translation_preserves_dataset_world_frame():
    half = np.sqrt(0.5)
    trajectory = _relative(
        np.asarray((0.0, 1.0)),
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        np.asarray(((half, 0.0, 0.0, half), (half, 0.0, 0.0, half))),
        "right",
        "synthetic",
    )
    # Translation is independent of the first wrist attitude. Rotating it into
    # the initial TCP frame made identical captured paths change shape solely
    # because the operator began with a different hand orientation.
    np.testing.assert_allclose(trajectory.position_m[1], (1.0, 0.0, 0.0), atol=1e-12)


def test_local_sample_translation_is_expressed_in_initial_tcp_frame():
    half = np.sqrt(0.5)
    data = {
        "time_s": np.asarray((0.0, 1.0, 2.0, 3.0, 4.0)),
        "position_m": np.asarray(((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), float),
        "quaternion_wxyz": np.tile((half, 0.0, 0.0, half), (5, 1)),
    }
    position, _ = relative_pose_sample(data, 2)
    np.testing.assert_allclose(position[1], (0.0, -1.0, 0.0), atol=1e-12)


def test_resampling_uses_uniform_time_not_uniform_source_indices():
    trajectory = RelativeTrajectory(
        time_s=np.asarray((0.0, 0.01, 0.02, 0.9, 1.0)),
        position_m=np.asarray(((0, 0, 0), (0.01, 0, 0), (0.02, 0, 0),
                               (0.9, 0, 0), (1.0, 0, 0)), float),
        quaternion_wxyz=np.tile((1.0, 0.0, 0.0, 0.0), (5, 1)),
        hand="right", source="synthetic",
    )
    sampled = resample_trajectory(trajectory, maximum_frames=4)
    np.testing.assert_allclose(sampled.time_s, np.linspace(0.0, 1.0, 4))
    np.testing.assert_allclose(sampled.position_m[:, 0], sampled.time_s)


def test_local_world_translation_is_placed_without_axis_flip():
    relative = np.asarray(((0.0, 0.0, 0.0), (0.1, -0.2, 0.3)))
    placed = place_relative_positions(relative, hand="right", anchor_z_m=0.35)
    np.testing.assert_allclose(placed[1] - placed[0], relative[1])


def test_task_anchor_is_lowest_value_that_preserves_table_clearance():
    assert minimum_safe_anchor_z(-0.19, clearance_m=0.10) == pytest.approx(0.29)
    assert minimum_safe_anchor_z(-0.02, clearance_m=0.10) == pytest.approx(0.20)
