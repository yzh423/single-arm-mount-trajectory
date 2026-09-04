from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.multitask_fixed_time_study import (
    STUDY_MODES,
    TrajectorySpec,
    discover_dual_hand_trajectories,
    validate_shard,
)
from factory_bimanual.task_family import TaskFamily


ROOT = Path(__file__).resolve().parents[2]


def _spec(tmp_path):
    source = tmp_path / "handheld_20260811_161044.csv"
    source.write_text("t\n5.0\n5.125\n5.25\n", encoding="utf-8")
    return TrajectorySpec(
        family=TaskFamily("8-11", "Fold_Box"),
        take="161044",
        path=source,
        source_sha256="a" * 64,
        row_count=3,
    )


def _valid_payload():
    source_time = np.asarray([0.0, 0.125, 0.25])
    return {
        "schema": "piperx-multitask-fixed-time-shard-v1",
        "mode": "upright_table",
        "source_time_s": source_time,
        "fixed_time_s": source_time.copy(),
        "retiming_applied": False,
        "qpos": np.zeros((3, 18)),
        "left_source_valid": np.ones(3, dtype=bool),
        "right_source_valid": np.ones(3, dtype=bool),
        "left_accept": np.ones(3, dtype=bool),
        "right_accept": np.ones(3, dtype=bool),
        "both_accept": np.ones(3, dtype=bool),
        "position_error_m": np.zeros((3, 2)),
        "orientation_error_rad": np.zeros((3, 2)),
        "collision": np.zeros(3, dtype=bool),
        "edge_collision": np.zeros(3, dtype=bool),
        "topology_valid": np.ones(3, dtype=bool),
        "velocity_rad_s": np.zeros((3, 12)),
        "acceleration_rad_s2": np.zeros((3, 12)),
    }


def test_repository_inventory_is_all_nonrejected_dual_hand_data():
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")

    assert len(specs) == 27
    assert len({item.family.key for item in specs}) == 12
    assert all("_rejected_short" not in item.path.parts for item in specs)
    assert all(item.row_count > 1 for item in specs)
    assert tuple(sorted(STUDY_MODES)) == (
        "baseline", "horizontal_wall", "inverted", "upright_table")


def test_fixed_time_shard_rejects_substituted_timestamps(tmp_path):
    payload = _valid_payload()
    validate_shard(payload, _spec(tmp_path), "upright_table")
    payload["fixed_time_s"][2] += 0.01

    with pytest.raises(ValueError, match="source timestamps"):
        validate_shard(payload, _spec(tmp_path), "upright_table")


def test_fixed_time_shard_rejects_timeline_that_differs_from_raw_csv(tmp_path):
    payload = _valid_payload()
    payload["source_time_s"][1] = 0.11
    payload["fixed_time_s"][1] = 0.11

    with pytest.raises(ValueError, match="raw CSV"):
        validate_shard(payload, _spec(tmp_path), "upright_table")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("collision", np.asarray([False, True, False]), "collision-free"),
        ("edge_collision", np.asarray([False, False, True]), "edge-collision-free"),
        ("topology_valid", np.asarray([True, False, True]), "topology-valid"),
    ],
)
def test_shard_rejects_any_unsafe_frame(tmp_path, field, value, message):
    payload = _valid_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_shard(payload, _spec(tmp_path), "upright_table")


@pytest.mark.parametrize(
    "field",
    ["source_time_s", "fixed_time_s", "qpos", "position_error_m",
     "orientation_error_rad", "velocity_rad_s", "acceleration_rad_s2"],
)
def test_shard_rejects_nonfinite_numeric_evidence(tmp_path, field):
    payload = _valid_payload()
    payload[field] = np.asarray(payload[field], dtype=float).copy()
    payload[field].flat[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        validate_shard(payload, _spec(tmp_path), "upright_table")


def test_shard_rejects_accept_flags_that_disagree_with_pose_errors(tmp_path):
    payload = _valid_payload()
    payload["position_error_m"][1, 0] = 0.002

    with pytest.raises(ValueError, match="pose tolerances"):
        validate_shard(payload, _spec(tmp_path), "upright_table")


def test_shard_rejects_acceptance_on_invalid_source_pose(tmp_path):
    payload = _valid_payload()
    payload["left_source_valid"][1] = False

    with pytest.raises(ValueError, match="source-valid"):
        validate_shard(payload, _spec(tmp_path), "upright_table")


def test_shard_requires_explicit_failed_and_unsafe_evidence(tmp_path):
    payload = _valid_payload()
    del payload["collision"]

    with pytest.raises(ValueError, match="collision"):
        validate_shard(payload, _spec(tmp_path), "upright_table")
