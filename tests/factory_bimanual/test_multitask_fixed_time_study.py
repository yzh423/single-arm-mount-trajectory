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


def _spec():
    return TrajectorySpec(
        family=TaskFamily("8-11", "Fold_Box"),
        take="161044",
        path=ROOT / "data/factory/8-11/Fold_Box/handheld_20260811_161044.csv",
        source_sha256="a" * 64,
        row_count=3,
    )


def _valid_payload():
    source_time = np.asarray([0.0, 0.1, 0.25])
    return {
        "schema": "piperx-multitask-fixed-time-shard-v1",
        "mode": "upright_table",
        "source_time_s": source_time,
        "fixed_time_s": source_time.copy(),
        "retiming_applied": False,
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


def test_fixed_time_shard_rejects_substituted_timestamps():
    payload = _valid_payload()
    validate_shard(payload, _spec(), "upright_table")
    payload["fixed_time_s"][2] += 0.01

    with pytest.raises(ValueError, match="source timestamps"):
        validate_shard(payload, _spec(), "upright_table")


def test_shard_requires_explicit_failed_and_unsafe_evidence():
    payload = _valid_payload()
    del payload["collision"]

    with pytest.raises(ValueError, match="collision"):
        validate_shard(payload, _spec(), "upright_table")
