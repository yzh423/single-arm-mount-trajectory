from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.multitask_fixed_time_study import TrajectorySpec
from factory_bimanual.task_family import TaskFamily
from scripts.run_piperx_controller_event_v4 import (
    EVENT_SOLVER_PROTOCOL,
    event_initializer_rows,
    event_validation_spec,
    validate_acceptance_evidence,
    validate_enforced_dynamics,
    validate_zero_safety_evidence,
)


def test_event_initializer_rows_map_poll_indices_to_event_indices():
    selected = {
        "sampled_source_rows": [0, 20, 40, 60],
        "disconnected_rows": [0, 20],
    }
    source_poll_rows = np.asarray([0, 2, 5, 40, 55, 61])

    assert event_initializer_rows(
        selected, source_poll_rows,
        source_poll_row_count=62) == [3, 5]


def test_event_validation_spec_uses_event_count_without_mutating_source_spec():
    spec = TrajectorySpec(
        family=TaskFamily("8-11", "Fold_Box"), take="161044",
        path=Path("source.csv"), source_sha256="a" * 64,
        row_count=1478)

    event_spec = event_validation_spec(spec, 755)

    assert spec.row_count == 1478
    assert event_spec.row_count == 755
    assert "controller-event" in EVENT_SOLVER_PROTOCOL


def test_enforced_dynamics_gate_accepts_values_at_official_limits():
    velocity = np.asarray([[0.0, 3.0], [-3.0, 1.0]])
    acceleration = np.asarray([[0.0, 5.0], [-5.0, 2.0]])

    assert validate_enforced_dynamics(velocity, acceleration)


@pytest.mark.parametrize(
    ("velocity", "acceleration", "message"),
    [
        (np.asarray([[3.01]]), np.asarray([[0.0]]), "velocity"),
        (np.asarray([[0.0]]), np.asarray([[5.01]]), "acceleration"),
    ],
)
def test_enforced_dynamics_gate_rejects_limit_violations(
        velocity, acceleration, message):
    with pytest.raises(ValueError, match=message):
        validate_enforced_dynamics(velocity, acceleration)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "collision": np.asarray([False, True]),
            "edge_collision": np.asarray([False, False]),
            "topology_valid": np.asarray([True, True]),
        },
        {
            "collision": np.asarray([False, False]),
            "edge_collision": np.asarray([False, True]),
            "topology_valid": np.asarray([True, True]),
        },
        {
            "collision": np.asarray([False, False]),
            "edge_collision": np.asarray([False, False]),
            "topology_valid": np.asarray([True, False]),
        },
    ],
)
def test_event_safety_gate_rejects_any_violation(payload):
    with pytest.raises(ValueError, match="violation"):
        validate_zero_safety_evidence(payload)


def test_event_acceptance_gate_rejects_mask_not_derived_from_strict_errors():
    payload = {
        "position_error_m": np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        "orientation_error_rad": np.zeros((2, 2)),
        "left_source_valid": np.ones(2, dtype=bool),
        "right_source_valid": np.ones(2, dtype=bool),
        "left_accept": np.ones(2, dtype=bool),
        "right_accept": np.ones(2, dtype=bool),
        "both_accept": np.ones(2, dtype=bool),
    }

    with pytest.raises(ValueError, match="left acceptance"):
        validate_acceptance_evidence(payload)
