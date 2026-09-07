import numpy as np
import pytest

from scripts.build_piperx_controller_event_manifest import (
    validate_archive_summary,
)
from scripts.build_piperx_multitask_fixed_time_bundle import (
    build_shard_arrays,
)
from scripts.run_piperx_controller_event_v4 import EVENT_SOLVER_PROTOCOL


def _payload():
    payload = build_shard_arrays(
        mode="baseline",
        source_time_s=np.asarray([0.0, 0.1]),
        qpos=np.zeros((2, 3)),
        position_error_m=np.zeros((2, 2)),
        orientation_error_rad=np.zeros((2, 2)),
        collision=np.zeros(2, dtype=bool),
        edge_collision=np.zeros(2, dtype=bool),
        topology_valid=np.ones(2, dtype=bool),
        left_source_valid=np.ones(2, dtype=bool),
        right_source_valid=np.ones(2, dtype=bool),
        velocity_rad_s=np.zeros((2, 2)),
        acceleration_rad_s2=np.zeros((2, 2)),
    )
    payload["solver_protocol"] = np.asarray(EVENT_SOLVER_PROTOCOL)
    return payload


def _summary():
    return {
        "source_frames": 2,
        "collision_frames": 0,
        "edge_collision_frames": 0,
        "topology_invalid_frames": 0,
        "both_accept_frames": 2,
        "both_accept_coverage": 1.0,
    }


def test_manifest_archive_gate_recomputes_safety_and_coverage():
    assert np.array_equal(
        validate_archive_summary(_summary(), _payload()),
        np.asarray([0.0, 0.1]),
    )


def test_manifest_archive_gate_rejects_unsafe_trajectory_even_if_summary_is_zero():
    payload = _payload()
    payload["collision"][0] = True

    with pytest.raises(ValueError, match="collision"):
        validate_archive_summary(_summary(), payload)


def test_manifest_archive_gate_rejects_stale_coverage_summary():
    summary = _summary()
    summary["both_accept_coverage"] = 0.5

    with pytest.raises(ValueError, match="coverage"):
        validate_archive_summary(summary, _payload())
