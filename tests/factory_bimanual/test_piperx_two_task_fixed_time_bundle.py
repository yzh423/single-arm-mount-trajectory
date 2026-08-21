import numpy as np
import pytest

from scripts.build_piperx_two_task_fixed_time_bundle import (
    build_fixed_time_arrays,
    validate_fixed_time_arrays,
)
from scripts.build_piperx_fixed_time_report import validate_report_manifest


def _source_payload():
    return {
        "source_time_s": np.asarray([0.0, 0.1, 0.25]),
        "source_qpos": np.arange(12, dtype=float).reshape(3, 4),
        "source_reached": np.ones(3, dtype=bool),
        "fixed_time_accepted": np.asarray([True, False, False]),
        "collision": np.zeros(3, dtype=bool),
        "source_velocity_rad_s": np.ones((3, 4)),
        "source_acceleration_rad_s2": np.ones((3, 4)),
    }


def test_fixed_time_arrays_use_source_schedule_and_source_qpos_exactly():
    source = _source_payload()

    fixed = build_fixed_time_arrays(source)

    np.testing.assert_array_equal(fixed["fixed_time_s"], source["source_time_s"])
    np.testing.assert_array_equal(fixed["fixed_time_qpos"], source["source_qpos"])
    assert fixed["timing_mode"].item() == "fixed_source_time"
    assert "execution_time_s" not in fixed
    assert "execution_qpos" not in fixed
    validate_fixed_time_arrays(fixed)


def test_fixed_time_validator_rejects_any_schedule_or_qpos_substitution():
    fixed = build_fixed_time_arrays(_source_payload())
    fixed["fixed_time_s"] = fixed["fixed_time_s"].copy()
    fixed["fixed_time_s"][1] += 0.01
    with pytest.raises(ValueError, match="source timestamps"):
        validate_fixed_time_arrays(fixed)

    fixed = build_fixed_time_arrays(_source_payload())
    fixed["fixed_time_qpos"] = fixed["fixed_time_qpos"].copy()
    fixed["fixed_time_qpos"][1, 0] += 0.01
    with pytest.raises(ValueError, match="source qpos"):
        validate_fixed_time_arrays(fixed)


def test_fixed_time_validator_requires_complete_zero_collision_source_path():
    source = _source_payload()
    source["collision"][2] = True
    with pytest.raises(ValueError, match="zero collision"):
        build_fixed_time_arrays(source)

    source = _source_payload()
    source["source_reached"][1] = False
    with pytest.raises(ValueError, match="complete source reach"):
        build_fixed_time_arrays(source)


def test_fixed_time_report_rejects_any_retiming_claim():
    task = {
        "timing": {
            "timestamp_identity": True,
            "qpos_identity": True,
            "inserted_frames": 0,
            "added_duration_s": 0.0,
        },
        "tracking": {"coverage": 1.0, "collision_frames": 0},
        "dynamics": {"limits_passed": False},
    }
    manifest = {
        "schema": "piperx-two-task-fixed-time-manifest-v1",
        "timing_mode": "fixed_source_time",
        "retiming_applied": False,
        "tasks": {"fold_box": task, "seal_bag": task.copy()},
    }
    validate_report_manifest(manifest)
    manifest["retiming_applied"] = True
    with pytest.raises(ValueError, match="must not contain retiming"):
        validate_report_manifest(manifest)
