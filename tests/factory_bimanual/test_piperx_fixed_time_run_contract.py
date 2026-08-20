import numpy as np
import pytest

from factory_bimanual.fixed_time_run_contract import (
    source_time_schedule,
    validate_synchronized_mount,
)


def test_source_time_schedule_preserves_relative_timestamps_without_extension():
    execution_time, intervals = source_time_schedule(
        np.asarray([12.0, 12.1, 12.35, 12.5]))

    np.testing.assert_allclose(execution_time, [0.0, 0.1, 0.35, 0.5])
    np.testing.assert_allclose(intervals, [0.1, 0.1, 0.25, 0.15])


@pytest.mark.parametrize("values", [
    [], [0.0], [0.0, 0.0], [0.0, np.nan], [0.0, 1.0, 0.5],
])
def test_source_time_schedule_rejects_invalid_timestamps(values):
    with pytest.raises(ValueError):
        source_time_schedule(np.asarray(values, dtype=float))


def test_synchronized_mount_requires_upright_shared_height_and_separation():
    mount = {
        "xy": {"left": [-0.3, 0.3], "right": [-0.3, -0.4]},
        "yaw": {"left": 15.0, "right": 15.0},
        "shared_base_z_m": 0.81,
        "roll": {"left": 0.0, "right": 0.0},
        "pitch": {"left": 0.0, "right": 0.0},
    }

    assert validate_synchronized_mount(mount) == pytest.approx(0.7)


def test_synchronized_mount_rejects_close_or_tilted_bases():
    close = {
        "xy": {"left": [-0.2, 0.2], "right": [-0.2, -0.2]},
        "yaw": {"left": 0.0, "right": 0.0},
        "shared_base_z_m": 0.81,
    }
    with pytest.raises(ValueError, match="separation"):
        validate_synchronized_mount(close)

    tilted = {
        **close,
        "xy": {"left": [-0.3, 0.3], "right": [-0.3, -0.4]},
        "roll": {"left": 1.0, "right": 0.0},
    }
    with pytest.raises(ValueError, match="upright"):
        validate_synchronized_mount(tilted)
