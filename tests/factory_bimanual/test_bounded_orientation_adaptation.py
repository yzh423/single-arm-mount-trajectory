import numpy as np
import pytest

from factory_bimanual.bounded_orientation_adaptation import (
    OrientationAdaptationConfig,
    orientation_adaptation_candidates,
    orientation_offset_angles,
)


def _quat(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = np.deg2rad(degrees) / 2.0
    return np.r_[np.cos(half), axis * np.sin(half)]


def test_candidates_are_layered_original_then_tool_axis_then_full_pose():
    config = OrientationAdaptationConfig(
        tool_soft_limit_deg=15.0,
        tool_hard_limit_deg=30.0,
        swing_soft_limit_deg=7.5,
        swing_hard_limit_deg=15.0,
        tool_step_deg=15.0,
        swing_step_deg=7.5,
    )

    candidates = orientation_adaptation_candidates(
        np.array([1.0, 0.0, 0.0, 0.0]), config=config)

    assert candidates[0].level == "original"
    assert candidates[0].tool_axis_offset_deg == pytest.approx(0.0)
    assert candidates[0].swing_offset_deg == pytest.approx(0.0)
    tool = [item for item in candidates if item.level == "tool_axis"]
    full = [item for item in candidates if item.level == "full_pose"]
    assert {round(item.tool_axis_offset_deg) for item in tool} == {
        -30, -15, 15, 30}
    assert full
    assert max(abs(item.tool_axis_offset_deg) for item in candidates) <= 30.0
    assert max(item.swing_offset_deg for item in candidates) <= 15.0
    assert all(np.linalg.norm(item.quaternion_wxyz) == pytest.approx(1.0)
               for item in candidates)


def test_previous_offset_is_included_and_clipped_to_hard_limits():
    config = OrientationAdaptationConfig()
    candidates = orientation_adaptation_candidates(
        np.array([1.0, 0.0, 0.0, 0.0]),
        previous_offset_rotvec_rad=np.deg2rad([40.0, -20.0, 50.0]),
        config=config,
    )

    continuation = [item for item in candidates if item.source == "continuation"]
    assert len(continuation) == 1
    tool_deg, swing_deg = orientation_offset_angles(
        continuation[0].offset_rotvec_rad)
    assert abs(tool_deg) <= config.tool_hard_limit_deg + 1e-10
    assert swing_deg <= config.swing_hard_limit_deg + 1e-10


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError, match="soft"):
        OrientationAdaptationConfig(
            tool_soft_limit_deg=31.0, tool_hard_limit_deg=30.0)
    with pytest.raises(ValueError, match="positive"):
        OrientationAdaptationConfig(tool_step_deg=0.0)

