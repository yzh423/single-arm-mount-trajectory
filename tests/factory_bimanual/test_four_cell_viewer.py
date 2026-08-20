import numpy as np
import pytest

import mujoco

from factory_bimanual.four_cell_viewer import LightweightFourCellIK, assert_reference_demo_only, lightweight_scene_text, load_reference_model, mapped_relative_pose


def test_factory_pose_is_retargeted_relative_to_cell_start():
    p0 = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.2, 3.3])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    cell_p = np.array([4.0, 5.0, 6.0])
    cell_q = np.array([1.0, 0.0, 0.0, 0.0])
    out_p, out_q = mapped_relative_pose(p, q0, p0, q0, cell_p, cell_q)
    np.testing.assert_allclose(out_p, cell_p + np.array([0.2, -0.1, 0.3]))
    np.testing.assert_allclose(out_q, cell_q)


def test_factory_orientation_delta_is_preserved_and_normalized():
    half = np.sqrt(0.5)
    q = np.array([half, 0.0, 0.0, half])
    _, out = mapped_relative_pose(
        np.zeros(3), q, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert np.linalg.norm(out) == 1.0
    np.testing.assert_allclose(np.abs(out), np.abs(np.array([half, 0.0, 0.0, -half])))


def test_reference_four_cell_scene_loads_without_editing_source():
    model = load_reference_model()
    assert model.nmocap == 8
    assert model.nv >= 48


def test_lightweight_controller_initializes_all_eight_arms_without_scene_copies():
    model = load_reference_model()
    controller = LightweightFourCellIK(model, mujoco.MjData(model))
    assert len(controller.arms) == 8
    assert not np.allclose(controller.data.qpos, 0.0)
    controller.step()
    assert controller.last_table_rollback is False


def test_lightweight_viewer_scene_removes_expensive_render_settings():
    text = lightweight_scene_text()
    assert 'shadowsize="4096"' not in text
    assert 'offsamples="8"' not in text
    assert 'njmax="3000"' in text
    assert 'name="willow_table_leg_1"' in text


def test_reference_four_cell_demo_is_rejected_for_factory_experiment():
    with pytest.raises(RuntimeError, match="Willow.*Doosan.*not the requested"):
        assert_reference_demo_only()
