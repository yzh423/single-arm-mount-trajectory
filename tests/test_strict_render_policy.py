import mujoco
import numpy as np

from scripts.render_strict_single_arm_task import FPS, playback_frame_count
from scripts.solve_strict_urdf_task_cache import build_model


def test_video_uses_original_playback_duration():
    assert FPS == 30
    assert playback_frame_count(np.asarray([0.0, 20.0])) == 600


def test_render_pedestal_is_narrow_and_translucent():
    model = build_model("xarm6", np.asarray([0.0, 0.0, 0.4]), 0.0, render_studio=True)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strict_pedestal")
    assert model.geom_size[geom, 0] <= 0.075
    assert model.geom_rgba[geom, 3] <= 0.5


def test_pedestal_stays_vertical_and_adapter_carries_mount_tilt():
    model = build_model("xarm6", np.asarray([0.0, -0.2, 0.4]), 45.0, render_studio=True)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strict_pedestal")
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    normal = data.geom_xmat[geom].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-12)
    adapter = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strict_mount_adapter")
    assert adapter >= 0
    np.testing.assert_allclose(data.geom_xpos[adapter], [0.0, -0.2, 0.4], atol=1e-12)
