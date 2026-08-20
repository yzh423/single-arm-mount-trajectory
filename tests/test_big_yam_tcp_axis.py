import numpy as np
import mujoco

from scripts.strict_urdf_model_audit import MODELS


def test_big_yam_tcp_extends_toward_the_native_gripper_tips():
    np.testing.assert_allclose(MODELS["big_yam"].tool_translation_m, [0.0, 0.0, -0.13])


def test_canonical_tcp_z_axis_matches_each_native_extension_axis():
    for robot in ("arx_x5", "big_yam", "willow", "nero"):
        entry = MODELS[robot]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(entry.tool_quaternion_wxyz))
        expected = np.asarray(entry.tool_axis, dtype=float)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(rotation.reshape(3, 3)[:, 2], expected, atol=1e-12)


def test_missing_vendor_tcp_uses_the_common_130mm_fallback_axis():
    np.testing.assert_allclose(MODELS["nero"].tool_translation_m, [0.0, 0.0, 0.13])
    np.testing.assert_allclose(MODELS["willow"].tool_translation_m, [-0.13, 0.0, 0.0])
