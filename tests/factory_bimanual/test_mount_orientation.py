from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.mount_orientation import mount_quaternions
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene


def _base_axes(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    values = {}
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_base_mount")
        values[side] = data.xmat[body_id].reshape(3, 3)[:, 2]
    return values


def test_three_mount_modes_compile_with_expected_axes_and_shared_z(tmp_path: Path):
    xy = {"left": (-.3, .2), "right": (.3, -.2)}
    target = np.asarray([0., 0., 1.05])
    expected = {
        "upright_table": {"left": [0, 0, 1], "right": [0, 0, 1]},
        "horizontal_wall": {
            side: ((np.asarray(xy[side]) - target[:2])
                   / np.linalg.norm(np.asarray(xy[side]) - target[:2])).tolist()
                  + [0]
            for side in ("left", "right")
        },
        "inverted": {"left": [0, 0, -1], "right": [0, 0, -1]},
    }
    for mode in expected:
        quaternions = mount_quaternions(mode, xy, target, {"left": 15., "right": -20.})
        path = tmp_path / f"{mode}.xml"
        manifest = build_same_model_scene(
            ROBOT_CONTRACTS["piperx"], .6, path,
            mount_xy_m=xy, mount_base_z_m=.9,
            mount_quaternion_wxyz=quaternions, mount_support_mode=mode)
        model = mujoco.MjModel.from_xml_path(str(path))
        assert manifest.left_base_position[2] == manifest.right_base_position[2] == .9
        for side, axis in _base_axes(model).items():
            np.testing.assert_allclose(axis, expected[mode][side], atol=2e-6)
        if mode == "horizontal_wall":
            for side in ("left", "right"):
                assert mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM,
                    f"{side}_mount_pole") >= 0
                assert mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM,
                    f"{side}_mount_wall") == -1
        if mode == "inverted":
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "mount_ceiling") >= 0


def test_mount_yaw_does_not_change_base_axis():
    xy = {"left": (-.3, .2), "right": (.3, -.2)}
    target = np.asarray([0., 0., 1.05])
    a = mount_quaternions("horizontal_wall", xy, target, {"left": 0., "right": 0.})
    b = mount_quaternions("horizontal_wall", xy, target, {"left": 80., "right": -50.})
    for side in ("left", "right"):
        ma = np.empty(9); mb = np.empty(9)
        mujoco.mju_quat2Mat(ma, a[side]); mujoco.mju_quat2Mat(mb, b[side])
        np.testing.assert_allclose(
            ma.reshape(3, 3)[:, 2], mb.reshape(3, 3)[:, 2], atol=1e-12)


def test_horizontal_forward_axes_are_parallel_and_point_to_task_center(tmp_path):
    xy = {"left": (-.4, -.3), "right": (.4, -.3)}
    target = np.asarray([0., .2, 1.05])
    quaternions = mount_quaternions(
        "horizontal_forward", xy, target, {"left": 20., "right": -25.})
    path = tmp_path / "horizontal_forward.xml"
    build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], .8, path, mount_xy_m=xy,
        mount_base_z_m=target[2], mount_quaternion_wxyz=quaternions,
        mount_support_mode="horizontal_forward")
    model = mujoco.MjModel.from_xml_path(str(path))
    expected = np.asarray([0., 1., 0.])
    for side, axis in _base_axes(model).items():
        np.testing.assert_allclose(axis, expected, atol=2e-6)
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_mount_pole") >= 0
