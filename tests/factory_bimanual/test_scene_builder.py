from pathlib import Path

import mujoco
import pytest

from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene


@pytest.mark.parametrize("robot", tuple(ROBOT_CONTRACTS))
def test_generated_scene_compiles_with_two_native_dof_arms(tmp_path: Path, robot: str):
    contract = ROBOT_CONTRACTS[robot]
    source_before = contract.source_urdf.read_bytes()
    output = tmp_path / f"{robot}.xml"
    manifest = build_same_model_scene(contract, 0.8, output)
    model = mujoco.MjModel.from_xml_path(str(output))

    assert model.nv >= 2 * contract.dof_per_arm  # Panda also retains two fingers per hand.
    assert manifest.spacing_m == 0.8
    assert manifest.left_base_position == pytest.approx(
        (-0.4, 0.0, manifest.table_height_m + 0.08))
    assert manifest.right_base_position == pytest.approx(
        (0.4, 0.0, manifest.table_height_m + 0.08))
    assert contract.source_urdf.read_bytes() == source_before
    for name in contract.prefixed_joint_names("left") + contract.prefixed_joint_names("right"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
    for name in ("left_target", "right_target", "left_tcp", "right_tcp", "workbench"):
        obj = (mujoco.mjtObj.mjOBJ_GEOM if name == "workbench" else
               mujoco.mjtObj.mjOBJ_BODY if name.endswith("target") else mujoco.mjtObj.mjOBJ_SITE)
        assert mujoco.mj_name2id(model, obj, name) >= 0


def test_spacing_must_be_positive(tmp_path: Path):
    with pytest.raises(ValueError, match="spacing"):
        build_same_model_scene(ROBOT_CONTRACTS["xarm6"], 0.0, tmp_path / "bad.xml")


@pytest.mark.parametrize("robot", tuple(ROBOT_CONTRACTS))
def test_both_bases_are_upright_on_same_table_height_and_face_each_other(tmp_path, robot):
    output = tmp_path / f"{robot}.xml"
    manifest = build_same_model_scene(ROBOT_CONTRACTS[robot], 0.8, output)
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_base_mount")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_base_mount")
    assert left >= 0 and right >= 0
    assert data.xpos[left, 2] == pytest.approx(data.xpos[right, 2])
    assert data.xmat[left].reshape(3, 3)[:, 2] == pytest.approx([0, 0, 1])
    assert data.xmat[right].reshape(3, 3)[:, 2] == pytest.approx([0, 0, 1])
    assert data.xmat[left].reshape(3, 3)[:2, 0] == pytest.approx(
        -data.xmat[right].reshape(3, 3)[:2, 0]
    )
    assert manifest.left_base_position[2] == manifest.right_base_position[2]


@pytest.mark.parametrize("robot", tuple(ROBOT_CONTRACTS))
def test_generated_scene_uses_native_joint_ranges_not_shrunk_canonical_ranges(tmp_path, robot):
    contract = ROBOT_CONTRACTS[robot]
    output = tmp_path / f"{robot}.xml"
    build_same_model_scene(contract, 0.8, output)
    model = mujoco.MjModel.from_xml_path(str(output))
    for side in ("left", "right"):
        actual = []
        for name in contract.prefixed_joint_names(side):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actual.append(tuple(model.jnt_range[joint]))
        import numpy as np
        np.testing.assert_allclose(actual, contract.joint_limits_rad, atol=1e-9)


@pytest.mark.parametrize("robot", ("franka_panda", "i2rt_yam"))
def test_tcp_matches_learning_reference_parent_and_tool_offset(tmp_path, robot):
    contract = ROBOT_CONTRACTS[robot]
    output = tmp_path / f"{robot}.xml"
    build_same_model_scene(contract, 0.8, output)
    model = mujoco.MjModel.from_xml_path(str(output))
    for side in ("left", "right"):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        parent = model.site_bodyid[site]
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) == f"{side}_{contract.tcp_link_name}"
        assert model.site_pos[site] == pytest.approx(contract.tcp_offset_m)


def test_official_piperx_collision_geometries_keep_side_and_link_names(tmp_path):
    output = tmp_path / "piperx.xml"
    build_same_model_scene(ROBOT_CONTRACTS["piperx"], .8, output)
    model = mujoco.MjModel.from_xml_path(str(output))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
             for index in range(model.ngeom)]
    assert "left_base_link_collision_0" in names
    assert "right_gripper_base_collision_0" in names
    assert all(name is not None for name in names)
