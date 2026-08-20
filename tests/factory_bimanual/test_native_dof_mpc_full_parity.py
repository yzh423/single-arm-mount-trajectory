from __future__ import annotations

import inspect
import sys
from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.native_dof_mpc import NativeDofDualArmMPCPVT
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene


EXTERNAL_ROOT = Path(r"E:\YZH123123\GoodGoodArmDayDayUp-Learning")


def _external_class():
    sys.path.insert(0, str(EXTERNAL_ROOT))
    try:
        from doosan_teleop.mpc_pvt import DualArmMPCPVT
        return DualArmMPCPVT
    finally:
        sys.path.remove(str(EXTERNAL_ROOT))


def test_native_port_preserves_complete_external_mpc_method_surface():
    expected = {name for name, value in inspect.getmembers(_external_class(), inspect.isfunction)}
    actual = {name for name, value in inspect.getmembers(NativeDofDualArmMPCPVT, inspect.isfunction)}
    assert expected <= actual


def test_native_port_runs_candidate_pipeline_for_seven_dof_panda(tmp_path):
    contract = ROBOT_CONTRACTS["franka_panda"]
    xml = tmp_path / "panda.xml"
    manifest = build_same_model_scene(contract, 0.8, xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    name_map = {
        side: {
            "joints": getattr(manifest, f"{side}_joint_names"),
            "site": f"{side}_tcp",
            "target": f"{side}_target",
            "gripper_prefix": f"{side}_",
        }
        for side in ("left", "right")
    }
    velocity = {side: np.ones(7) for side in ("left", "right")}
    controller = NativeDofDualArmMPCPVT(
        model, data, name_map=name_map, robot_kind="franka_panda", velocity_limits=velocity
    )
    controller.initialize_home()
    controller.step()
    assert all(arm.dq_des.shape == (7,) for arm in controller.arms.values())
    assert all(arm.last_debug["candidates"] >= 5 for arm in controller.arms.values())


def test_six_dof_full_step_matches_external_candidate_policy():
    # A serial zero-length chain is sufficient to compare deterministic policy,
    # filtering, candidate scoring and state updates with collision disabled.
    sides = []
    for side, x in (("left", -0.4), ("right", 0.4)):
        chain = "".join(
                f'<body name="{side}_b{i}"><joint name="{side}_j{i}" type="hinge" range="-3 3"/><geom type="sphere" size="0.01" mass="0.1"/>'
            for i in range(6)
        ) + f'<site name="{side}_tcp"/>' + '</body>' * 6
        sides.append(f'<body name="{side}_root" pos="{x} 0 0">{chain}</body>')
        sides.append(f'<body name="{side}_target" mocap="true" pos="{x} 0.1 0"/>')
    xml = '<mujoco><option timestep="0.002"/><worldbody>' + ''.join(sides) + '</worldbody></mujoco>'
    models = [mujoco.MjModel.from_xml_string(xml) for _ in range(2)]
    datas = [mujoco.MjData(model) for model in models]
    name_map = {
            side: {"joints": [f"{side}_j{i}" for i in range(6)], "site": f"{side}_tcp", "target": f"{side}_target", "gripper_prefix": f"{side}_gripper_"}
        for side in ("left", "right")
    }
    external = _external_class()(models[0], datas[0], name_map=name_map, robot_kind="doosan")
    native = NativeDofDualArmMPCPVT(models[1], datas[1], name_map=name_map, robot_kind="doosan")
    for controller in (external, native):
        controller.initialize_home()
        controller.step()
    np.testing.assert_allclose(datas[1].qpos, datas[0].qpos, atol=1e-12)
    for side in ("left", "right"):
        np.testing.assert_allclose(native.arms[side].dq_des, external.arms[side].dq_des, atol=1e-12)
        assert native.arms[side].selected_candidate_index == external.arms[side].selected_candidate_index
