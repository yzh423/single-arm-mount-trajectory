from __future__ import annotations

import mujoco
import numpy as np
import pytest

from factory_bimanual.controller_adapter import (
    UnsupportedModelControllerContract,
    make_easyik,
    make_mpc,
    set_bimanual_targets,
)
from factory_bimanual.native_dof_mpc import MPCConfig
from factory_bimanual.robot_contracts import BimanualRobotContract


def _model(dof: int = 7):
    bodies = []
    for side, x in (("left", -0.3), ("right", 0.3)):
        nested = ""
        for i in reversed(range(dof)):
            site = f'<site name="{side}_tcp"/>' if i == dof - 1 else ""
            nested = (
                f'<body name="{side}_b{i}" pos="0 0 .04">'
                f'<joint name="{side}_j{i}" type="hinge" range="-2 2"/>'
                f'<geom type="capsule" size=".005 .02" mass=".01"/>{site}{nested}</body>'
            )
        bodies.append(f'<body name="{side}_base" pos="{x} 0 0">{nested}</body>')
        bodies.append(
            f'<body name="{side}_target" mocap="true" pos="{x} 0 .3">'
            f'<geom type="sphere" size=".005" contype="0" conaffinity="0"/></body>'
        )
    xml = (
        '<mujoco><option timestep=".002"/><worldbody>' + "".join(bodies) +
        '</worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def _contract(dof: int = 7):
    return BimanualRobotContract(
        "synthetic", __file__, tuple(f"j{i}" for i in range(dof)), "tcp", "base"
    )


def _name_map(dof: int = 7):
    return {
        side: {
            "joints": [f"{side}_j{i}" for i in range(dof)],
            "site": f"{side}_tcp",
            "target": f"{side}_target",
        }
        for side in ("left", "right")
    }


def test_native_dof_controllers_allocate_seven_joint_state():
    model, data = _model(7)
    config = MPCConfig(collision_detection_enabled=False, velocity_scale=0.5)
    easy = make_easyik(model, data, _contract(7), config, name_map=_name_map(7))
    mpc = make_mpc(model, data, _contract(7), config, name_map=_name_map(7))

    assert set(easy.arms) == {"left", "right"}
    for controller in (easy, mpc):
        assert controller.config is config
        assert all(arm.qpos_ids.shape == (7,) for arm in controller.arms.values())
        assert all(arm.dq_des.shape == (7,) for arm in controller.arms.values())
        assert all(arm.dq_limit.shape == (7,) for arm in controller.arms.values())


def test_set_bimanual_targets_writes_both_mocap_poses():
    model, data = _model(7)
    controller = make_easyik(
        model, data, _contract(7), MPCConfig(collision_detection_enabled=False),
        name_map=_name_map(7),
    )
    left = (np.array([1.0, 2.0, 3.0]), np.array([2.0, 0.0, 0.0, 0.0]))
    right = (np.array([-1.0, -2.0, -3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    set_bimanual_targets(data, controller, left, right)

    assert np.allclose(data.mocap_pos[controller.arms["left"].mocap_id], left[0])
    assert np.allclose(data.mocap_quat[controller.arms["left"].mocap_id], [1, 0, 0, 0])
    assert np.allclose(data.mocap_pos[controller.arms["right"].mocap_id], right[0])
    assert np.allclose(data.mocap_quat[controller.arms["right"].mocap_id], right[1])


def test_adapter_rejects_contract_dof_that_disagrees_with_scene():
    model, data = _model(6)
    with pytest.raises(UnsupportedModelControllerContract, match="expected 7.*found 6"):
        make_mpc(model, data, _contract(7), name_map=_name_map(6))
