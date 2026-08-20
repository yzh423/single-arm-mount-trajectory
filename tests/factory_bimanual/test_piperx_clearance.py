import mujoco
import numpy as np

from factory_bimanual.mujoco_collision_adapter import (
    MuJoCoPairedCollisionChecker,
)
from factory_bimanual.bimanual_collision import CollisionClass


def _checker(*, clearance_margin_m=None):
    xml = """
    <mujoco><worldbody>
      <body name="left_arm">
        <joint name="left_joint1" type="slide" axis="1 0 0" range="-1 1"/>
        <geom name="left_base_link_collision_0" type="sphere" pos="-1 0 0" size=".01"/>
        <geom name="left_link3_collision_0" type="sphere" pos="-.8 0 0" size=".01"/>
        <geom name="left_link4_collision_0" type="sphere" pos="0 0 0" size=".01"/>
        <geom name="left_link5_collision_0" type="sphere" pos=".4 0 0" size=".01"/>
        <geom name="left_link6_collision_0" type="sphere" pos=".5 0 0" size=".01"/>
        <geom name="left_gripper_base_collision_0" type="sphere" pos=".1 0 0" size=".01"/>
        <geom name="left_gripper_link1_collision_0" type="sphere" pos=".2 0 0" size=".01"/>
        <geom name="left_gripper_link2_collision_0" type="sphere" pos=".25 0 0" size=".01"/>
      </body>
      <body name="right_arm">
        <joint name="right_joint1" type="slide" axis="1 0 0" range="-1 1"/>
        <geom name="right_base_link_collision_0" type="sphere" pos="1 0 0" size=".01"/>
        <geom name="right_link3_collision_0" type="sphere" pos=".8 0 0" size=".01"/>
        <geom name="right_link4_collision_0" type="sphere" pos=".6 0 0" size=".01"/>
        <geom name="right_link5_collision_0" type="sphere" pos=".5 0 0" size=".01"/>
        <geom name="right_link6_collision_0" type="sphere" pos=".4 0 0" size=".01"/>
        <geom name="right_gripper_base_collision_0" type="sphere" pos=".05 0 0" size=".01"/>
        <geom name="right_gripper_link1_collision_0" type="sphere" pos=".3 0 0" size=".01"/>
        <geom name="right_gripper_link2_collision_0" type="sphere" pos=".35 0 0" size=".01"/>
      </body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    names = {
        "left": {"joints": ("left_joint1",)},
        "right": {"joints": ("right_joint1",)},
    }
    return MuJoCoPairedCollisionChecker(
        model, mujoco.MjData(model), names, transition_steps=5,
        clearance_margin_m=clearance_margin_m)


def test_piperx_clearance_reports_named_pairs_and_hard_margin():
    checker = _checker()
    report = checker.clearance(np.zeros(1), np.zeros(1), margin_m=.015)

    assert set(report.pair_distances_m) == {
        "left_link4__right_gripper_base",
        "right_link4__left_gripper_base",
        "gripper_base__gripper_base",
        "fingers__opposite_wrist",
        "elbow__opposite_base",
    }
    assert report.minimum_m == min(report.pair_distances_m.values())
    assert report.pair_distances_m["elbow__opposite_base"] > 1.0
    assert report.minimum_m >= .015
    assert report.valid
    assert not checker.clearance(
        np.zeros(1), np.zeros(1), margin_m=.04).valid


def test_swept_clearance_uses_same_margin_at_intermediate_states():
    checker = _checker()
    report = checker.transition_clearance(
        (np.zeros(1), np.zeros(1)),
        (np.zeros(1), np.asarray([-.04])),
        margin_m=.015,
    )

    assert not report.valid
    assert report.minimum_m < .015


def test_configured_clearance_margin_is_a_hard_state_and_edge_constraint():
    checker = _checker(clearance_margin_m=.04)

    assert CollisionClass.CLEARANCE in checker.state(
        np.zeros(1), np.zeros(1)).classes
    assert CollisionClass.CLEARANCE in checker.transition(
        (np.zeros(1), np.zeros(1)),
        (np.zeros(1), np.asarray([-.04])),
    ).classes
