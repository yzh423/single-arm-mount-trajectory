import mujoco
import numpy as np

from factory_bimanual.bimanual_collision import CollisionClass
from factory_bimanual.mujoco_collision_adapter import MuJoCoPairedCollisionChecker


def _scene():
    xml = '''<mujoco><worldbody>
      <geom name="workbench" type="plane" size="2 2 .1"/>
      <body name="left_base"><joint name="left_j" type="slide" axis="1 0 0"/><geom name="left_link" type="sphere" size=".2" pos="-.6 0 .1"/></body>
      <body name="right_base"><joint name="right_j" type="slide" axis="1 0 0"/><geom name="right_link" type="sphere" size=".2" pos=".6 0 .1"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml); return model, mujoco.MjData(model)


def test_classifies_cross_arm_and_table_contacts_without_mutating_live_state():
    model, data = _scene(); data.qpos[:] = [9, 9]
    checker = MuJoCoPairedCollisionChecker(model, data,
        {"left": {"joints": ["left_j"]}, "right": {"joints": ["right_j"]}})
    report = checker.state([.6], [-.6])
    assert CollisionClass.CROSS_ARM in report.classes
    assert CollisionClass.TABLE in report.classes
    assert np.array_equal(data.qpos, [9, 9])


def test_transition_interpolation_detects_midpoint_collision():
    model, data = _scene()
    checker = MuJoCoPairedCollisionChecker(model, data,
        {"left": {"joints": ["left_j"]}, "right": {"joints": ["right_j"]}},
        transition_steps=7)
    report = checker.transition((np.array([0.]), np.array([0.])),
                                (np.array([1.2]), np.array([-1.2])))
    assert CollisionClass.CROSS_ARM in report.classes


def test_cross_gripper_base_contact_is_not_robot_base_collision():
    xml = '''<mujoco><worldbody>
      <body name="left_gripper_base"><joint name="left_j" type="slide"/>
        <geom name="left_gripper_base_collision_0" type="sphere" size=".1"/></body>
      <body name="right_gripper_base" pos=".1 0 0"><joint name="right_j" type="slide"/>
        <geom name="right_gripper_base_collision_0" type="sphere" size=".1"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    checker = MuJoCoPairedCollisionChecker(
        model, mujoco.MjData(model),
        {"left": {"joints": ["left_j"]},
         "right": {"joints": ["right_j"]}},
    )

    report = checker.state([0.0], [0.0])

    assert CollisionClass.CROSS_ARM in report.classes
    assert CollisionClass.BASE not in report.classes


def test_ignores_only_the_designed_same_arm_base_to_link1_attachment():
    xml = '''<mujoco><worldbody>
      <body name="left_base"><geom name="left_base_collision_0" type="sphere" size=".1"/></body>
      <body name="left_link1"><joint name="left_j" type="slide"/><geom name="left_link1_collision_0" type="sphere" size=".1"/></body>
      <body name="right_base" pos="2 0 0"><geom name="right_base_collision_0" type="sphere" size=".1"/></body>
      <body name="right_link1" pos="2 0 0"><joint name="right_j" type="slide"/><geom name="right_link1_collision_0" type="sphere" size=".1"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    checker = MuJoCoPairedCollisionChecker(
        model, mujoco.MjData(model),
        {"left": {"joints": ["left_j"]}, "right": {"joints": ["right_j"]}},
    )
    assert checker.side_state("left", [0.]).classes == ()


def test_ignores_closed_finger_to_finger_contact_on_the_same_gripper():
    xml = '''<mujoco><worldbody>
      <body name="left_link1" pos="3 0 0"><joint name="left_j" type="slide"/><geom name="left_link1_collision" type="sphere" size=".05"/></body>
      <body name="left_gripper_link1"><joint name="left_finger" type="slide"/><geom name="left_gripper_link1_collision_0" type="sphere" size=".1"/></body>
      <body name="left_gripper_link2"><geom name="left_gripper_link2_collision_0" type="sphere" size=".1"/></body>
      <body name="right_link1" pos="5 0 0"><joint name="right_j" type="slide"/><geom name="right_link1_collision" type="sphere" size=".05"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    checker = MuJoCoPairedCollisionChecker(
        model, mujoco.MjData(model),
        {"left": {"joints": ["left_j"]}, "right": {"joints": ["right_j"]}},
    )
    assert checker.side_state("left", [0.]).classes == ()


def test_ignores_numerical_mesh_touch_but_keeps_material_self_penetration():
    xml = '''<mujoco><worldbody>
      <body name="left_link4"><joint name="left_j" type="slide" axis="1 0 0"/>
        <geom name="left_link4_collision_0" type="sphere" size=".1"/></body>
      <body name="left_link6" pos=".19995 0 0">
        <geom name="left_link6_collision_0" type="sphere" size=".1"/></body>
      <body name="right_link1" pos="2 0 0"><joint name="right_j" type="slide"/>
        <geom name="right_link1_collision_0" type="sphere" size=".05"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    checker = MuJoCoPairedCollisionChecker(
        model, mujoco.MjData(model),
        {"left": {"joints": ["left_j"]},
         "right": {"joints": ["right_j"]}},
    )

    assert checker.side_state("left", [0.]).classes == ()
    assert CollisionClass.SELF in checker.side_state(
        "left", [.002]).classes
