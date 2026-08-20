import numpy as np
import mujoco

from factory_bimanual.bimanual_collision import (
    CollisionClass,
    CollisionReport,
    CallbackCollisionChecker,
)
from factory_bimanual.mujoco_collision_adapter import MuJoCoPairedCollisionChecker


def test_mount_attachment_filter_is_narrow(monkeypatch):
    """Only same-arm mount/link1-2 attachment overlap is exempt."""
    class Model:
        geom_bodyid = np.asarray((0, 1, 2, 3))
        body_parentid = np.asarray((0, 0, 0, 0))

    names = {
        (mujoco.mjtObj.mjOBJ_GEOM, 0): "left_mount_adapter",
        (mujoco.mjtObj.mjOBJ_GEOM, 1): "left_link1_collision_0",
        (mujoco.mjtObj.mjOBJ_GEOM, 2): "left_link3_collision_0",
        (mujoco.mjtObj.mjOBJ_GEOM, 3): "right_link1_collision_0",
        (mujoco.mjtObj.mjOBJ_BODY, 1): "left_link1",
        (mujoco.mjtObj.mjOBJ_BODY, 2): "left_link3",
        (mujoco.mjtObj.mjOBJ_BODY, 3): "right_link1",
    }
    monkeypatch.setattr(mujoco, "mj_id2name", lambda _model, kind, index: names.get((kind, index)))
    checker = object.__new__(MuJoCoPairedCollisionChecker)
    checker.model = Model()
    assert checker._is_mount_attachment_contact(0, 1)
    assert not checker._is_mount_attachment_contact(0, 2)
    assert not checker._is_mount_attachment_contact(0, 3)


def test_side_collision_rejects_only_the_requested_arm(tmp_path):
    from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
    from factory_bimanual.scene_builder import build_same_model_scene
    xml = tmp_path / "scene.xml"
    contract = ROBOT_CONTRACTS["xarm6"]
    build_same_model_scene(contract, .8, xml)
    model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
    names = {side: {"joints": contract.prefixed_joint_names(side)}
             for side in ("left", "right")}
    checker = MuJoCoPairedCollisionChecker(model, data, names)
    report = checker.side_state("left", np.zeros(6))
    assert CollisionClass.CROSS_ARM not in report.classes


def test_pair_collision_rejects_individually_valid_branches():
    def evaluate(left, right):
        return CollisionReport(
            classes=(CollisionClass.CROSS_ARM,) if left[0] == right[0] else ()
        )

    checker = CallbackCollisionChecker(pair_evaluator=evaluate)

    assert checker.state(np.array([0.0]), np.array([1.0])).valid
    report = checker.state(np.array([1.0]), np.array([1.0]))
    assert not report.valid
    assert report.classes == (CollisionClass.CROSS_ARM,)


def test_transition_collision_is_reported_separately():
    checker = CallbackCollisionChecker(
        transition_evaluator=lambda previous, current: CollisionReport(
            classes=(CollisionClass.TABLE,) if current[0][0] > 0.5 else ()
        )
    )

    report = checker.transition(
        (np.array([0.0]), np.array([0.0])),
        (np.array([1.0]), np.array([0.0])),
    )
    assert not report.valid
    assert report.classes == (CollisionClass.TABLE,)
