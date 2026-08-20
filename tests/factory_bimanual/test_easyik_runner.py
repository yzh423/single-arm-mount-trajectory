from types import SimpleNamespace

import mujoco
import numpy as np

from factory_bimanual.easyik_runner import EasyIKBudget, EasyIKScene, run_easyik_task
from factory_bimanual.bimanual_collision import CollisionClass, CollisionReport
from factory_bimanual.robot_contracts import BimanualRobotContract


def _fixture():
    xml = """<mujoco><worldbody>
      <body name="left_target" mocap="true"/><body name="right_target" mocap="true"/>
      <body><joint name="left_j" type="slide" axis="1 0 0" range="-2 2"/><geom type="sphere" size=".01" mass=".01"/><site name="left_tcp"/></body>
      <body><joint name="right_j" type="slide" axis="1 0 0" range="-2 2"/><geom type="sphere" size=".01" mass=".01"/><site name="right_tcp"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    contract = BimanualRobotContract("synthetic", __file__, ("j",), "tcp", "base")
    name_map = {side: {"joints": [f"{side}_j"], "site": f"{side}_tcp", "target": f"{side}_target"}
                for side in ("left", "right")}
    scene = EasyIKScene(model, data, contract, name_map)
    task = SimpleNamespace(
        time_s=np.array([0.0, 0.1]), source_row_indices=np.array([17, 23]),
        left_position_m=np.array([[0., 0., 0.], [.2, 0., 0.]]),
        right_position_m=np.array([[0., 0., 0.], [-.2, 0., 0.]]),
        left_quaternion_wxyz=np.tile([1., 0., 0., 0.], (2, 1)),
        right_quaternion_wxyz=np.tile([1., 0., 0., 0.], (2, 1)),
    )
    return scene, task


def test_runner_preserves_rows_and_scores_actual_mujoco_tcp(monkeypatch):
    scene, task = _fixture()
    calls = []

    class Controller:
        def __init__(self):
            self.arms = {
                side: SimpleNamespace(
                    mocap_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_target") - 1,
                    site_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp"),
                    qpos_ids=np.array([mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_j")]),
                ) for side in ("left", "right")
            }

        def step(self, enabled_sides=("left", "right")):
            calls.append(tuple(enabled_sides))
            for side in enabled_sides:
                arm = self.arms[side]
                scene.data.qpos[arm.qpos_ids] = scene.data.mocap_pos[arm.mocap_id, 0]
            mujoco.mj_forward(scene.model, scene.data)

    monkeypatch.setattr("factory_bimanual.easyik_runner.make_easyik", lambda *a, **k: Controller())
    result = run_easyik_task(scene, task, EasyIKBudget(iterations_per_target=3))

    assert result.source_row_indices.tolist() == [17, 23]
    assert result.left_q.shape == result.right_q.shape == (2, 1)
    assert np.allclose(result.left_actual_tcp[:, 0], [0, .2])
    assert np.allclose(result.right_actual_tcp[:, 0], [0, -.2])
    assert result.success.tolist() == [True, True]
    assert calls == [("left", "right"), ("left", "right")]
    assert result.declared_iterations_per_target == 3
    assert result.iterations_used.tolist() == [1, 1]


def test_runner_marks_convergence_exhausted_without_dropping_frame(monkeypatch):
    scene, task = _fixture()

    class Stuck:
        def __init__(self):
            self.arms = {
                side: SimpleNamespace(
                    mocap_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_target") - 1,
                    site_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp"),
                    qpos_ids=np.array([mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_j")]),
                ) for side in ("left", "right")
            }
        def step(self, enabled_sides=("left", "right")):
            pass

    monkeypatch.setattr("factory_bimanual.easyik_runner.make_easyik", lambda *a, **k: Stuck())
    result = run_easyik_task(scene, task, EasyIKBudget(iterations_per_target=2))

    assert len(result.success) == len(task.time_s)
    assert result.success.tolist() == [True, False]
    assert result.failures == [None, "easyik_convergence_exhausted"]
    assert result.iterations_used.tolist() == [1, 2]
    assert np.isclose(result.left_position_error_m[1], .2)
    assert result.paired_branch_indices == [(0, 0), None]


def test_external_collision_audit_can_reject_converged_tcp(monkeypatch):
    scene, task = _fixture()
    checker = SimpleNamespace(state=lambda left, right: CollisionReport((CollisionClass.CROSS_ARM,)))
    scene = EasyIKScene(scene.model, scene.data, scene.contract, scene.name_map,
                        collision_checker=checker)

    class Controller:
        def __init__(self):
            self.arms = {
                side: SimpleNamespace(
                    mocap_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_target") - 1,
                    site_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp"),
                    qpos_ids=np.array([mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_j")]),
                ) for side in ("left", "right")
            }
        def step(self, enabled_sides=("left", "right")):
            for side in enabled_sides:
                arm = self.arms[side]
                scene.data.qpos[arm.qpos_ids] = scene.data.mocap_pos[arm.mocap_id, 0]

    monkeypatch.setattr("factory_bimanual.easyik_runner.make_easyik", lambda *a, **k: Controller())
    result = run_easyik_task(scene, task, EasyIKBudget(1))

    assert result.success.tolist() == [False, False]
    assert result.failures == ["cross_arm_collision", "cross_arm_collision"]
    assert result.collision_classes == [("cross_arm_collision",), ("cross_arm_collision",)]


def test_runner_accepts_loader_source_row_index_field(monkeypatch):
    scene, task = _fixture()
    task.source_row_index = task.source_row_indices
    del task.source_row_indices

    class Stuck:
        def __init__(self):
            self.arms = {
                side: SimpleNamespace(
                    mocap_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_target") - 1,
                    site_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp"),
                    qpos_ids=np.array([mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_j")]),
                ) for side in ("left", "right")
            }
        def step(self, enabled_sides=("left", "right")):
            pass

    monkeypatch.setattr("factory_bimanual.easyik_runner.make_easyik", lambda *a, **k: Stuck())
    result = run_easyik_task(scene, task, EasyIKBudget(1))
    assert result.source_row_indices.tolist() == [17, 23]
