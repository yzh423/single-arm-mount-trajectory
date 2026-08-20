from types import SimpleNamespace

import mujoco
import numpy as np

from factory_bimanual.branch_equivalence import BranchDecision
from factory_bimanual.mpc_runner import MPCScene, scheduled_mpc_runs, run_mpc_task
from factory_bimanual.robot_contracts import BimanualRobotContract


def _fixture():
    xml = """<mujoco><option timestep=".05"/><worldbody>
    <body name="left_target" mocap="true"/><body name="right_target" mocap="true"/>
    <body><joint name="left_j" type="slide" axis="1 0 0" range="-2 2"/><geom type="sphere" size=".01" mass=".01"/><site name="left_tcp"/></body>
    <body><joint name="right_j" type="slide" axis="1 0 0" range="-2 2"/><geom type="sphere" size=".01" mass=".01"/><site name="right_tcp"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    contract = BimanualRobotContract("synthetic", __file__, ("j",), "tcp", "base")
    names = {s: {"joints": [f"{s}_j"], "site": f"{s}_tcp", "target": f"{s}_target"} for s in ("left", "right")}
    task = SimpleNamespace(time_s=np.array([2., 2.1, 2.2]), source_row_index=np.array([5, 8, 13]),
        left_position_m=np.array([[0.,0,0],[.1,0,0],[.2,0,0]]), right_position_m=np.array([[0.,0,0],[-.1,0,0],[-.2,0,0]]),
        left_quaternion_wxyz=np.tile([1.,0,0,0], (3,1)), right_quaternion_wxyz=np.tile([1.,0,0,0], (3,1)))
    return MPCScene(model, data, contract, names), task


def test_schedule_uses_decision_modes_and_correct_initializers():
    a = (np.array([1.]), np.array([2.])); easy = (np.array([3.]), np.array([4.]))
    split = BranchDecision(.2, .2, False, ("mpc_a", "mpc_easyik"), None)
    folded = BranchDecision(.01, .01, True, ("canonical_a",), "a")
    assert scheduled_mpc_runs(split, a, easy) == (("mpc_a", a), ("mpc_easyik", easy))
    assert scheduled_mpc_runs(folded, a, easy) == (("canonical_a", a),)


def test_source_timing_interpolation_and_row_aligned_external_metrics(monkeypatch):
    scene, task = _fixture(); targets = []
    class Controller:
        def __init__(self):
            self.arms = {s: SimpleNamespace(
                mocap_id=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_BODY,f"{s}_target")-1,
                site_id=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_SITE,f"{s}_tcp"),
                qpos_ids=np.array([mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_JOINT,f"{s}_j")]),
                dof_ids=np.array([mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_JOINT,f"{s}_j")]),
                safety_rollback_last_step=False) for s in ("left","right")}
        def step(self, enabled_sides=("left","right")):
            targets.append(tuple(scene.data.mocap_pos[a.mocap_id,0] for a in self.arms.values()))
            for a in self.arms.values(): scene.data.qpos[a.qpos_ids] = scene.data.mocap_pos[a.mocap_id,0]
    monkeypatch.setattr("factory_bimanual.mpc_runner.make_mpc", lambda *a, **k: Controller())
    result = run_mpc_task(scene, task, (np.array([.02]), np.array([-.02])), None, "mpc_a")
    assert result.source_row_indices.tolist() == [5,8,13]
    assert result.control_steps_per_interval.tolist() == [0,2,2]
    assert np.allclose(targets, [(.05,-.05),(.1,-.1),(.15,-.15),(.2,-.2)])
    assert np.allclose(result.left_actual_tcp[:,0], [.02,.1,.2])
    assert np.allclose(result.left_target_lag_m, [.02,0,0])
    assert result.initial_left_q.tolist() == [.02]
    assert len(result.step_wall_time_s) == 3


def test_records_rollback_and_collision_per_source_row(monkeypatch):
    scene, task = _fixture()
    checker = SimpleNamespace(state=lambda l,r: SimpleNamespace(valid=False, classes=(SimpleNamespace(value="cross_arm_collision"),)))
    scene = MPCScene(scene.model, scene.data, scene.contract, scene.name_map, collision_checker=checker)
    class Controller:
        def __init__(self):
            self.arms = {s: SimpleNamespace(mocap_id=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_BODY,f"{s}_target")-1,
                site_id=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_SITE,f"{s}_tcp"), qpos_ids=np.array([i]), dof_ids=np.array([i]), safety_rollback_last_step=False)
                for i,s in enumerate(("left","right"))}
        def step(self, enabled_sides=("left","right")):
            for a in self.arms.values(): a.safety_rollback_last_step=True
    monkeypatch.setattr("factory_bimanual.mpc_runner.make_mpc", lambda *a, **k: Controller())
    result = run_mpc_task(scene, task, (np.array([0.]),np.array([0.])), None, "mpc_a")
    assert result.rollback.tolist() == [False,True,True]
    assert result.collision_classes == [("cross_arm_collision",)] * 3


def test_non_integer_control_interval_uses_exact_source_duration(monkeypatch):
    scene, task = _fixture()
    task.time_s = np.array([0.0, 0.12])
    task.source_row_index = np.array([0, 1])
    for side in ("left", "right"):
        setattr(task, f"{side}_position_m", np.array([[0., 0., 0.], [.12, 0., 0.]]))
        setattr(task, f"{side}_quaternion_wxyz", np.tile([1., 0., 0., 0.], (2, 1)))

    class Controller:
        def __init__(self):
            self.dt = float(scene.model.opt.timestep)
            self.elapsed = 0.0
            self.arms = {s: SimpleNamespace(
                mocap_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_target") - 1,
                site_id=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_tcp"),
                qpos_ids=np.array([i]), dof_ids=np.array([i]),
                safety_rollback_last_step=False,
            ) for i, s in enumerate(("left", "right"))}

        def step(self, enabled_sides=("left", "right")):
            self.elapsed += self.dt
            for arm in self.arms.values():
                scene.data.qpos[arm.qpos_ids] = self.elapsed

    controller = Controller()
    monkeypatch.setattr("factory_bimanual.mpc_runner.make_mpc", lambda *a, **k: controller)
    result = run_mpc_task(scene, task, (np.array([0.]), np.array([0.])), None, "mpc_a")
    assert result.control_steps_per_interval.tolist() == [0, 3]
    assert np.isclose(controller.elapsed, 0.12)
    assert np.isclose(result.left_q[-1, 0], 0.12)
