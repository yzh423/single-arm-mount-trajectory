"""Play factory bimanual recordings in GoodGoodArmDayDayUp's four-cell scene."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .source_data import FactoryBimanualTask, load_factory_task


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = WORKSPACE_ROOT / "GoodGoodArmDayDayUp-main"
REFERENCE_SCENE = REFERENCE_ROOT / "models/four_robot_dual_arm_benchmark_scene.xml"
DEFAULT_TASKS = (
    WORKSPACE_ROOT / "single-arm-mount/data/factory/8-11/Screw_Cap/handheld_20260811_162854.csv",
    WORKSPACE_ROOT / "single-arm-mount/data/factory/8-12/PourRawMaterial/handheld_20260812_111542.csv",
)


def assert_reference_demo_only() -> None:
    raise RuntimeError(
        "GoodGoodArmDayDayUp-main's combined scene contains Willow, xArm6, UR5, "
        "and Doosan; it is not the requested xArm6, Franka Panda, I2RT, and UR5 "
        "experiment. Direct factory-CSV retargeting is disabled because it can "
        "display unreachable, colliding false solutions."
    )


def load_reference_model() -> mujoco.MjModel:
    """Load the original scene with a MuJoCo-3.3 compatibility shim in memory."""
    text = lightweight_scene_text()
    meshdir = str(REFERENCE_SCENE.parent).replace("\\", "/")
    text = text.replace('meshdir="."', f'meshdir="{meshdir}"', 1)
    for name in ("willow_wrist_camera_body", "willow_wrist_camera_body_left"):
        marker = f'<body name="{name}"'
        start = text.index(marker)
        close = text.index(">", start) + 1
        text = text[:close] + '<inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>' + text[close:]
    return mujoco.MjModel.from_xml_string(text)


def lightweight_scene_text() -> str:
    """Keep the reference geometry but avoid oversized realtime-view buffers."""
    text = REFERENCE_SCENE.read_text(encoding="utf-8")
    text = text.replace('njmax="12000" nconmax="5000"', 'njmax="3000" nconmax="1200"')
    text = text.replace('shadowsize="4096"', 'shadowsize="1024"')
    text = text.replace('offsamples="8"', 'offsamples="2"')
    tables = {
        "willow": (-1.35, 0.81), "xarm6": (1.35, 0.81),
        "ur5": (-1.35, -1.49), "doosan": (1.35, -1.49),
    }
    legs = []
    for name, (cx, cy) in tables.items():
        for index, (dx, dy) in enumerate(((-.58, -.42), (-.58, .42), (.58, -.42), (.58, .42)), 1):
            legs.append(
                f'<geom name="{name}_table_leg_{index}" type="box" '
                f'pos="{cx+dx} {cy+dy} 0.1675" size="0.035 0.035 0.1675" '
                f'rgba="0.22 0.24 0.27 1" contype="1" conaffinity="1"/>'
            )
    text = text.replace('<geom name="benchmark_floor_geom"', ''.join(legs) + '<geom name="benchmark_floor_geom"', 1)
    return text


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


_MAP_QUAT = np.array([math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)])
_MAP_ROTATION = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def mapped_relative_pose(position, quaternion, source_start_position,
                         source_start_quaternion, cell_start_position,
                         cell_start_quaternion):
    """Apply the reference scene's table-frame mapping without scaling/cropping."""
    delta_p = _MAP_ROTATION @ (np.asarray(position) - np.asarray(source_start_position))
    delta_q = _quat_mul(np.asarray(quaternion), _quat_conjugate(np.asarray(source_start_quaternion)))
    mapped_q = _quat_mul(_quat_mul(_MAP_QUAT, delta_q), _quat_conjugate(_MAP_QUAT))
    out_q = _quat_mul(mapped_q, np.asarray(cell_start_quaternion))
    out_q /= max(float(np.linalg.norm(out_q)), 1e-12)
    return np.asarray(cell_start_position) + delta_p, out_q


def _sample(task: FactoryBimanualTask, elapsed: float, side: str):
    t = task.time_s - task.time_s[0]
    hi = min(int(np.searchsorted(t, elapsed, side="right")), len(t) - 1)
    lo = max(0, hi - 1)
    alpha = np.clip((elapsed - t[lo]) / max(float(t[hi] - t[lo]), 1e-12), 0.0, 1.0)
    positions = getattr(task, f"{side}_position_m")
    quats = getattr(task, f"{side}_quaternion_wxyz")
    p = (1 - alpha) * positions[lo] + alpha * positions[hi]
    q1 = quats[hi] if float(quats[lo] @ quats[hi]) >= 0 else -quats[hi]
    q = (1 - alpha) * quats[lo] + alpha * q1
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return p, q


def _four_cell_names(robot: str, side: str):
    if robot == "willow":
        suffix = "_left" if side == "left" else ""
        return ([f"willow_joint{i}{suffix}" for i in range(1, 7)],
                f"willow_ee_site{suffix}", f"willow_mocap_target{suffix}")
    return ([f"{robot}_{side}_joint_{i}" for i in range(1, 7)],
            f"{robot}_{side}_tcp", f"{robot}_{side}_target")


class LightweightFourCellIK:
    """Memory-bounded DLS viewer controller; one model/data pair, no scene copies."""
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data, self.arms = model, data, {}
        for robot in ("willow", "xarm6", "ur5", "doosan"):
            for side in ("left", "right"):
                joints, site_name, target_name = _four_cell_names(robot, side)
                jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joints])
                qids = model.jnt_qposadr[jids]
                dids = model.jnt_dofadr[jids]
                site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
                mocap = int(model.body_mocapid[body])
                self.arms[(robot, side)] = (qids, dids, site, mocap)
        homes = {
            "willow": [0, 65, 70, 0, 0, 0],
            "xarm6": [0, -30, -60, 0, 90, 0],
            "ur5": [-124.72047312, -89.99996451, -90.00003534, -90.00000018, 89.99998702, 25.27953967],
            "doosan_left": [106.4113, -14.5932, -54.1046, -179.9914, 111.3136, 151.4240],
            "doosan_right": [73.6126, 14.5924, 54.0834, -180.0095, -111.3126, 28.5758],
        }
        for (robot, side), (qids, _, _, _) in self.arms.items():
            key = f"{robot}_{side}" if robot == "doosan" else robot
            data.qpos[qids] = np.deg2rad(homes[key])
        mujoco.mj_forward(model, data)
        self.table_geom_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{robot}_table")
            for robot in ("willow", "xarm6", "ur5", "doosan")
        }
        self.last_table_rollback = False
        for _, (_, _, site, mocap) in self.arms.items():
            data.mocap_pos[mocap] = data.site_xpos[site]
            mujoco.mju_mat2Quat(data.mocap_quat[mocap], data.site_xmat[site])

    def step(self):
        safe_qpos = self.data.qpos.copy()
        for qids, dids, site, mocap in self.arms.values():
            jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site)
            current_q = np.zeros(4); mujoco.mju_mat2Quat(current_q, self.data.site_xmat[site])
            rot = np.zeros(3); mujoco.mju_subQuat(rot, self.data.mocap_quat[mocap], current_q)
            error = np.r_[self.data.mocap_pos[mocap] - self.data.site_xpos[site], rot]
            jac = np.vstack((jacp[:, dids], jacr[:, dids]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.08 * np.eye(6), error)
            norm = float(np.linalg.norm(dq))
            if norm > 0.12: dq *= 0.12 / norm
            lo, hi = self.model.jnt_range[self.model.dof_jntid[dids]].T
            self.data.qpos[qids] = np.clip(self.data.qpos[qids] + 0.18 * dq, lo, hi)
            self.data.qvel[dids] = 0
            mujoco.mj_forward(self.model, self.data)
        self.last_table_rollback = any(
            contact.dist < 0 and
            (contact.geom1 in self.table_geom_ids or contact.geom2 in self.table_geom_ids)
            for contact in self.data.contact
        )
        if self.last_table_rollback:
            self.data.qpos[:] = safe_qpos
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)


def main() -> None:
    assert_reference_demo_only()
    parser = argparse.ArgumentParser(description="Four original MuJoCo cells playing two factory tasks.")
    parser.add_argument("--task", type=Path, action="append", dest="tasks")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0:
        raise ValueError("speed must be positive")
    paths = tuple(args.tasks or DEFAULT_TASKS)
    tasks = [load_factory_task(path, Path(path).parent.name) for path in paths]
    if not REFERENCE_SCENE.is_file():
        raise FileNotFoundError(REFERENCE_SCENE)

    sys.path.insert(0, str(REFERENCE_ROOT))
    model = load_reference_model()
    data = mujoco.MjData(model)
    controller = LightweightFourCellIK(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.55]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 5.0, 135, -32
        while viewer.is_running():
            for task in tasks:
                starts = {}
                for (robot, side), (_, _, site, _) in controller.arms.items():
                    q = np.zeros(4); mujoco.mju_mat2Quat(q, data.site_xmat[site])
                    starts[(robot, side)] = (data.site_xpos[site].copy(), q)
                started = time.perf_counter()
                duration = float(task.time_s[-1] - task.time_s[0])
                print(f"[factory-four-cell] task={task.name} rows={len(task.time_s)} duration={duration:.3f}s")
                while viewer.is_running():
                    elapsed = (time.perf_counter() - started) * args.speed
                    if elapsed > duration:
                        break
                    for (robot, side), (_, _, _, mocap) in controller.arms.items():
                            p, q = _sample(task, elapsed, side)
                            wp, wq = mapped_relative_pose(
                                p, q, getattr(task, f"{side}_position_m")[0],
                                getattr(task, f"{side}_quaternion_wxyz")[0], *starts[(robot, side)])
                            data.mocap_pos[mocap] = wp
                            data.mocap_quat[mocap] = wq
                    controller.step()
                    viewer.sync()
                    time.sleep(max(0.0, model.opt.timestep / args.speed))
                if not viewer.is_running():
                    return
            if not args.loop:
                while viewer.is_running():
                    mujoco.mj_forward(model, data); viewer.sync(); time.sleep(1 / 60)


if __name__ == "__main__":
    main()
