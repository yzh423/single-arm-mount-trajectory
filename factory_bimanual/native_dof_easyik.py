"""Native-DOF port of the existing ``doosan_teleop.easy_ik`` controller."""

from __future__ import annotations

from typing import Iterable
import mujoco
import numpy as np

from .native_dof_mpc import NativeDofDualArmMPCPVT, solve_small_system


class NativeDofDualArmEasyIKPVT(NativeDofDualArmMPCPVT):
    """Existing continuous DLS EasyIK equations with native joint dimension."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_iters, self.alpha, self.dls = 6, 0.18, 8e-2
        self.dq_joint_limit, self.dq_norm_limit, self.beta = 0.06, 0.12, 0.35
        self.position_stop = self.rotation_stop = 0.001
        self._dq_filtered = {side: np.zeros(arm.joint_ids.size) for side, arm in self.arms.items()}

    def initialize_home(self):
        super().initialize_home()
        for value in self._dq_filtered.values():
            value[:] = 0.0

    def step_arm(self, arm):
        final_error, applied_norm, iterations = np.zeros(6), 0.0, 0
        for iterations in range(1, self.sub_iters + 1):
            target_position = self.data.mocap_pos[arm.mocap_id]
            target_quaternion = self.data.mocap_quat[arm.mocap_id]
            error, jacobian, _ = self._pose_error_and_jacobian(
                self.data, arm, target_position, target_quaternion
            )
            final_error = error
            if np.linalg.norm(error[:3]) < self.position_stop and np.linalg.norm(error[3:]) < self.rotation_stop:
                break
            intermediate = solve_small_system(jacobian @ jacobian.T + self.dls * np.eye(6), error)
            increment = np.clip(jacobian.T @ intermediate, -self.dq_joint_limit, self.dq_joint_limit)
            norm = float(np.linalg.norm(increment))
            if norm > self.dq_norm_limit:
                increment *= self.dq_norm_limit / (norm + 1e-12)
            filtered = self._dq_filtered[arm.side]
            filtered[:] = (1 - self.beta) * filtered + self.beta * increment
            applied = self.alpha * filtered
            self.data.qpos[arm.qpos_ids] = np.clip(self.data.qpos[arm.qpos_ids] + applied, arm.q_min, arm.q_max)
            self.data.qvel[arm.dof_ids] = 0.0
            applied_norm = float(np.linalg.norm(applied))
            mujoco.mj_forward(self.model, self.data)
        arm.dq_prev[:] = arm.dq_des[:] = 0.0
        arm.last_debug = {"controller": "easyik_pvt", "pos_err": float(np.linalg.norm(final_error[:3])),
                          "rot_err": float(np.linalg.norm(final_error[3:])), "damping": self.dls,
                          "dq_norm": applied_norm, "iterations": float(iterations)}

    def step(self, enabled_sides: Iterable[str] = ("left", "right")):
        for side in enabled_sides:
            self.step_arm(self.arms[side])
