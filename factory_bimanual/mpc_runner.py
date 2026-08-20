"""Source-time-preserving execution and external audit of bimanual MPC."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import mujoco
import numpy as np

from .controller_adapter import make_mpc, set_bimanual_targets


@dataclass(frozen=True)
class MPCScene:
    model: Any
    data: Any
    contract: Any
    name_map: Mapping[str, Mapping[str, Any]]
    velocity_limits: Any = None
    collision_checker: Any = None


@dataclass
class BimanualMPCResult:
    mode: str
    initial_left_q: np.ndarray
    initial_right_q: np.ndarray
    left_q: np.ndarray
    right_q: np.ndarray
    left_actual_tcp: np.ndarray
    right_actual_tcp: np.ndarray
    left_target_lag_m: np.ndarray
    right_target_lag_m: np.ndarray
    rollback: np.ndarray
    collision_classes: list[tuple[str, ...]]
    control_steps_per_interval: np.ndarray
    step_wall_time_s: np.ndarray
    source_row_indices: np.ndarray


def scheduled_mpc_runs(decision, a_initial, easy_initial):
    """Map the approved branch gate to initial states without changing targets."""
    initializers = {"mpc_a": a_initial, "canonical_a": a_initial, "mpc_easyik": easy_initial}
    return tuple((mode, initializers[mode]) for mode in decision.required_mpc_modes)


def _quat_interp(a, b, fraction):
    b = np.asarray(b, float)
    a = np.asarray(a, float)
    if np.dot(a, b) < 0:
        b = -b
    value = (1.0 - fraction) * a + fraction * b
    return value / np.linalg.norm(value)


def _pose(data, site_id):
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    return np.r_[data.site_xpos[site_id].copy(), quaternion]


def run_mpc_task(scene: MPCScene, task: Any, initial_state, config, mode: str) -> BimanualMPCResult:
    """Follow identical registered targets at model control rate and audit source rows."""
    if mode not in ("mpc_a", "mpc_easyik", "canonical_a"):
        raise ValueError(f"unsupported MPC mode: {mode}")
    model, data = scene.model, scene.data
    controller = make_mpc(model, data, scene.contract, config, name_map=scene.name_map,
                          velocity_limits=scene.velocity_limits)
    initial = tuple(np.asarray(value, float).copy() for value in initial_state)
    for side, value in zip(("left", "right"), initial):
        arm = controller.arms[side]
        if value.shape != arm.qpos_ids.shape:
            raise ValueError(f"invalid {side} initial state")
        data.qpos[arm.qpos_ids] = value
        data.qvel[arm.dof_ids] = 0.0
    mujoco.mj_forward(model, data)

    count, dof = len(task.time_s), scene.contract.dof_per_arm
    q = {s: np.full((count, dof), np.nan) for s in ("left", "right")}
    tcp = {s: np.full((count, 7), np.nan) for s in ("left", "right")}
    lag = {s: np.full(count, np.nan) for s in ("left", "right")}
    rollback = np.zeros(count, bool); steps = np.zeros(count, int); wall = np.zeros(count)
    collisions: list[tuple[str, ...]] = [()] * count
    positions = {"left": np.asarray(task.left_position_m), "right": np.asarray(task.right_position_m)}
    quats = {"left": np.asarray(task.left_quaternion_wxyz), "right": np.asarray(task.right_quaternion_wxyz)}

    def audit(row):
        for side in ("left", "right"):
            arm = controller.arms[side]
            q[side][row] = data.qpos[arm.qpos_ids]
            tcp[side][row] = _pose(data, arm.site_id)
            lag[side][row] = np.linalg.norm(tcp[side][row, :3] - positions[side][row])
        if scene.collision_checker is not None:
            report = scene.collision_checker.state(q["left"][row], q["right"][row])
            collisions[row] = tuple(item.value for item in report.classes)

    audit(0)
    dt = float(model.opt.timestep)
    for row in range(1, count):
        interval = float(task.time_s[row] - task.time_s[row - 1])
        if interval <= 0:
            raise ValueError("source time must be strictly increasing")
        steps[row] = max(1, int(np.ceil(interval / dt - 1e-12)))
        started = perf_counter()
        elapsed = 0.0
        for step in range(1, steps[row] + 1):
            step_dt = min(dt, interval - elapsed)
            elapsed += step_dt
            fraction = min(1.0, elapsed / interval)
            poses = []
            for side in ("left", "right"):
                position = positions[side][row - 1] + fraction * (positions[side][row] - positions[side][row - 1])
                poses.append((position, _quat_interp(quats[side][row - 1], quats[side][row], fraction)))
            set_bimanual_targets(data, controller, poses[0], poses[1])
            original_model_dt = float(model.opt.timestep)
            original_controller_dt = getattr(controller, "dt", None)
            model.opt.timestep = step_dt
            if original_controller_dt is not None:
                controller.dt = step_dt
            try:
                controller.step(enabled_sides=("left", "right"))
            finally:
                model.opt.timestep = original_model_dt
                if original_controller_dt is not None:
                    controller.dt = original_controller_dt
            rollback[row] |= any(controller.arms[s].safety_rollback_last_step for s in ("left", "right"))
            mujoco.mj_forward(model, data)
        wall[row] = perf_counter() - started
        audit(row)
    rows = task.source_row_indices if hasattr(task, "source_row_indices") else task.source_row_index
    return BimanualMPCResult(mode, initial[0], initial[1], q["left"], q["right"],
        tcp["left"], tcp["right"], lag["left"], lag["right"], rollback, collisions,
        steps, wall, np.asarray(rows).copy())


__all__ = ["BimanualMPCResult", "MPCScene", "run_mpc_task", "scheduled_mpc_runs"]
