"""Full-source-timeline evaluation of the isolated native-DOF EasyIK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from .controller_adapter import make_easyik, set_bimanual_targets
from .strict_bimanual_ik import BimanualIKResult


@dataclass(frozen=True)
class EasyIKBudget:
    """An identical, declared number of controller steps for every target."""

    iterations_per_target: int
    position_tolerance_m: float = 0.001
    orientation_tolerance_rad: float = np.deg2rad(1.5)

    def __post_init__(self):
        if self.iterations_per_target < 1:
            raise ValueError("iterations_per_target must be positive")


@dataclass(frozen=True)
class EasyIKScene:
    model: Any
    data: Any
    contract: Any
    name_map: Mapping[str, Mapping[str, Any]]
    controller_config: Any = None
    velocity_limits: Any = None
    collision_checker: Any = None


@dataclass
class EasyIKResult(BimanualIKResult):
    declared_iterations_per_target: int = 0
    iterations_used: np.ndarray | None = None


def _actual_pose(data, site_id: int) -> np.ndarray:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    return np.concatenate((data.site_xpos[site_id].copy(), quaternion))


def _orientation_error(target_wxyz: np.ndarray, actual_wxyz: np.ndarray) -> float:
    dot = float(abs(np.dot(target_wxyz, actual_wxyz)))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def run_easyik_task(scene: EasyIKScene, task: Any, budget: EasyIKBudget) -> EasyIKResult:
    """Run synchronized targets and externally score actual MuJoCo TCP poses."""
    model, data = scene.model, scene.data
    controller = make_easyik(
        model, data, scene.contract, scene.controller_config,
        name_map=scene.name_map, velocity_limits=scene.velocity_limits,
    )
    count = len(task.time_s)
    dof = int(scene.contract.dof_per_arm)
    q = {side: np.full((count, dof), np.nan) for side in ("left", "right")}
    tcp = {side: np.full((count, 7), np.nan) for side in ("left", "right")}
    pos = {side: np.full(count, np.nan) for side in ("left", "right")}
    rot = {side: np.full(count, np.nan) for side in ("left", "right")}
    success = np.zeros(count, dtype=bool)
    used = np.zeros(count, dtype=int)
    failures: list[str | None] = [None] * count
    collisions: list[tuple[str, ...]] = [()] * count
    branches: list[tuple[int, int] | None] = [None] * count

    positions = {"left": task.left_position_m, "right": task.right_position_m}
    quaternions = {"left": task.left_quaternion_wxyz, "right": task.right_quaternion_wxyz}
    for row in range(count):
        set_bimanual_targets(
            data, controller,
            (positions["left"][row], quaternions["left"][row]),
            (positions["right"][row], quaternions["right"][row]),
        )
        for iteration in range(1, budget.iterations_per_target + 1):
            controller.step(enabled_sides=("left", "right"))
            mujoco.mj_forward(model, data)
            used[row] = iteration
            within = True
            for side in ("left", "right"):
                arm = controller.arms[side]
                tcp[side][row] = _actual_pose(data, arm.site_id)
                pos[side][row] = np.linalg.norm(tcp[side][row, :3] - positions[side][row])
                rot[side][row] = _orientation_error(quaternions[side][row], tcp[side][row, 3:])
                within &= (pos[side][row] <= budget.position_tolerance_m and
                           rot[side][row] <= budget.orientation_tolerance_rad)
            if within:
                success[row] = True
                break
        for side in ("left", "right"):
            q[side][row] = data.qpos[controller.arms[side].qpos_ids]
        if success[row] and scene.collision_checker is not None:
            report = scene.collision_checker.state(q["left"][row], q["right"][row])
            if not report.valid:
                success[row] = False
                collisions[row] = tuple(item.value for item in report.classes)
                failures[row] = collisions[row][0]
        if success[row]:
            branches[row] = (0, 0)
        elif failures[row] is None:
            failures[row] = "easyik_convergence_exhausted"

    return EasyIKResult(
        q["left"], q["right"], tcp["left"], tcp["right"], pos["left"], pos["right"],
        rot["left"], rot["right"], branches, success, failures, collisions,
        np.asarray(task.source_row_indices if hasattr(task, "source_row_indices")
                   else task.source_row_index).copy(),
        budget.iterations_per_target, used,
    )


__all__ = ["EasyIKBudget", "EasyIKResult", "EasyIKScene", "run_easyik_task"]
