from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


class UnsupportedModelControllerContract(RuntimeError):
    pass


_ROBOT_LIMITS_PATH = Path(__file__).resolve().parents[1] / "configs" / "robot_limits.json"


def _configured_joint_velocity(robot_kind: str | None) -> np.ndarray:
    """Load audited model limits while retaining a conservative standalone fallback."""
    kind = (robot_kind or "").lower()
    key = (
        "kinova_gen3_lite"
        if "kinova" in kind or "gen3_lite" in kind
        else "xarm6"
        if "xarm6" in kind
        else "ur5_classic"
        if "ur5" in kind
        else "doosan_m0609"
    )
    fallback = {
        "kinova_gen3_lite": [0.5] * 6,
        "xarm6": [3.14] * 6,
        "ur5_classic": [math.pi] * 6,
        "doosan_m0609": [2.618, 2.618, 3.1416, 3.927, 3.927, 3.927],
    }[key]
    if not _ROBOT_LIMITS_PATH.exists():
        return np.asarray(fallback, dtype=np.float64)
    try:
        limits = json.loads(_ROBOT_LIMITS_PATH.read_text(encoding="utf-8"))
        velocity = np.asarray(limits[key]["joint_velocity_rad_s"], dtype=np.float64)
        if velocity.shape != (6,) or not np.all(np.isfinite(velocity)) or np.any(velocity <= 0):
            raise ValueError(f"invalid joint velocity limits for {key}: {velocity}")
        return velocity
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load audited robot limits from {_ROBOT_LIMITS_PATH}") from exc


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm <= 1e-12:
        return vector
    return vector * (maximum / norm)


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    a = _quat_normalize(q0)
    b = _quat_normalize(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _quat_normalize(a + amount * (b - a))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - amount) * theta) / sin_theta * a
        + math.sin(amount * theta) / sin_theta * b
    )


def _normal_matrix(jacobian: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute J.T W J without calling the environment's BLAS runtime."""
    rows, cols = jacobian.shape
    result = np.zeros((cols, cols), dtype=np.float64)
    for row in range(rows):
        weight = float(weights[row])
        for i in range(cols):
            left = float(jacobian[row, i]) * weight
            for j in range(cols):
                result[i, j] += left * float(jacobian[row, j])
    return result


def _weighted_rhs(jacobian: np.ndarray, weights: np.ndarray, vector: np.ndarray) -> np.ndarray:
    result = np.zeros(jacobian.shape[1], dtype=np.float64)
    for row in range(jacobian.shape[0]):
        scaled = float(weights[row]) * float(vector[row])
        for col in range(jacobian.shape[1]):
            result[col] += float(jacobian[row, col]) * scaled
    return result


def _symmetric_min_eigenvalue(matrix: np.ndarray) -> float:
    """Jacobi eigenvalue iteration for a tiny real symmetric matrix."""
    a = np.asarray(matrix, dtype=np.float64).copy()
    n = a.shape[0]
    for _ in range(80):
        p, q = 0, 1
        maximum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                value = abs(float(a[i, j]))
                if value > maximum:
                    maximum, p, q = value, i, j
        if maximum < 1e-12:
            break
        app, aqq, apq = float(a[p, p]), float(a[q, q]), float(a[p, q])
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        cosine, sine = math.cos(angle), math.sin(angle)
        for k in range(n):
            if k in (p, q):
                continue
            akp, akq = float(a[k, p]), float(a[k, q])
            a[k, p] = a[p, k] = cosine * akp - sine * akq
            a[k, q] = a[q, k] = sine * akp + cosine * akq
        a[p, p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        a[q, q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        a[p, q] = a[q, p] = 0.0
    return max(0.0, min(float(a[i, i]) for i in range(n)))


def _solve_small_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Partial-pivot Gaussian elimination; avoids NumPy LAPACK DLL loading."""
    a = np.asarray(matrix, dtype=np.float64).copy()
    b = np.asarray(rhs, dtype=np.float64).copy()
    n = b.size
    for pivot in range(n):
        best = max(range(pivot, n), key=lambda row: abs(float(a[row, pivot])))
        if best != pivot:
            a[[pivot, best]] = a[[best, pivot]]
            b[pivot], b[best] = b[best], b[pivot]
        diagonal = float(a[pivot, pivot])
        if abs(diagonal) < 1e-12:
            a[pivot, pivot] = diagonal = 1e-12
        for row in range(pivot + 1, n):
            factor = float(a[row, pivot]) / diagonal
            if factor == 0.0:
                continue
            a[row, pivot:] -= factor * a[pivot, pivot:]
            b[row] -= factor * b[pivot]
    result = np.zeros(n, dtype=np.float64)
    for row in range(n - 1, -1, -1):
        remainder = 0.0
        for col in range(row + 1, n):
            remainder += float(a[row, col]) * float(result[col])
        result[row] = (float(b[row]) - remainder) / float(a[row, row])
    return result


solve_small_system = _solve_small_system


@dataclass
class MPCConfig:
    horizon: int = 1
    candidate_scales: tuple[float, ...] = (0.0, 0.35, 0.65, 1.0, 1.25)
    task_tau: float = 0.032
    position_weight: float = 30.0
    orientation_weight: float = 1.6
    terminal_scale: float = 2.0
    damping: float = 0.010
    singular_value_soft: float = 0.015
    singular_speed_floor: float = 0.12
    singular_orientation_floor: float = 0.08
    singular_damping_gain: float = 8.0
    max_linear_speed: float = 1.8
    max_angular_speed: float = 5.0
    velocity_scale: float = 0.92
    acceleration_limit: float = 36.0
    jerk_limit: float = 1500.0
    jerk_cost: float = 1e-6
    velocity_cost: float = 0.0
    acceleration_cost: float = 0.0
    candidate_switch_cost: float = 0.0
    settle_position_deadband: float = 0.0
    settle_orientation_deadband: float = 0.0
    center_cost: float = 2e-4
    collision_cost: float = 800.0
    collision_margin: float = 0.015
    collision_avoidance_speed: float = 0.35
    singularity_cost: float = 0.5
    singularity_floor: float = 0.008
    wrist_redistribution_enabled: bool = False
    wrist_redistribution_threshold_rad: float = 0.14
    wrist_redistribution_gain: float = 1.5
    wrist_redistribution_speed_limit: float = 0.6
    joint_limit_cost: float = 0.002
    joint_limit_soft_margin: float = 0.18
    branch_reference_cost: float = 0.0
    branch_reference_candidate_enabled: bool = True
    branch_reference_tau: float = 0.08
    # Softly track the velocity of an offline/analytic IK branch inside the
    # weighted least-squares solve.  Unlike a null-space term this remains
    # effective for a regular 6-by-6 Jacobian, while zero preserves the
    # original Cartesian servo exactly.
    branch_velocity_cost: float = 0.0
    branch_velocity_activation_position_m: float = 0.0
    branch_velocity_activation_orientation_rad: float = 0.0
    branch_nullspace_gain: float = 0.0
    branch_nullspace_speed_limit: float = 0.8
    collision_detection_enabled: bool = True
    # Experimental random joint-space branch search.  A local search can make
    # a full trajectory worse after committing to the wrong homotopy class, so
    # keep this opt-in until a time-parameterized global branch planner exists.
    joint_bypass_enabled: bool = False
    target_position_tau: float = 0.012
    target_rotation_tau: float = 0.016
    target_max_speed: float = 3.0
    target_max_angular_speed: float = 8.0
    mocap_workspace_bounds_enabled: bool = False
    mocap_rate_limit_enabled: bool = False


@dataclass
class ArmState:
    side: str
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    site_id: int
    mocap_id: int
    q_min: np.ndarray
    q_max: np.ndarray
    dq_limit: np.ndarray
    q_center: np.ndarray
    dq_des: np.ndarray
    dq_prev: np.ndarray
    ddq_des: np.ndarray
    recovery_q: np.ndarray | None = None
    branch_reference_q: np.ndarray | None = None
    branch_reference_dq: np.ndarray | None = None
    recovery_retry_countdown: int = 0
    safety_rollback_last_step: bool = False
    safety_rollback_count: int = 0
    selected_candidate_index: int = -1
    filtered_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    filtered_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    target_initialized: bool = False
    last_debug: dict[str, float | str] = field(default_factory=dict)


class NativeDofDualArmMPCPVT:
    """Finite-candidate Cartesian MPC with ideal joint PVT tracking."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: MPCConfig | None = None,
        *,
        name_map: dict[str, dict[str, object]] | None = None,
        robot_kind: str | None = None,
        velocity_limits: dict[str, np.ndarray] | None = None,
    ):
        self.model = model
        self.data = data
        self.config = config or MPCConfig()
        self.name_map = name_map
        self.robot_kind = robot_kind
        self.velocity_limits = velocity_limits
        self.dt = float(model.opt.timestep)
        self.scratch = mujoco.MjData(model)
        self.arms = {side: self._make_arm(side) for side in ("left", "right")}
        self._robot_geom = self._build_robot_geom_mask()
        if self.config.collision_detection_enabled and self.config.collision_margin > 0.0:
            self.model.geom_margin[self._robot_geom] = np.maximum(
                self.model.geom_margin[self._robot_geom],
                self.config.collision_margin,
            )
        self._allowed_body_pairs = self._build_allowed_body_pairs()

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"MuJoCo object not found: {name}")
        return object_id

    def _make_arm(self, side: str) -> ArmState:
        names = self.name_map[side] if self.name_map is not None else None
        joint_ids = np.array(
            [
                self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in (
                    names["joints"]
                    if names is not None
                    else [f"{side}_joint_{i}" for i in range(1, 7)]
                )
            ],
            dtype=np.int32,
        )
        qpos_ids = self.model.jnt_qposadr[joint_ids].astype(np.int32)
        dof_ids = self.model.jnt_dofadr[joint_ids].astype(np.int32)
        limits = self.model.jnt_range[joint_ids].copy()
        q_min, q_max = limits[:, 0], limits[:, 1]
        model_names = self.robot_kind or bytes(self.model.names).decode(errors="ignore")
        velocity_from_urdf = (
            np.asarray(self.velocity_limits[side], dtype=np.float64)
            if self.velocity_limits is not None
            else _configured_joint_velocity(model_names)
        )
        if self.velocity_limits is None and velocity_from_urdf.shape == (6,) and joint_ids.size != 6:
            velocity_from_urdf = np.full(joint_ids.size, math.pi, dtype=np.float64)
        if velocity_from_urdf.shape != (joint_ids.size,):
            raise ValueError(f"invalid {side} velocity limits for {joint_ids.size} DOF")
        site_id = self._id(
            mujoco.mjtObj.mjOBJ_SITE,
            str(names["site"]) if names is not None else f"{side}_tcp",
        )
        target_body = self._id(
            mujoco.mjtObj.mjOBJ_BODY,
            str(names["target"]) if names is not None else f"{side}_target",
        )
        mocap_id = int(self.model.body_mocapid[target_body])
        return ArmState(
            side=side,
            joint_ids=joint_ids,
            qpos_ids=qpos_ids,
            dof_ids=dof_ids,
            site_id=site_id,
            mocap_id=mocap_id,
            q_min=q_min,
            q_max=q_max,
            dq_limit=velocity_from_urdf * self.config.velocity_scale,
            q_center=(q_min + q_max) * 0.5,
            dq_des=np.zeros(joint_ids.size),
            dq_prev=np.zeros(joint_ids.size),
            ddq_des=np.zeros(joint_ids.size),
        )

    def _build_robot_geom_mask(self) -> np.ndarray:
        result = np.zeros(self.model.ngeom, dtype=bool)
        # Joint-1 bodies are the roots of the two controlled kinematic trees.
        # Descendant membership is safer than name matching in a composite scene:
        # Willow demo blocks share the "willow_" prefix but are not robot geometry.
        root_bodies = {int(self.model.jnt_bodyid[arm.joint_ids[0]]) for arm in self.arms.values()}
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            ancestor = body_id
            while ancestor > 0 and ancestor not in root_bodies:
                ancestor = int(self.model.body_parentid[ancestor])
            result[geom_id] = ancestor in root_bodies
        return result

    def _build_allowed_body_pairs(self) -> set[tuple[int, int]]:
        allowed: set[tuple[int, int]] = set()
        for body_id in range(1, self.model.nbody):
            parent = int(self.model.body_parentid[body_id])
            if parent >= 0:
                allowed.add(tuple(sorted((body_id, parent))))
        # SRDF-style exclusion for compact second-neighbour geometry. A safety
        # distance otherwise reports permanent proximity inside UR-style wrists
        # even though those link pairs cannot collide independently.
        for first in range(1, self.model.nbody):
            first_chain = {first: 0}
            parent = first
            for distance in range(1, 4):
                parent = int(self.model.body_parentid[parent])
                if parent <= 0:
                    break
                first_chain[parent] = distance
            for second in range(first + 1, self.model.nbody):
                parent = second
                graph_distance = None
                for distance in range(0, 4):
                    if parent in first_chain:
                        graph_distance = distance + first_chain[parent]
                        break
                    parent = int(self.model.body_parentid[parent])
                    if parent <= 0:
                        break
                if graph_distance is not None and graph_distance <= 2:
                    allowed.add((first, second))
        # All contacts inside one closed-chain gripper are governed by its equality constraints.
        for side in ("left", "right"):
            hand_bodies = []
            for body_id in range(self.model.nbody):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                gripper_prefix = (
                    str(self.name_map[side].get("gripper_prefix", f"{side}_gripper_"))
                    if self.name_map is not None
                    else f"{side}_gripper_"
                )
                if name.startswith(gripper_prefix):
                    hand_bodies.append(body_id)
            for first in hand_bodies:
                for second in hand_bodies:
                    if first != second:
                        allowed.add(tuple(sorted((first, second))))
            # The rigid gripper mounting body stays close to the compact wrist
            # across several adjacent bodies and is part of the same assembly.
            wrist_bodies = []
            for body_id in range(self.model.nbody):
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                ) or ""
                if name.startswith(f"{side}_wrist") or name.startswith(
                    f"{self.robot_kind}_{side}_wrist"
                ):
                    wrist_bodies.append(body_id)
            for hand in hand_bodies:
                ancestor = hand
                for _ in range(5):
                    ancestor = int(self.model.body_parentid[ancestor])
                    if ancestor <= 0:
                        break
                    allowed.add(tuple(sorted((hand, ancestor))))
                for wrist in wrist_bodies:
                    allowed.add(tuple(sorted((hand, wrist))))
        return allowed

    def initialize_home(self) -> None:
        # Rounded hand-taught dual-arm startup pose. Keep the two arms
        # independent: the right values are canonicalized to [-180, 180]
        # (335掳 == -25掳, 190掳 == -170掳).
        model_names = (self.robot_kind or bytes(self.model.names).decode(errors="ignore")).lower()
        if "kinova" in model_names or "gen3_lite" in model_names:
            home_deg = {
                "left": np.array([0.0, -35.0, 105.0, 0.0, 65.0, 0.0]),
                "right": np.array([0.0, -35.0, 105.0, 0.0, 65.0, 0.0]),
            }
        elif "xarm6" in model_names:
            home_deg = {
                "left": np.array([0.0, -30.0, -60.0, 0.0, 90.0, 0.0]),
                "right": np.array([0.0, -30.0, -60.0, 0.0, 90.0, 0.0]),
            }
        elif "ur5" in model_names:
            home_deg = {
                "left": np.array([30.0, -90.0, 90.0, -90.0, -90.0, 0.0]),
                "right": np.array([30.0, -90.0, 90.0, -90.0, -90.0, 0.0]),
            }
        elif "doosan" in model_names:
            # Symmetric, table-forward benchmark pose. TCPs are approximately
            # 430 mm forward, 240 mm inward and share xArm6's flat wrist pose.
            home_deg = {
                "left": np.array(
                    [106.4113, -14.5932, -54.1046, -179.9914, 111.3136, 151.4240]
                ),
                "right": np.array(
                    [73.6126, 14.5924, 54.0834, -180.0095, -111.3126, 28.5758]
                ),
            }
        elif "willow" in model_names:
            home_deg = {
                "left": np.array([0.0, 65.0, 70.0, 0.0, 0.0, 0.0]),
                "right": np.array([0.0, 65.0, 70.0, 0.0, 0.0, 0.0]),
            }
        else:
            home_deg = {
                "left": np.array([140.0, 5.0, -115.0, -160.0, 40.0, 0.0]),
                "right": np.array([40.0, -25.0, 115.0, -170.0, -50.0, -10.0]),
            }
        for side, arm in self.arms.items():
            home = np.deg2rad(home_deg[side])
            if home.size != arm.joint_ids.size:
                home = self.data.qpos[arm.qpos_ids].copy()
            self.data.qpos[arm.qpos_ids] = np.clip(home, arm.q_min, arm.q_max)
            arm.dq_des[:] = 0.0
            arm.dq_prev[:] = 0.0
            arm.ddq_des[:] = 0.0
            arm.recovery_q = None
            arm.branch_reference_q = None
            arm.branch_reference_dq = None
            arm.recovery_retry_countdown = 0
            arm.selected_candidate_index = -1
        mujoco.mj_forward(self.model, self.data)

    def _filtered_target(self, arm: ArmState) -> tuple[np.ndarray, np.ndarray]:
        raw_position = self.data.mocap_pos[arm.mocap_id].copy()
        raw_quat = _quat_normalize(self.data.mocap_quat[arm.mocap_id].copy())
        if not arm.target_initialized:
            arm.filtered_position = raw_position
            arm.filtered_quaternion = raw_quat
            arm.target_initialized = True
            return raw_position, raw_quat

        pos_alpha = 1.0 - math.exp(-self.dt / max(self.config.target_position_tau, self.dt))
        desired_step = (raw_position - arm.filtered_position) * pos_alpha
        if self.config.mocap_rate_limit_enabled:
            desired_step = _limit_norm(desired_step, self.config.target_max_speed * self.dt)
        arm.filtered_position = arm.filtered_position + desired_step

        rot_alpha = 1.0 - math.exp(-self.dt / max(self.config.target_rotation_tau, self.dt))
        delta = np.zeros(3)
        mujoco.mju_subQuat(delta, raw_quat, arm.filtered_quaternion)
        angle = float(np.linalg.norm(delta))
        if self.config.mocap_rate_limit_enabled and angle > 1e-12:
            rot_alpha = min(rot_alpha, self.config.target_max_angular_speed * self.dt / angle)
        arm.filtered_quaternion = _quat_slerp(arm.filtered_quaternion, raw_quat, rot_alpha)
        return arm.filtered_position.copy(), arm.filtered_quaternion.copy()

    def _pose_error_and_jacobian(
        self,
        data: mujoco.MjData,
        arm: ArmState,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        current_position = data.site_xpos[arm.site_id]
        current_quat = np.zeros(4)
        mujoco.mju_mat2Quat(current_quat, data.site_xmat[arm.site_id])
        rotation_error_local = np.zeros(3)
        mujoco.mju_subQuat(rotation_error_local, target_quaternion, current_quat)
        # mju_subQuat expresses the tangent in the current site's local frame,
        # while mj_jacSite's rotational Jacobian is world-aligned.
        current_rotation = data.site_xmat[arm.site_id].reshape(3, 3)
        rotation_error = np.array(
            [
                sum(float(current_rotation[row, col]) * float(rotation_error_local[col]) for col in range(3))
                for row in range(3)
            ],
            dtype=np.float64,
        )
        error = np.concatenate((target_position - current_position, rotation_error))

        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jac_pos, jac_rot, arm.site_id)
        jacobian = np.vstack((jac_pos[:, arm.dof_ids], jac_rot[:, arm.dof_ids]))
        sigma_min = math.sqrt(_symmetric_min_eigenvalue(_normal_matrix(jacobian, np.ones(6))))
        return error, jacobian, sigma_min

    def _nominal_velocity(
        self,
        data: mujoco.MjData,
        arm: ArmState,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        error, jacobian, sigma_min = self._pose_error_and_jacobian(
            data, arm, target_position, target_quaternion
        )
        # A Cartesian command inside both deadbands is already satisfied.
        # Returning an exact zero velocity prevents the finite candidate set
        # from alternating between numerically equivalent wrist solutions.
        if (
            self.config.settle_position_deadband > 0.0
            and self.config.settle_orientation_deadband > 0.0
            and np.linalg.norm(error[:3]) <= self.config.settle_position_deadband
            and np.linalg.norm(error[3:])
            <= self.config.settle_orientation_deadband
        ):
            return np.zeros(len(arm.joint_ids)), error, sigma_min, self.config.damping
        twist = error / max(self.config.task_tau, self.dt)
        twist[:3] = _limit_norm(twist[:3], self.config.max_linear_speed)
        twist[3:] = _limit_norm(twist[3:], self.config.max_angular_speed)
        singular_amount = 1.0 - _smoothstep(sigma_min / self.config.singular_value_soft)
        safe_scale = 1.0 - singular_amount
        # DR_VAR_VEL-style behavior: start reacting before the Jacobian is
        # numerically singular.  Preserve Cartesian position authority while
        # temporarily relaxing orientation, then let the reference move beyond
        # the bad region instead of chasing it with explosive joint velocity.
        twist[:3] *= self.config.singular_speed_floor + (
            1.0 - self.config.singular_speed_floor
        ) * safe_scale
        twist[3:] *= self.config.singular_orientation_floor + (
            1.0 - self.config.singular_orientation_floor
        ) * safe_scale
        damping = self.config.damping * (
            1.0 + self.config.singular_damping_gain * singular_amount
        )
        weights = np.array(
            [self.config.position_weight] * 3 + [self.config.orientation_weight] * 3
        )
        lhs = _normal_matrix(jacobian, weights)
        lhs.flat[:: lhs.shape[0] + 1] += damping * damping
        rhs = _weighted_rhs(jacobian, weights, twist)
        if (
            arm.branch_reference_q is not None
            and self.config.branch_velocity_cost > 0.0
            and (
                self.config.branch_velocity_activation_position_m <= 0.0
                or np.linalg.norm(error[:3])
                > self.config.branch_velocity_activation_position_m
                or np.linalg.norm(error[3:])
                > self.config.branch_velocity_activation_orientation_rad
            )
        ):
            reference_error = arm.branch_reference_q - data.qpos[arm.qpos_ids]
            reference_error = (reference_error + np.pi) % (2.0 * np.pi) - np.pi
            reference_velocity = reference_error / max(
                self.config.branch_reference_tau, self.dt
            )
            if arm.branch_reference_dq is not None:
                reference_velocity = reference_velocity + arm.branch_reference_dq
            reference_weight = self.config.branch_velocity_cost
            lhs.flat[:: lhs.shape[0] + 1] += reference_weight
            rhs += reference_weight * reference_velocity
        dq = _solve_small_system(lhs, rhs)
        if (
            arm.branch_reference_q is not None
            and self.config.branch_nullspace_gain > 0.0
        ):
            reference_error = arm.branch_reference_q - data.qpos[arm.qpos_ids]
            reference_error = (
                reference_error + np.pi
            ) % (2.0 * np.pi) - np.pi
            # Damped task nullspace: for a healthy non-redundant 6-DoF arm
            # this is almost zero, but it opens smoothly as a singular
            # direction develops.  The branch front-end can therefore choose
            # the disappearing wrist/shoulder coordinate without directly
            # fighting the Cartesian command in regular configurations.
            weighted_jt = jacobian.T * weights[np.newaxis, :]
            task_inverse = np.column_stack(
                [
                    _solve_small_system(lhs, weighted_jt[:, column])
                    for column in range(6)
                ]
            )
            null_projector = np.eye(len(arm.joint_ids))
            for row in range(len(arm.joint_ids)):
                for column in range(len(arm.joint_ids)):
                    null_projector[row, column] -= sum(
                        float(task_inverse[row, inner])
                        * float(jacobian[inner, column])
                        for inner in range(6)
                    )
            projected_reference = np.asarray(
                [
                    sum(
                        float(null_projector[row, column])
                        * float(reference_error[column])
                        for column in range(len(arm.joint_ids))
                    )
                    for row in range(len(arm.joint_ids))
                ]
            )
            branch_velocity = (
                self.config.branch_nullspace_gain * projected_reference
            )
            branch_velocity = _limit_norm(
                branch_velocity, self.config.branch_nullspace_speed_limit
            )
            dq += branch_velocity
        if (
            self.config.wrist_redistribution_enabled
            and "doosan" in (self.robot_kind or "").lower()
        ):
            q = data.qpos[arm.qpos_ids]
            wrist_angle = abs(float(q[4]))
            threshold = max(
                self.config.wrist_redistribution_threshold_rad, 1e-6
            )
            wrist_amount = 1.0 - _smoothstep(wrist_angle / threshold)
            if wrist_amount > 0.0:
                # At the M0609 spherical-wrist singularity J4 and J6 axes
                # coincide.  [0,0,0,+1,0,-1] is therefore an analytic
                # instantaneous null direction.  Use it to balance the two
                # wrist coordinates instead of letting numerical DLS chatter
                # between arbitrary decompositions of the same tool twist.
                null_direction = np.array(
                    [0.0, 0.0, 0.0, 1.0, 0.0, -1.0]
                ) / math.sqrt(2.0)
                center_error = arm.q_center - q
                scalar = self.config.wrist_redistribution_gain * float(
                    null_direction @ center_error
                )
                scalar = float(
                    np.clip(
                        scalar,
                        -self.config.wrist_redistribution_speed_limit,
                        self.config.wrist_redistribution_speed_limit,
                    )
                )
                dq += wrist_amount * scalar * null_direction
        return dq, error, sigma_min, damping

    def _velocity_bounds(self, arm: ArmState, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        margin = 0.025
        lower = np.maximum(-arm.dq_limit, (arm.q_min + margin - q) / self.dt)
        upper = np.minimum(arm.dq_limit, (arm.q_max - margin - q) / self.dt)
        return lower, upper

    def _rate_limit_velocity(
        self,
        desired: np.ndarray,
        previous_velocity: np.ndarray,
        previous_acceleration: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply component-wise acceleration and jerk bounds."""
        acceleration = (desired - previous_velocity) / self.dt
        acceleration = np.clip(
            acceleration,
            -self.config.acceleration_limit,
            self.config.acceleration_limit,
        )
        maximum_acceleration_change = self.config.jerk_limit * self.dt
        acceleration = previous_acceleration + np.clip(
            acceleration - previous_acceleration,
            -maximum_acceleration_change,
            maximum_acceleration_change,
        )
        return previous_velocity + acceleration * self.dt, acceleration

    def _recenter_equivalent_joint_angles(
        self, arm: ArmState, q: np.ndarray
    ) -> np.ndarray:
        """Choose the safest equivalent coordinate for multi-turn revolute joints.

        UR-style joints commonly expose a +/-360 degree coordinate range.  A
        controller that never changes the 2*pi representation can needlessly
        drive J6 into +360 degrees even though the identical physical pose is
        available near zero.  This is a coordinate-branch change, not an
        instantaneous physical joint motion, and is only allowed when both the
        current and equivalent coordinates are inside a range spanning at
        least two full turns.
        """
        recentered = q.copy()
        span = arm.q_max - arm.q_min
        for index in range(len(recentered)):
            if span[index] < (4.0 * math.pi - 1e-3):
                continue
            candidates = recentered[index] + 2.0 * math.pi * np.arange(-2, 3)
            valid = candidates[
                (candidates >= arm.q_min[index] + 0.05)
                & (candidates <= arm.q_max[index] - 0.05)
            ]
            if valid.size:
                recentered[index] = valid[
                    np.argmin(np.abs(valid - arm.q_center[index]))
                ]
        return recentered

    def _relevant_contact_distances(
        self, data: mujoco.MjData
    ) -> Iterable[float]:
        for contact in data.contact:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if not (self._robot_geom[geom1] or self._robot_geom[geom2]):
                continue
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if body1 == body2 or tuple(sorted((body1, body2))) in self._allowed_body_pairs:
                continue
            yield float(contact.dist)

    def collision_diagnostics(self, data: mujoco.MjData) -> dict[str, float]:
        if not self.config.collision_detection_enabled:
            return {
                "penalty": 0.0,
                "minimum_distance_m": float("inf"),
                "clearance_pairs": 0.0,
                "penetrating_pairs": 0.0,
            }
        distances = list(self._relevant_contact_distances(data))
        penalty = 0.0
        for distance in distances:
            clearance_violation = max(
                0.0, self.config.collision_margin - distance
            )
            if clearance_violation > 0.0:
                normalized = clearance_violation / max(
                    self.config.collision_margin, 1e-6
                )
                penalty += self.config.collision_cost * normalized * normalized
        return {
            "penalty": penalty,
            "minimum_distance_m": min(distances, default=float("inf")),
            "clearance_pairs": float(
                sum(distance < self.config.collision_margin for distance in distances)
            ),
            "penetrating_pairs": float(sum(distance < 0.0 for distance in distances)),
        }

    def _collision_penalty(self, data: mujoco.MjData) -> float:
        return float(self.collision_diagnostics(data)["penalty"])

    def _collision_avoidance_velocity(
        self, arm: ArmState, q: np.ndarray
    ) -> np.ndarray:
        """Finite-difference clearance ascent for contacts inside the margin."""
        full_qpos = self.data.qpos.copy()
        full_qpos[arm.qpos_ids] = q
        self._prepare_scratch(full_qpos)
        if self._collision_penalty(self.scratch) <= 0.0:
            return np.zeros(len(arm.joint_ids))
        epsilon = 2e-3
        gradient = np.zeros(len(arm.joint_ids))
        for joint in range(len(arm.joint_ids)):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[joint] = min(q_plus[joint] + epsilon, arm.q_max[joint])
            q_minus[joint] = max(q_minus[joint] - epsilon, arm.q_min[joint])
            full_qpos[arm.qpos_ids] = q_plus
            self._prepare_scratch(full_qpos)
            plus = self._collision_penalty(self.scratch)
            full_qpos[arm.qpos_ids] = q_minus
            self._prepare_scratch(full_qpos)
            minus = self._collision_penalty(self.scratch)
            denominator = max(q_plus[joint] - q_minus[joint], 1e-9)
            gradient[joint] = (plus - minus) / denominator
        # Descend collision cost. Normalisation avoids coupling the commanded
        # speed to arbitrary geom-count/cost scaling.
        norm = float(np.linalg.norm(gradient))
        if norm < 1e-9:
            return np.zeros(len(arm.joint_ids))
        return -self.config.collision_avoidance_speed * gradient / norm

    def _one_step_collision_penalty(
        self, arm: ArmState, q: np.ndarray, dq: np.ndarray
    ) -> float:
        full_qpos = self.data.qpos.copy()
        full_qpos[arm.qpos_ids] = np.clip(
            q + self.dt * dq, arm.q_min, arm.q_max
        )
        self._prepare_scratch(full_qpos)
        return self._collision_penalty(self.scratch)

    def _plan_collision_recovery(
        self,
        arm: ArmState,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> np.ndarray | None:
        """Find an alternate IK branch for a joint-space collision bypass."""
        rng = np.random.default_rng(4242 if arm.side == "left" else 4343)
        seeds = [self.data.qpos[arm.qpos_ids].copy()]
        lower = np.maximum(arm.q_min + 1e-4, np.radians(-175.0))
        upper = np.minimum(arm.q_max - 1e-4, np.radians(175.0))
        seeds.extend(rng.uniform(lower, upper) for _ in range(18))
        full_qpos = self.data.qpos.copy()
        weights = np.array(
            [self.config.position_weight] * 3
            + [self.config.orientation_weight] * 3
        )
        best_score = float("inf")
        best_q: np.ndarray | None = None
        for seed in seeds:
            q_candidate = seed.copy()
            for _ in range(100):
                full_qpos[arm.qpos_ids] = q_candidate
                self._prepare_scratch(full_qpos)
                error, jacobian, _ = self._pose_error_and_jacobian(
                    self.scratch,
                    arm,
                    target_position,
                    target_quaternion,
                )
                if (
                    np.linalg.norm(error[:3]) < 8e-4
                    and np.linalg.norm(error[3:]) < 1e-3
                ):
                    break
                lhs = _normal_matrix(jacobian, weights)
                lhs.flat[:: lhs.shape[0] + 1] += 0.05**2
                rhs = _weighted_rhs(jacobian, weights, error)
                increment = _solve_small_system(lhs, rhs)
                increment = _limit_norm(increment, 0.16)
                q_candidate = np.clip(
                    q_candidate + 0.45 * increment,
                    arm.q_min,
                    arm.q_max,
                )
            full_qpos[arm.qpos_ids] = q_candidate
            self._prepare_scratch(full_qpos)
            error, _, sigma = self._pose_error_and_jacobian(
                self.scratch,
                arm,
                target_position,
                target_quaternion,
            )
            collision = self.collision_diagnostics(self.scratch)
            margin = float(
                np.min(
                    np.minimum(
                        q_candidate - arm.q_min,
                        arm.q_max - q_candidate,
                    )
                )
            )
            score = (
                1000.0 * float(np.linalg.norm(error[:3]))
                + 10.0 * float(np.linalg.norm(error[3:]))
                + 10.0 * float(collision["penalty"])
                + 10000.0 * float(collision["penetrating_pairs"])
                + 0.2 * max(0.0, 0.04 - sigma) / 0.04
                + 0.1 * max(0.0, np.radians(10.0) - margin)
            )
            if (
                float(collision["penetrating_pairs"]) == 0.0
                and float(collision["penalty"]) == 0.0
                and score < best_score
            ):
                best_score = score
                best_q = q_candidate.copy()
        if best_q is None:
            return None
        # Do not trigger a bypass to essentially the same branch.
        if np.linalg.norm(best_q - self.data.qpos[arm.qpos_ids]) < 0.15:
            return None
        return best_q

    def _prepare_scratch(self, qpos: np.ndarray) -> None:
        self.scratch.qpos[:] = qpos
        self.scratch.qvel[:] = 0.0
        self.scratch.mocap_pos[:] = self.data.mocap_pos
        self.scratch.mocap_quat[:] = self.data.mocap_quat
        mujoco.mj_forward(self.model, self.scratch)

    def _rollout_cost(
        self,
        arm: ArmState,
        q0: np.ndarray,
        dq0: np.ndarray,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> float:
        full_qpos = self.data.qpos.copy()
        q = q0.copy()
        dq = dq0.copy()
        previous = arm.dq_des.copy()
        previous_acceleration = arm.ddq_des.copy()
        cost = 0.0
        for step in range(max(1, self.config.horizon)):
            if step > 0:
                full_qpos[arm.qpos_ids] = q
                self._prepare_scratch(full_qpos)
                dq, _, _, _ = self._nominal_velocity(
                    self.scratch, arm, target_position, target_quaternion
                )
            lower, upper = self._velocity_bounds(arm, q)
            dq = np.clip(dq, lower, upper)
            dq, acceleration = self._rate_limit_velocity(
                dq, previous, previous_acceleration
            )
            dq = np.clip(dq, lower, upper)
            q = np.clip(q + self.dt * dq, arm.q_min, arm.q_max)
            full_qpos[arm.qpos_ids] = q
            self._prepare_scratch(full_qpos)
            error, _, sigma_min = self._pose_error_and_jacobian(
                self.scratch, arm, target_position, target_quaternion
            )
            terminal = self.config.terminal_scale if step == self.config.horizon - 1 else 1.0
            cost += terminal * (
                self.config.position_weight * float(error[:3] @ error[:3])
                + self.config.orientation_weight * float(error[3:] @ error[3:])
            )
            cost += self.config.velocity_cost * float(dq @ dq)
            cost += self.config.acceleration_cost * float(acceleration @ acceleration)
            cost += self.config.jerk_cost * float((dq - previous) @ (dq - previous))
            centered = (q - arm.q_center) / np.maximum(arm.q_max - arm.q_min, 1e-6)
            cost += self.config.center_cost * float(centered @ centered)
            singular_violation = max(
                0.0,
                (self.config.singular_value_soft - sigma_min)
                / max(self.config.singular_value_soft, 1e-6),
            )
            cost += self.config.singularity_cost * singular_violation**2
            joint_margin = np.minimum(q - arm.q_min, arm.q_max - q)
            joint_violation = np.maximum(
                0.0, self.config.joint_limit_soft_margin - joint_margin
            )
            cost += self.config.joint_limit_cost * float(
                joint_violation @ joint_violation
            )
            cost += self._collision_penalty(self.scratch)
            previous = dq.copy()
            previous_acceleration = acceleration.copy()
        return cost

    def step_arm(self, arm: ArmState) -> None:
        target_position, target_quaternion = self._filtered_target(arm)
        q_raw = self.data.qpos[arm.qpos_ids].copy()
        q = self._recenter_equivalent_joint_angles(arm, q_raw)
        if not np.array_equal(q, q_raw):
            self.data.qpos[arm.qpos_ids] = q
            mujoco.mj_forward(self.model, self.data)
        dq_nominal, error, sigma_min, damping = self._nominal_velocity(
            self.data, arm, target_position, target_quaternion
        )
        avoidance_velocity = self._collision_avoidance_velocity(arm, q)
        candidates: list[np.ndarray] = []
        for scale in self.config.candidate_scales:
            candidate = arm.dq_des + scale * (dq_nominal - arm.dq_des)
            lower, upper = self._velocity_bounds(arm, q)
            candidate = np.clip(candidate, lower, upper)
            candidate, _ = self._rate_limit_velocity(
                candidate, arm.dq_des, arm.ddq_des
            )
            candidate = np.clip(candidate, lower, upper)
            candidates.append(candidate)
            if np.any(avoidance_velocity):
                avoidance_candidate = np.clip(
                    candidate + avoidance_velocity, lower, upper
                )
                avoidance_candidate, _ = self._rate_limit_velocity(
                    avoidance_candidate, arm.dq_des, arm.ddq_des
                )
                candidates.append(np.clip(avoidance_candidate, lower, upper))
        if (
            arm.branch_reference_q is not None
            and self.config.branch_reference_candidate_enabled
        ):
            reference_error = arm.branch_reference_q - q
            reference_error = (
                reference_error + np.pi
            ) % (2.0 * np.pi) - np.pi
            reference_velocity = reference_error / max(
                self.config.branch_reference_tau, self.dt
            )
            if arm.branch_reference_dq is not None:
                reference_velocity = reference_velocity + arm.branch_reference_dq
            reference_velocity = np.clip(reference_velocity, lower, upper)
            reference_velocity, _ = self._rate_limit_velocity(
                reference_velocity, arm.dq_des, arm.ddq_des
            )
            candidates.append(np.clip(reference_velocity, lower, upper))
        predicted_penalties = [
            self._one_step_collision_penalty(arm, q, candidate)
            for candidate in candidates
        ]
        if (
            predicted_penalties
            and min(predicted_penalties) > 0.0
            and not np.any(avoidance_velocity)
        ):
            safest_index = int(np.argmin(predicted_penalties))
            safest = candidates[safest_index]
            predicted_q = np.clip(
                q + self.dt * safest, arm.q_min, arm.q_max
            )
            predictive_avoidance = self._collision_avoidance_velocity(
                arm, predicted_q
            )
            if np.any(predictive_avoidance):
                detour, _ = self._rate_limit_velocity(
                    safest + predictive_avoidance,
                    arm.dq_des,
                    arm.ddq_des,
                )
                candidates.append(np.clip(detour, lower, upper))
                predicted_penalties.append(
                    self._one_step_collision_penalty(arm, q, candidates[-1])
                )
        collision_trap = bool(
            predicted_penalties and min(predicted_penalties) > 0.0
        )
        if arm.recovery_retry_countdown > 0:
            arm.recovery_retry_countdown -= 1
        if (
            self.config.joint_bypass_enabled
            and
            collision_trap
            and arm.recovery_q is None
            and arm.recovery_retry_countdown <= 0
        ):
            arm.recovery_q = self._plan_collision_recovery(
                arm, target_position, target_quaternion
            )
            arm.recovery_retry_countdown = (
                25 if arm.recovery_q is not None else 100
            )
        if not self.config.joint_bypass_enabled:
            arm.recovery_q = None
        if arm.recovery_q is not None:
            recovery_error = arm.recovery_q - q
            if np.linalg.norm(recovery_error) < 0.04:
                arm.recovery_q = None
            else:
                recovery_velocity = _limit_norm(
                    recovery_error / 0.45,
                    min(float(np.max(arm.dq_limit)), 1.2),
                )
                recovery_candidate, _ = self._rate_limit_velocity(
                    recovery_velocity,
                    arm.dq_des,
                    arm.ddq_des,
                )
                candidates.append(
                    np.clip(recovery_candidate, lower, upper)
                )
        if (
            self._collision_penalty(self.data) > 0.0
            or (
                predicted_penalties
                and min(predicted_penalties) > 0.0
            )
        ):
            # Safety-first fallback: if every task-following option enters the
            # margin, add a jerk/acceleration-limited braking trajectory.
            brake, _ = self._rate_limit_velocity(
                np.zeros(len(arm.joint_ids)), arm.dq_des, arm.ddq_des
            )
            candidates.append(np.clip(brake, lower, upper))
        scored = []
        for candidate_index, candidate in enumerate(candidates):
            candidate_cost = self._rollout_cost(
                    arm, q, candidate, target_position, target_quaternion
            )
            if arm.recovery_q is not None:
                predicted_q = np.clip(
                    q + self.dt * candidate, arm.q_min, arm.q_max
                )
                recovery_error = predicted_q - arm.recovery_q
                candidate_cost += 2.0 * float(
                    recovery_error @ recovery_error
                )
            if arm.branch_reference_q is not None and self.config.branch_reference_cost > 0.0:
                predicted_q = np.clip(
                    q + self.dt * candidate, arm.q_min, arm.q_max
                )
                reference_error = predicted_q - arm.branch_reference_q
                # Compare periodic joints through the shortest equivalent angle.
                reference_error = (
                    reference_error + np.pi
                ) % (2.0 * np.pi) - np.pi
                candidate_cost += self.config.branch_reference_cost * float(
                    reference_error @ reference_error
                )
            if (
                self.config.candidate_switch_cost > 0.0
                and arm.selected_candidate_index >= 0
                and candidate_index != arm.selected_candidate_index
            ):
                candidate_cost += self.config.candidate_switch_cost
            scored.append((candidate_cost, candidate, candidate_index))
        finite = [
            item
            for item in scored
            if np.isfinite(item[0]) and np.all(np.isfinite(item[1]))
        ]
        if finite:
            best_cost, best_dq, best_candidate_index = min(
                finite, key=lambda item: item[0]
            )
        else:
            # Zero velocity is a safety fallback, not a normal MPC candidate.
            # With a one-step horizon it otherwise becomes a sticky local optimum.
            best_cost, best_dq = float("inf"), np.zeros(len(arm.joint_ids))
            best_candidate_index = -1
        q_des = np.clip(q + self.dt * best_dq, arm.q_min, arm.q_max)
        arm.dq_prev = arm.dq_des.copy()
        arm.dq_des = best_dq.copy()
        arm.ddq_des = (arm.dq_des - arm.dq_prev) / self.dt
        arm.selected_candidate_index = best_candidate_index
        self.data.qpos[arm.qpos_ids] = q_des
        self.data.qvel[arm.dof_ids] = best_dq
        collision_debug = self.collision_diagnostics(self.data)
        arm.last_debug = {
            "controller": "mpc_pvt",
            "pos_err": float(np.linalg.norm(error[:3])),
            "rot_err": float(np.linalg.norm(error[3:])),
            "sigma_min": sigma_min,
            "damping": damping,
            "dq_norm": float(np.linalg.norm(best_dq)),
            "mpc_cost": float(best_cost),
            "candidates": float(len(candidates)),
            "selected_candidate_index": float(best_candidate_index),
            "filtered_target_x": float(target_position[0]),
            "filtered_target_y": float(target_position[1]),
            "filtered_target_z": float(target_position[2]),
            "collision_penalty": float(collision_debug["penalty"]),
            "collision_minimum_distance_m": float(
                collision_debug["minimum_distance_m"]
            ),
            "dual_arm_safety_rollback": 0.0,
        }

    def step(self, enabled_sides: Iterable[str] = ("left", "right")) -> None:
        enabled = tuple(enabled_sides)
        q_snapshot = {
            side: self.data.qpos[self.arms[side].qpos_ids].copy()
            for side in enabled
        }
        v_snapshot = {
            side: self.data.qvel[self.arms[side].dof_ids].copy()
            for side in enabled
        }
        state_snapshot = {
            side: (
                self.arms[side].dq_des.copy(),
                self.arms[side].dq_prev.copy(),
                self.arms[side].ddq_des.copy(),
                self.arms[side].selected_candidate_index,
            )
            for side in enabled
        }
        before_penetration = float(
            self.collision_diagnostics(self.data)["penetrating_pairs"]
        )
        for side in enabled:
            self.arms[side].safety_rollback_last_step = False
        for side in enabled:
            self.step_arm(self.arms[side])
            mujoco.mj_forward(self.model, self.data)
        after_penetration = float(
            self.collision_diagnostics(self.data)["penetrating_pairs"]
        )
        if after_penetration > before_penetration:
            # Per-arm moves can each be locally safe yet form an unsafe pair.
            # Treat the complete dual-arm update as one atomic transaction.
            for side in enabled:
                arm = self.arms[side]
                self.data.qpos[arm.qpos_ids] = q_snapshot[side]
                self.data.qvel[arm.dof_ids] = v_snapshot[side]
                dq_des, dq_prev, ddq_des, selected_candidate_index = state_snapshot[side]
                arm.dq_des[:] = dq_des
                arm.dq_prev[:] = dq_prev
                arm.ddq_des[:] = ddq_des
                arm.selected_candidate_index = selected_candidate_index
                arm.safety_rollback_last_step = True
                arm.safety_rollback_count += 1
                arm.last_debug["dual_arm_safety_rollback"] = 1.0
            mujoco.mj_forward(self.model, self.data)


class NativeDofDualArmQPServoPVT(NativeDofDualArmMPCPVT):
    """Direct constrained Cartesian QP-servo with the same PVT parameters."""

    def step_arm(self, arm: ArmState) -> None:
        target_position, target_quaternion = self._filtered_target(arm)
        q_raw = self.data.qpos[arm.qpos_ids].copy()
        q = self._recenter_equivalent_joint_angles(arm, q_raw)
        if not np.array_equal(q, q_raw):
            self.data.qpos[arm.qpos_ids] = q
            mujoco.mj_forward(self.model, self.data)
        dq_nominal, error, sigma_min, damping = self._nominal_velocity(
            self.data, arm, target_position, target_quaternion
        )
        lower, upper = self._velocity_bounds(arm, q)
        dq_des = np.clip(dq_nominal, lower, upper)
        dq_des, acceleration = self._rate_limit_velocity(
            dq_des, arm.dq_des, arm.ddq_des
        )
        dq_des = np.clip(dq_des, lower, upper)
        q_des = np.clip(q + self.dt * dq_des, arm.q_min, arm.q_max)

        # One-step collision backoff. It keeps the direct QP-servo path cheap
        # while preserving the same collision metric used by MPC rollouts.
        full_qpos = self.data.qpos.copy()
        full_qpos[arm.qpos_ids] = q_des
        self._prepare_scratch(full_qpos)
        collision_penalty = self._collision_penalty(self.scratch)
        if collision_penalty > 0.0:
            dq_des *= 0.15
            q_des = np.clip(q + self.dt * dq_des, arm.q_min, arm.q_max)
            acceleration = (dq_des - arm.dq_des) / self.dt

        arm.dq_prev = arm.dq_des.copy()
        arm.dq_des = dq_des.copy()
        arm.ddq_des = acceleration.copy()
        self.data.qpos[arm.qpos_ids] = q_des
        self.data.qvel[arm.dof_ids] = dq_des
        arm.last_debug = {
            "controller": "qpservo_pvt",
            "pos_err": float(np.linalg.norm(error[:3])),
            "rot_err": float(np.linalg.norm(error[3:])),
            "sigma_min": sigma_min,
            "damping": damping,
            "dq_norm": float(np.linalg.norm(dq_des)),
            "collision_penalty": float(collision_penalty),
            "mpc_cost": 0.0,
            "candidates": 1.0,
        }

