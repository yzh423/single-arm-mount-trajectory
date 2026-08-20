"""Solve one domain/robot/task against the exact real-mesh MuJoCo model."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.strict_mujoco_ik import (solve_pose_path_multibranch, solve_position_path,
                                      wrapped_joint_delta)
from scripts.strict_urdf_model_audit import MODELS, OFFICIAL, load_native_spec
from scripts.strict_trajectory_sources import (load_relative_task_trajectory, place_relative_positions,
                                                resample_trajectory, task_anchor_rotations,
                                                local_task_anchor_z)
from design_optimization.installation_search_space import mount_rotation_matrix
from design_optimization.episode_follow_metrics import episode_follow_metrics, failure_reason_diagnostics
from scripts.high_quality_robot_scene import base_mount_depth

ANCHOR_ROTATION = np.asarray(((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
_MOUNT_DEPTH_CACHE: dict[str, float] = {}

# Vendor collision proxies that overlap in every sampled posture by design.
# This is deliberately a body-pair allowlist, not a broad chain-distance rule.
PANDA_ALLOWED_COLLISION_PAIRS = {
    ("panda_leftfinger", "panda_rightfinger"),
    ("panda_link1_sc", "panda_link3_sc"),
}
OPENARM_ALLOWED_COLLISION_PAIRS = {
    ("openarm_left_left_finger", "openarm_left_right_finger"),
    ("openarm_left_link5", "openarm_left_link7"),
}


def vendor_allowed_collision_pairs(model: mujoco.MjModel) -> set[tuple[str, str]]:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand_tcp") >= 0:
        return PANDA_ALLOWED_COLLISION_PAIRS
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_hand_tcp") >= 0:
        return OPENARM_ALLOWED_COLLISION_PAIRS
    return set()


def target_poses_for(domain: str, task: str, *, split: str = "test",
                     episode_artifact: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    trajectory = resample_trajectory(
        load_relative_task_trajectory(domain, task, split=split,
                                      episode_artifact=episode_artifact), maximum_frames=300)
    # Local position_m is a translation in the capture/world frame, not in the
    # first hand frame. Preserve the measured path geometry when translating
    # it to the common task anchor.
    anchor_z = local_task_anchor_z(task) if domain == "local" else 0.35
    positions = place_relative_positions(trajectory.position_m, hand=trajectory.hand,
                                         anchor_z_m=anchor_z)
    quaternions = []
    for world_rotation in task_anchor_rotations(task, trajectory.quaternion_wxyz):
        world_quaternion = np.zeros(4); mujoco.mju_mat2Quat(world_quaternion, world_rotation.reshape(-1))
        quaternions.append(world_quaternion)
    return positions, np.asarray(quaternions), trajectory.time_s, trajectory.source


def targets_for(domain: str, task: str) -> np.ndarray:
    """Position-only compatibility view used by the mount screener."""
    return target_poses_for(domain, task)[0]


def optimization(domain: str, robot: str, task: str) -> dict[str, object]:
    strict = ROOT / "videos/single_arm/strict_cache" / domain / robot / f"{task}.json"
    if strict.is_file():
        row = json.loads(strict.read_text(encoding="utf-8"))
        if all(key in row for key in ("base_xyz_m", "tilt_deg", "yaw_deg", "roll_deg")):
            return row
    source = ROOT / (
        "reports/single_arm/dense_search_results.json"
        if domain == "local"
        else f"reports/single_arm/domains/{domain}/dense_results.json"
    )
    return json.loads(source.read_text(encoding="utf-8"))["robots"][robot]["per_task"][task]


def build_model(robot: str, base_xyz: np.ndarray, tilt_deg: float, yaw_deg: float = 0.0,
                roll_deg: float = 0.0, *, render_studio: bool = False):
    entry = MODELS[robot]
    rotation = mount_rotation_matrix(tilt_pitch_deg=float(tilt_deg), yaw_deg=float(yaw_deg), roll_deg=float(roll_deg))
    spec = load_native_spec(entry)
    parent = spec.body(entry.tcp_parent)
    if parent is None:
        raise RuntimeError(f"{robot}: missing TCP parent {entry.tcp_parent}")
    flange_parent = spec.body(entry.flange_parent or entry.tcp_parent)
    if flange_parent is None:
        raise RuntimeError(f"{robot}: missing flange parent {entry.flange_parent}")
    flange_parent.add_site(name="strict_flange", pos=[0.0, 0.0, 0.0], size=[0.006, 0.0, 0.0])
    if not entry.path.resolve().is_relative_to(OFFICIAL.resolve()):
        raise RuntimeError(f"{robot}: model source must be inside third_party")
    if robot not in _MOUNT_DEPTH_CACHE:
        _MOUNT_DEPTH_CACHE[robot] = base_mount_depth(spec, entry.joints)
    mount_depth = _MOUNT_DEPTH_CACHE[robot]
    parent.add_site(name="strict_tracking_tcp", pos=entry.tool_translation_m,
                    quat=entry.tool_quaternion_wxyz, size=[0.008, 0.0, 0.0])
    table_kwargs = {}
    if render_studio:
        spec.visual.headlight.diffuse[:] = [0.58, 0.58, 0.58]
        spec.visual.headlight.ambient[:] = [0.40, 0.40, 0.40]
        spec.visual.headlight.specular[:] = [0.12, 0.12, 0.12]
        spec.visual.quality.shadowsize = 4096
        spec.visual.global_.fovy = 52
        spec.add_texture(name="strict_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.86, 0.90, 0.96], rgb2=[0.48, 0.58, 0.72], width=512, height=3072)
        spec.add_texture(name="strict_table_texture", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.90, 0.92, 0.94], rgb2=[0.72, 0.76, 0.80], width=512, height=512)
        material = spec.add_material(name="strict_table_material", texrepeat=[10.0, 8.0], texuniform=True,
                                     reflectance=0.04, roughness=0.88)
        material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "strict_table_texture"
        table_kwargs["material"] = "strict_table_material"
    spec.worldbody.add_geom(name="strict_table", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[0.95, 0.72, 0.04], pos=[0.0, 0.0, -0.04], **table_kwargs)
    if render_studio:
        for index, (leg_x, leg_y) in enumerate(((-0.82, -0.59), (-0.82, 0.59), (0.82, -0.59), (0.82, 0.59))):
            spec.worldbody.add_geom(name=f"strict_table_leg_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=[0.045, 0.045, 0.34], pos=[leg_x, leg_y, -0.42], rgba=[0.18, 0.21, 0.25, 1.0])
        spec.worldbody.add_geom(name="strict_studio_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[3.0, 3.0, 0.05], pos=[0.0, 0.0, -0.76], rgba=[0.46, 0.50, 0.56, 1.0])
        spec.worldbody.add_light(name="strict_key_light", pos=[0.9, -0.8, 1.6], dir=[-0.45, 0.4, -1.0],
                                 directional=True, diffuse=[0.66, 0.66, 0.66], specular=[0.16, 0.16, 0.16], castshadow=True)
        spec.worldbody.add_light(name="strict_fill_light", pos=[-0.8, 0.7, 1.3], dir=[0.5, -0.35, -1.0],
                                 directional=True, diffuse=[0.38, 0.38, 0.38], specular=[0.05, 0.05, 0.05], castshadow=False)
    if float(base_xyz[2]) > 1e-6:
        mount_normal = rotation[:, 2]
        interface = np.asarray(base_xyz, dtype=float)
        bottom = np.asarray((interface[0], interface[1], 0.0), dtype=float)
        top = np.asarray((interface[0], interface[1], max(0.001, interface[2] - 0.035)), dtype=float)
        spec.worldbody.add_geom(
            name="strict_pedestal",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.055, 0.0, 0.0],
            fromto=[*bottom, *top],
            rgba=[0.30, 0.33, 0.38, 0.45] if render_studio else [0.45, 0.47, 0.50, 1.0],
        )
        spec.worldbody.add_geom(
            name="strict_mount_adapter",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.075, 0.0, 0.0],
            fromto=[*(interface - 0.025 * mount_normal), *(interface + 0.025 * mount_normal)],
            rgba=[0.22, 0.26, 0.32, 0.78] if render_studio else [0.32, 0.35, 0.40, 1.0],
        )
    model = spec.compile()
    roots = [body for body in range(1, model.nbody) if int(model.body_parentid[body]) == 0]
    if len(roots) != 1:
        raise RuntimeError(f"{robot}: expected one root, found {roots}")
    root = roots[0]
    mount_quat = np.zeros(4); mujoco.mju_mat2Quat(mount_quat, rotation.reshape(-1))
    source_position = model.body_pos[root].copy()
    source_quat = model.body_quat[root].copy()
    # base_xyz is the fixed table-column adapter center. Move the vendor root
    # outward along the selected mount normal so its base surface sits on it.
    model.body_pos[root] = np.asarray(base_xyz) + rotation[:, 2] * mount_depth + rotation @ source_position
    combined = np.zeros(4)
    mujoco.mju_mulQuat(combined, mount_quat, source_quat)
    model.body_quat[root] = combined
    # The official gripper URDFs use zero as fully closed. Arm-only IK leaves
    # these passive joints fixed, so start them slightly open instead of with
    # the two finger collision meshes interpenetrating.
    for finger_joint in (
        "panda_finger_joint1", "panda_finger_joint2",
        "openarm_left_finger_joint1", "openarm_left_finger_joint2",
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_joint)
        if joint_id >= 0:
            address = int(model.jnt_qposadr[joint_id])
            lower, upper = model.jnt_range[joint_id]
            model.qpos0[address] = float(np.clip(0.02, lower, upper))
    return model


def _set_cylinder_fromto(
    model: mujoco.MjModel, geom_id: int, start: np.ndarray, stop: np.ndarray
) -> None:
    vector = np.asarray(stop, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise ValueError("cylinder endpoints must be distinct")
    model.geom_pos[geom_id] = (np.asarray(start, dtype=float) + np.asarray(stop, dtype=float)) / 2.0
    model.geom_size[geom_id, 1] = length / 2.0
    mujoco.mju_quatZ2Vec(model.geom_quat[geom_id], vector / length)
    model.geom_rbound[geom_id] = math.hypot(
        float(model.geom_size[geom_id, 0]), float(model.geom_size[geom_id, 1]))


@dataclass
class MountModelTemplate:
    robot: str
    model: mujoco.MjModel
    root_body: int
    source_position: np.ndarray
    source_quaternion: np.ndarray
    mount_depth: float
    pedestal_geom: int
    adapter_geom: int

    def apply(
        self, base_xyz: np.ndarray, tilt_deg: float, yaw_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> tuple[mujoco.MjModel, mujoco.MjData]:
        base = np.asarray(base_xyz, dtype=float)
        if base.shape != (3,) or base[2] <= 1e-6:
            raise ValueError("reusable mount template requires a positive three-dimensional base")
        rotation = mount_rotation_matrix(
            tilt_pitch_deg=float(tilt_deg), yaw_deg=float(yaw_deg), roll_deg=float(roll_deg))
        self.model.body_pos[self.root_body] = (
            base + rotation[:, 2] * self.mount_depth + rotation @ self.source_position)
        mount_quat = np.zeros(4)
        mujoco.mju_mat2Quat(mount_quat, rotation.reshape(-1))
        mujoco.mju_mulQuat(
            self.model.body_quat[self.root_body], mount_quat, self.source_quaternion)
        bottom = np.asarray((base[0], base[1], 0.0), dtype=float)
        top = np.asarray((base[0], base[1], max(0.001, base[2] - 0.035)), dtype=float)
        _set_cylinder_fromto(self.model, self.pedestal_geom, bottom, top)
        _set_cylinder_fromto(
            self.model, self.adapter_geom,
            base - 0.025 * rotation[:, 2], base + 0.025 * rotation[:, 2])
        data = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model, data)
        mujoco.mj_forward(self.model, data)
        return self.model, data


def build_mount_model_template(robot: str) -> MountModelTemplate:
    """Compile one mutable real-mesh topology for repeated installation poses."""
    base = np.asarray((0.0, -0.15, 0.335), dtype=float)
    model = build_model(robot, base, 0.0, 0.0, 0.0)
    roots = [body for body in range(1, model.nbody) if int(model.body_parentid[body]) == 0]
    if len(roots) != 1:
        raise RuntimeError(f"{robot}: expected one reusable root, found {roots}")
    root = roots[0]
    mount_depth = _MOUNT_DEPTH_CACHE[robot]
    source_position = model.body_pos[root].copy() - base - np.asarray((0.0, 0.0, mount_depth))
    pedestal = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strict_pedestal")
    adapter = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strict_mount_adapter")
    if pedestal < 0 or adapter < 0:
        raise RuntimeError(f"{robot}: reusable mount geometry is missing")
    return MountModelTemplate(
        robot=robot, model=model, root_body=root,
        source_position=source_position, source_quaternion=model.body_quat[root].copy(),
        mount_depth=mount_depth, pedestal_geom=pedestal, adapter_geom=adapter)


def active_ancestor_map(parent_ids, active_bodies: set[int]) -> dict[int, int]:
    """Map fixed descendants to the nearest body driven by an evaluated arm joint."""
    mapped = {0: 0}
    for body in range(1, len(parent_ids)):
        cursor = body
        while cursor and cursor not in active_bodies:
            cursor = int(parent_ids[cursor])
        mapped[body] = cursor
    return mapped


def active_chain_distance(body1: int, body2: int, parent_ids, active_ancestor: dict[int, int]) -> int:
    def lineage(body: int) -> list[int]:
        result = []
        while body:
            result.append(body)
            body = active_ancestor[int(parent_ids[body])]
        return result
    first, second = lineage(body1), lineage(body2)
    second_index = {body: index for index, body in enumerate(second)}
    return min((index + second_index[body] for index, body in enumerate(first) if body in second_index), default=10**9)


def hold_invalid_frames(q_path: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Prevent collision/IK failures from causing discontinuous random motion."""
    stable = np.asarray(q_path, dtype=float).copy()
    invalid = np.asarray(invalid, dtype=bool)
    last_safe = stable[0].copy()
    for index in range(len(stable)):
        if invalid[index]:
            stable[index] = last_safe
        else:
            last_safe = stable[index].copy()
    return stable


def evaluate_q_path(model, joint_names, q_path, targets, target_quaternions):
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "strict_tracking_tcp")
    reached, reached_q, pe, oe = [], [], [], []
    for q, target, target_q in zip(q_path, targets, target_quaternions):
        data.qpos[addresses] = q; mujoco.mj_forward(model, data)
        quaternion = np.zeros(4); mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
        residual = np.zeros(3); mujoco.mju_subQuat(residual, target_q, quaternion)
        reached.append(data.site_xpos[site].copy()); reached_q.append(quaternion)
        pe.append(float(np.linalg.norm(target - data.site_xpos[site]))); oe.append(float(np.linalg.norm(residual)))
    return np.asarray(reached), np.asarray(reached_q), np.asarray(pe), np.asarray(oe)


def collision_flags(model: mujoco.MjModel, joint_names, q_path: np.ndarray,
                    *, allowed_collision_pairs: set[tuple[str, str]] | None = None):
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    active_bodies = {int(model.jnt_bodyid[joint_id]) for joint_id in joint_ids}
    active_ancestor = active_ancestor_map(model.body_parentid, active_bodies)
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strict_table")
    pedestal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strict_pedestal")
    adapter_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strict_mount_adapter")
    pairs = vendor_allowed_collision_pairs(model) if allowed_collision_pairs is None else allowed_collision_pairs
    allowed = {frozenset(pair) for pair in pairs}
    panda_collision_layers = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_link0_sc") >= 0)
    table, self_collision = [], []
    for q in q_path:
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        table_hit = False
        self_hit = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) >= -1e-5:
                continue
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if table_id in (geom1, geom2):
                table_hit = True
                continue
            if pedestal_id in (geom1, geom2) or adapter_id in (geom1, geom2):
                mount_geom = pedestal_id if pedestal_id in (geom1, geom2) else adapter_id
                robot_geom = geom2 if geom1 == mount_geom else geom1
                robot_body = int(model.geom_bodyid[robot_geom])
                robot_body_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, robot_body) or ""
                # Panda *_sc bodies are internal self-collision proxies, not
                # environment geometry. Detailed meshes remain authoritative
                # for table, pedestal, and adapter collision.
                if panda_collision_layers and robot_body_name.endswith("_sc"):
                    continue
                chain = active_ancestor[robot_body]
                # Root housing contact is the intended bolted interface;
                # every actuated descendant must remain outside the pedestal.
                if chain:
                    table_hit = True
                continue
            body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
            chain1, chain2 = active_ancestor[body1], active_ancestor[body2]
            body_name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
            if panda_collision_layers:
                first_proxy = bool(body_name1 and body_name1.endswith("_sc"))
                second_proxy = bool(body_name2 and body_name2.endswith("_sc"))
                # The official model contains detailed environment meshes and
                # coarse *_sc bodies for self collision. Cross-layer contacts
                # are intentional overlap, and detailed meshes must not
                # duplicate the proxy self-collision layer.
                if first_proxy != second_proxy or not first_proxy:
                    continue
            explicitly_allowed = frozenset((body_name1, body_name2)) in allowed
            adjacent = active_chain_distance(chain1, chain2, model.body_parentid, active_ancestor) <= 1
            if not chain1 and not chain2:
                continue
            if chain1 and chain2 and not adjacent and not explicitly_allowed:
                self_hit = True
        table.append(table_hit)
        self_collision.append(self_hit)
    return np.asarray(table, dtype=bool), np.asarray(self_collision, dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("local", "droid", "egodex"), required=True)
    parser.add_argument("--robot", choices=tuple(MODELS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--input-fingerprint", help="result-affecting input hash recorded for safe resume")
    args = parser.parse_args()
    targets, target_quaternions, time_s, trajectory_source = target_poses_for(args.domain, args.task)
    selected = optimization(args.domain, args.robot, args.task)
    base = np.asarray(selected["base_xyz_m"], dtype=float)
    model = build_model(args.robot, base, float(selected["tilt_deg"]), float(selected.get("yaw_deg", 0.0)),
                        float(selected.get("roll_deg", 0.0)))
    entry = MODELS[args.robot]
    position_seed = solve_position_path(
        model, "strict_tracking_tcp", entry.joints, targets[:1],
        tolerance_m=1e-3, iterations=280, restarts=16)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in entry.joints]
    joint_ranges = np.asarray([model.jnt_range[joint_id] for joint_id in joint_ids])
    periodic = (joint_ranges[:, 1] - joint_ranges[:, 0]) >= (2.0 * np.pi - 1e-6)
    state_safety_cache: dict[tuple[float, ...], bool] = {}
    edge_safety_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], bool] = {}

    def state_collision_free(q: np.ndarray) -> bool:
        key = tuple(np.round(np.asarray(q, dtype=float), 7))
        if key not in state_safety_cache:
            table_hit, self_hit = collision_flags(model, entry.joints, np.asarray(q)[None, :])
            state_safety_cache[key] = not bool(table_hit[0] or self_hit[0])
        return state_safety_cache[key]

    def edge_collision_free(previous_q: np.ndarray, candidate_q: np.ndarray) -> bool:
        first = tuple(np.round(np.asarray(previous_q, dtype=float), 7))
        second = tuple(np.round(np.asarray(candidate_q, dtype=float), 7))
        key = (first, second)
        if key not in edge_safety_cache:
            delta = wrapped_joint_delta(candidate_q, previous_q, periodic)
            samples = max(2, int(np.ceil(np.max(np.abs(delta)) / np.deg2rad(5.0))) + 1)
            path = np.asarray(previous_q)[None, :] + np.linspace(0.0, 1.0, samples)[:, None] * delta[None, :]
            table_hit, self_hit = collision_flags(model, entry.joints, path)
            edge_safety_cache[key] = not bool(table_hit.any() or self_hit.any())
        return edge_safety_cache[key]

    result = solve_pose_path_multibranch(
        model, "strict_tracking_tcp", entry.joints, targets, target_quaternions,
        time_s=time_s, branch_candidates=8, horizon=12, beam_width=8,
        velocity_limit_rad_s=np.deg2rad(720.0), maximum_frame_jump_rad=np.deg2rad(25.0),
        position_tolerance_m=1e-3, orientation_tolerance_rad=np.deg2rad(1.5),
        iterations=240, initial_q=position_seed.q[0],
        candidate_collision_free=state_collision_free,
        transition_collision_free=edge_collision_free)
    table_collision, self_collision = collision_flags(model, entry.joints, result.q)
    q_path = hold_invalid_frames(result.q, table_collision | self_collision)
    if not np.array_equal(q_path, result.q):
        reached_xyz, reached_quaternion, position_error, orientation_error = evaluate_q_path(
            model, entry.joints, q_path, targets, target_quaternions)
        success = (position_error <= 1e-3) & (orientation_error <= np.deg2rad(1.5))
        table_collision, self_collision = collision_flags(model, entry.joints, q_path)
    else:
        reached_xyz, reached_quaternion = result.reached_xyz_m, result.reached_quaternion_wxyz
        position_error, orientation_error, success = result.position_error_m, result.orientation_error_rad, result.success
    output = ROOT / "videos/single_arm/strict_cache" / args.domain / args.robot / f"{args.task}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    failure = failure_reason_diagnostics(
        position_error_m=position_error, orientation_error_rad=orientation_error,
        table_collision=table_collision, self_collision=self_collision,
        joint_discontinuity=result.joint_discontinuity)
    np.savez_compressed(
        output,
        q=q_path,
        target_xyz_m=targets,
        target_quaternion_wxyz=target_quaternions,
        time_s=time_s,
        reached_xyz_m=reached_xyz,
        reached_quaternion_wxyz=reached_quaternion,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
        success=success,
        joint_discontinuity=result.joint_discontinuity,
        velocity_violation=result.velocity_violation,
        acceleration_warning=result.acceleration_warning,
        branch_count=result.branch_count,
        chosen_branch_index=result.chosen_branch_index,
        recovery_mode=result.recovery_mode,
        singularity_margin=result.singularity_margin,
        joint_limit_margin=result.joint_limit_margin,
        failure_reason= failure["primary_per_frame"],
        table_collision=table_collision,
        self_collision=self_collision,
        base_xyz_m=base,
        tilt_deg=np.asarray(float(selected["tilt_deg"])),
        yaw_deg=np.asarray(float(selected.get("yaw_deg", 0.0))),
        roll_deg=np.asarray(float(selected.get("roll_deg", 0.0))),
        joint_names=np.asarray(entry.joints),
    )
    follow = episode_follow_metrics(
        pose_success=success, collision=table_collision | self_collision,
        position_error_m=position_error, orientation_error_rad=orientation_error)
    audit = {
        "domain": args.domain,
        "robot": args.robot,
        "task": args.task,
        "source_model": str(entry.path),
        "trajectory_source": trajectory_source,
        "frames": len(targets),
        "input_fingerprint": args.input_fingerprint,
        "episode_success": follow["episode_success"],
        "frame_coverage": follow["frame_coverage"],
        "longest_failure_run_frames": follow["longest_failure_run_frames"],
        "failure_reasons": {
            "primary_frame_counts": failure["primary_frame_counts"],
            "affected_frame_counts": failure["affected_frame_counts"],
        },
        "position_error_m": {
            "mean": float(position_error.mean()),
            "p95": float(np.quantile(position_error, 0.95)),
            "max": float(position_error.max()),
        },
        "orientation_error_deg": {
            "mean": float(np.degrees(orientation_error.mean())),
            "p95": float(np.degrees(np.quantile(orientation_error, 0.95))),
            "max": float(np.degrees(orientation_error.max())),
        },
        "joint_limits_respected": result.joint_limits_respected,
        "table_collision_frames": int(table_collision.sum()),
        "self_collision_frames": int(self_collision.sum()),
        "planner": {
            "type": "rolling_multibranch",
            "branch_candidates": 8,
            "horizon_frames": 12,
            "beam_width": 8,
            "velocity_limit_deg_s": 720.0,
            "frame_jump_cap_deg": 25.0,
            "velocity_violation_frames": int(result.velocity_violation.sum()),
            "acceleration_warning_frames": int(result.acceleration_warning.sum()),
            "recovery_mode_counts": {
                mode: int(np.sum(result.recovery_mode == mode))
                for mode in ("none", "limited_step", "bidirectional", "hold")
            },
        },
        "base_xyz_m": base.tolist(),
        "tilt_deg": float(selected["tilt_deg"]),
        "yaw_deg": float(selected.get("yaw_deg", 0.0)),
        "roll_deg": float(selected.get("roll_deg", 0.0)),
        "cache": str(output.relative_to(ROOT)).replace("\\", "/"),
        "status": "pass" if bool(success.all()) and result.joint_limits_respected and not table_collision.any() and not self_collision.any() else "fail",
    }
    audit_path = output.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
