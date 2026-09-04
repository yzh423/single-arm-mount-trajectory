"""Seal-Bag-style paired mount search for the PiperX Fold Box task."""
from __future__ import annotations

from dataclasses import replace
from collections import Counter
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig, MuJoCoCandidateGenerator,
)
from factory_bimanual.mujoco_collision_adapter import MuJoCoPairedCollisionChecker
from factory_bimanual.mount_topology import (
    MountTopologyConfig, MuJoCoMountTopologyChecker,
)
from factory_bimanual.mount_orientation import mount_quaternions
from factory_bimanual.mount_constraints import MINIMUM_BIMANUAL_BASE_SEPARATION_M
from factory_bimanual.per_task_mount_search import summarize_quality_arrays
from factory_bimanual.quaternion_trajectory import reconstruct_held_quaternions
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from scripts.render_factory_dual_xarm6_fold_box import (
    bounded_smooth_pose_series, mapped_quaternions_at_joint_midpoint, subset,
)
from scripts.render_factory_dual_xarm6_se3_follow import (
    audit_bimanual_collisions, solve_collision_safe_bimanual_method,
)
from scripts.search_fold_box_piperx_mount import (
    _normalize_yaw, _registered_task, layered_sample_indices,
    local_paired_refinements, rank_paired_mount_candidate, select_paired_mount,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/factory_bimanual/fold_box_dual_piperx/fold_box_piperx_paired_mount_search.json"
WORK = ROOT / ".tmp/piperx_fold_box_paired_mount"
CONTRACT = ROBOT_CONTRACTS["piperx"]
TABLE_HEIGHT_M = .75


def advance_connected_pairs(previous_pairs, valid_pairs, edge_valid):
    """Advance from the last connected layer without resetting after a gap."""
    previous_pairs = list(previous_pairs)
    valid_pairs = list(valid_pairs)
    if not valid_pairs:
        return previous_pairs, True
    if not previous_pairs:
        return valid_pairs, False
    connected = [current for current in valid_pairs
                 if any(edge_valid(previous, current)
                        for previous in previous_pairs)]
    return (connected, False) if connected else (previous_pairs, True)


def _on_table(xy):
    # 75 mm mounting-disc radius on the 1.8 x 1.4 m workbench.
    return abs(float(xy[0])) <= .825 and abs(float(xy[1])) <= .625


def _valid_mount(mount):
    left = np.asarray(mount["xy"]["left"], float)
    right = np.asarray(mount["xy"]["right"], float)
    maximum_z = 1.50 if "mode" in mount else .90
    return (_on_table(left) and _on_table(right) and
            np.linalg.norm(left - right) >= (
                MINIMUM_BIMANUAL_BASE_SEPARATION_M
                if "mode" in mount else .18) and
            TABLE_HEIGHT_M < float(mount["shared_base_z_m"]) <= maximum_z)


def _scene_mount_kwargs(task, mount):
    if "mode" not in mount:
        return {"mount_adapter_height_m": (
            float(mount["shared_base_z_m"]) - TABLE_HEIGHT_M)}
    center = np.mean(np.vstack((task.left_position_m,
                                task.right_position_m)), axis=0)
    return {
        "mount_base_z_m": float(mount["shared_base_z_m"]),
        "mount_adapter_height_m": (
            float(mount["shared_base_z_m"]) - TABLE_HEIGHT_M
            if mount["mode"] == "upright_table" else .08),
        "mount_quaternion_wxyz": mount_quaternions(
            mount["mode"], mount["xy"], center, mount["yaw"]),
        "mount_support_mode": mount["mode"],
    }


def _coarse_single_mounts(center, *, z=.81):
    records = []
    for radius in (.26, .34, .42):
        for angle_deg in np.arange(0.0, 360.0, 45.0):
            angle = np.deg2rad(angle_deg)
            xy = np.asarray(center) + radius * np.asarray(
                (np.cos(angle), np.sin(angle)))
            face = np.rad2deg(np.arctan2(center[1] - xy[1],
                                        center[0] - xy[0]))
            for yaw_offset in (-45.0, -30.0, -15.0, 0.0,
                               15.0, 30.0, 45.0):
                if _on_table(xy):
                    records.append({"xy": xy.tolist(),
                                    "yaw": _normalize_yaw(face + yaw_offset),
                                    "z": float(z)})
    return records


def deterministic_pair_mounts(task, *, maximum=72, shared_base_z_m=.81):
    """Expanded paired grid reduced by deterministic feature-space coverage."""
    if maximum < 1:
        raise ValueError("maximum paired mounts must be positive")
    centers = {
        side: np.asarray(getattr(task, f"{side}_position_m"), float)[:, :2].mean(axis=0)
        for side in ("left", "right")}
    pools = {side: _coarse_single_mounts(centers[side], z=shared_base_z_m)
             for side in ("left", "right")}
    task_axis = centers["right"] - centers["left"]
    records = []
    for left in pools["left"]:
        for right in pools["right"]:
            base_axis = np.asarray(right["xy"]) - np.asarray(left["xy"])
            if np.dot(base_axis, task_axis) <= 0:
                continue
            mount = {
                "xy": {"left": left["xy"], "right": right["xy"]},
                "yaw": {"left": left["yaw"], "right": right["yaw"]},
                "shared_base_z_m": float(shared_base_z_m),
            }
            if _valid_mount(mount):
                records.append(mount)
    records.sort(key=lambda mount: (
        *mount["xy"]["left"], *mount["xy"]["right"],
        mount["yaw"]["left"], mount["yaw"]["right"]))
    if len(records) <= maximum:
        return records
    features = []
    for mount in records:
        yaw_left = np.deg2rad(mount["yaw"]["left"])
        yaw_right = np.deg2rad(mount["yaw"]["right"])
        features.append([
            mount["xy"]["left"][0] / .5,
            mount["xy"]["left"][1] / .5,
            mount["xy"]["right"][0] / .5,
            mount["xy"]["right"][1] / .5,
            np.sin(yaw_left), np.cos(yaw_left),
            np.sin(yaw_right), np.cos(yaw_right),
        ])
    features = np.asarray(features)
    chosen = [0]
    minimum_distance = np.sum((features - features[0]) ** 2, axis=1)
    minimum_distance[0] = -np.inf
    while len(chosen) < maximum:
        index = int(np.argmax(minimum_distance))
        chosen.append(index)
        distance = np.sum((features - features[index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[chosen] = -np.inf
    return [records[index] for index in chosen]


def _prepare_targets(model, task, indices, mapped_quaternions=None):
    if mapped_quaternions is not None:
        sampled = subset(task, indices)
        sampled.left_quaternion_wxyz = np.asarray(
            mapped_quaternions["left"])[indices]
        sampled.right_quaternion_wxyz = np.asarray(
            mapped_quaternions["right"])[indices]
        return sampled
    mapped = mapped_quaternions_at_joint_midpoint(
        model, task, robot_name="piperx")
    values = {}
    for side in ("left", "right"):
        reconstructed = reconstruct_held_quaternions(task.time_s, mapped[side])
        position, quaternion = bounded_smooth_pose_series(
            getattr(task, f"{side}_position_m"),
            reconstructed.quaternion_wxyz,
            position_cap_m=.003, orientation_cap_rad=np.deg2rad(1.0),
            sigma_frames=1.0)
        values[f"{side}_position_m"] = position
        values[f"{side}_quaternion_wxyz"] = quaternion
    sampled = subset(replace(
        task, left_position_m=values["left_position_m"],
        right_position_m=values["right_position_m"]), indices)
    sampled.left_quaternion_wxyz = values["left_quaternion_wxyz"][indices]
    sampled.right_quaternion_wxyz = values["right_quaternion_wxyz"][indices]
    return sampled


def evaluate_pair(task, mount, serial, *, uniform_count=14,
                  global_seed_count=2, max_iterations=55,
                  maximum_candidates=2, constrained_fallback_enabled=True,
                  position_tolerance_m=.001,
                  orientation_tolerance_rad=np.deg2rad(1.5),
                  mapped_quaternions=None, clearance_margin_m=.015):
    if not _valid_mount(mount):
        raise ValueError("paired mount is outside table/non-overlap bounds")
    xy = mount["xy"]; yaw = mount["yaw"]
    distance = float(np.linalg.norm(
        np.asarray(xy["left"]) - np.asarray(xy["right"])))
    scene = WORK / f"pair_{serial:04d}.xml"
    build_same_model_scene(
        CONTRACT, distance, scene, table_height_m=TABLE_HEIGHT_M,
        mount_xy_m=xy, mount_yaw_deg=yaw,
        **_scene_mount_kwargs(task, mount))
    model = mujoco.MjModel.from_xml_path(str(scene))
    combined = .5 * (task.left_position_m + task.right_position_m)
    indices = layered_sample_indices(
        len(task.time_s), combined, uniform_count=uniform_count)
    sampled = _prepare_targets(
        model, task, indices, mapped_quaternions=mapped_quaternions)
    names = {side: {"joints": CONTRACT.prefixed_joint_names(side),
                    "site": f"{side}_tcp"} for side in ("left", "right")}
    data = mujoco.MjData(model)
    generator = MuJoCoCandidateGenerator(
        model, data, CONTRACT, name_map=names,
        config=CandidateGeneratorConfig(
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            max_iterations=max_iterations, maximum_candidates=maximum_candidates,
            global_seed_count=global_seed_count,
            dedup_rad=np.deg2rad(1.0),
            constrained_fallback_enabled=constrained_fallback_enabled))
    checker = MuJoCoPairedCollisionChecker(
        model, data, names, transition_steps=3,
        clearance_margin_m=clearance_margin_m)
    topology = MuJoCoMountTopologyChecker(
        model, data, names,
        config=MountTopologyConfig(transition_steps=3))
    independent = {"left": 0, "right": 0}
    pair_success = 0
    pair_collision_frames = 0
    pair_edge_collision_frames = 0
    structural_crossing_frames = 0
    structural_edge_crossing_frames = 0
    gripper_overlap_violation_frames = 0
    gripper_overlap_edge_violation_frames = 0
    gripper_overlap_frames = 0
    maximum_structural_crossing_m = 0.0
    maximum_gripper_overlap_m = 0.0
    disconnected_rows = []
    pair_counts = []
    sigmas = []
    errors = []
    collision_class_counts = Counter()
    previous_pairs = []
    for row in range(len(indices)):
        proposals = {}
        for side in ("left", "right"):
            proposals[side] = generator(model, CONTRACT, sampled, row, side)
            independent[side] += int(bool(proposals[side]))
        pair_reports = [
            ((left, right), checker.state(left.q, right.q))
            for left in proposals["left"] for right in proposals["right"]]
        collision_safe_pairs = [pair for pair, report in pair_reports
                                if report.valid]
        topology_reports = [
            (pair, topology.state(pair[0].q, pair[1].q))
            for pair in collision_safe_pairs]
        for _, report in topology_reports:
            maximum_structural_crossing_m = max(
                maximum_structural_crossing_m,
                report.maximum_structural_crossing_m)
            maximum_gripper_overlap_m = max(
                maximum_gripper_overlap_m, report.gripper_overlap_m)
        gripper_overlap_frames += int(any(
            report.gripper_overlap_count for _, report in topology_reports))
        structural_blocked = bool(topology_reports) and not any(
            report.valid for _, report in topology_reports)
        structural_crossing_frames += int(structural_blocked and any(
            report.structural_crossing_count for _, report in topology_reports))
        gripper_overlap_violation_frames += int(structural_blocked and any(
            report.gripper_overlap_m > topology.config.gripper_overlap_limit_m
            for _, report in topology_reports))
        valid_pairs = [pair for pair, report in topology_reports
                       if report.valid]
        pair_counts.append(len(valid_pairs))
        if not valid_pairs:
            for _pair, report in pair_reports:
                collision_class_counts.update(item.value for item in report.classes)
            pair_collision_frames += int(
                bool(proposals["left"] and proposals["right"])
                and not collision_safe_pairs)
            disconnected_rows.append(int(indices[row]))
            continue
        if not previous_pairs:
            transition_safe = valid_pairs
        else:
            collision_edges = []
            valid_edges = []
            topology_edges = []
            for previous in previous_pairs:
                old = (previous[0].q, previous[1].q)
                for pair in valid_pairs:
                    current = (pair[0].q, pair[1].q)
                    if not checker.transition(old, current).valid:
                        continue
                    collision_edges.append((previous, pair))
                    report = topology.transition(old, current)
                    maximum_structural_crossing_m = max(
                        maximum_structural_crossing_m,
                        report.maximum_structural_crossing_m)
                    maximum_gripper_overlap_m = max(
                        maximum_gripper_overlap_m,
                        report.gripper_overlap_m)
                    topology_edges.append(report)
                    if report.valid:
                        valid_edges.append((previous, pair))
            transition_safe = {id(pair) for _, pair in valid_edges}
            transition_safe = [pair for pair in valid_pairs
                               if id(pair) in transition_safe]
            if not transition_safe:
                if not collision_edges:
                    pair_edge_collision_frames += 1
            structural_edge_crossing_frames += int(any(
                report.structural_crossing_count for report in topology_edges))
            gripper_overlap_edge_violation_frames += int(any(
                report.gripper_overlap_m
                > topology.config.gripper_overlap_limit_m
                for report in topology_edges))
        if not transition_safe:
            disconnected_rows.append(int(indices[row]))
            continue
        chosen = max(transition_safe, key=lambda pair: (
            min(pair[0].singularity_margin, pair[1].singularity_margin),
            pair[0].joint_limit_margin_rad + pair[1].joint_limit_margin_rad))
        previous_pairs = transition_safe
        pair_success += 1
        sigmas.append(min(chosen[0].singularity_margin,
                          chosen[1].singularity_margin))
        errors.append(sum(item.position_error_m + item.orientation_error_rad
                          for item in chosen))
    count = len(indices)
    return {
        "mount": mount,
        "base_distance_m": distance,
        "sampled_frames": count,
        "sampled_source_rows": indices.tolist(),
        "left_coverage": independent["left"] / count,
        "right_coverage": independent["right"] / count,
        "synchronous_pair_coverage": pair_success / count,
        "continuous_pair_coverage": pair_success / count,
        "pair_collision_frames": pair_collision_frames,
        "pair_edge_collision_frames": pair_edge_collision_frames,
        "collision_class_counts": dict(collision_class_counts),
        "structural_crossing_frames": structural_crossing_frames,
        "structural_edge_crossing_frames": structural_edge_crossing_frames,
        "gripper_overlap_violation_frames": gripper_overlap_violation_frames,
        "gripper_overlap_edge_violation_frames": (
            gripper_overlap_edge_violation_frames),
        "gripper_overlap_frames": gripper_overlap_frames,
        "maximum_structural_crossing_m": maximum_structural_crossing_m,
        "maximum_gripper_overlap_m": maximum_gripper_overlap_m,
        "disconnected_rows": disconnected_rows,
        "rejection_reason": (
            "state_collision" if pair_collision_frames else
            "swept_collision" if pair_edge_collision_frames else
            None),
        "pair_candidate_sum": int(sum(pair_counts)),
        "p10_pair_singularity_margin": (
            float(np.percentile(sigmas, 10)) if sigmas else 0.0),
        "mean_pair_pose_error": float(np.mean(errors)) if errors else 1e9,
    }


def _longest_false_run(values):
    longest = current = 0
    for value in np.asarray(values, bool):
        current = 0 if value else current + 1
        longest = max(longest, current)
    return int(longest)


def _sparse_safe(record):
    return all(int(record.get(field, 0)) == 0 for field in (
        "pair_collision_frames", "pair_edge_collision_frames"))


def _sparse_mount(record):
    mount = json.loads(json.dumps(record["mount"]))
    mount["base_distance_m"] = float(record["base_distance_m"])
    return mount


def bounded_full_planning_indices(count, maximum=512):
    """Keep finalist planning memory bounded while preserving both endpoints."""
    count = int(count)
    maximum = int(maximum)
    if count < 1 or maximum < 2:
        raise ValueError("count must be positive and maximum must be at least two")
    if count <= maximum:
        return np.arange(count, dtype=int)
    stride = int(np.ceil((count - 1) / (maximum - 1)))
    return np.unique(np.r_[np.arange(0, count, stride, dtype=int), count - 1])


def _interpolate_joint_path(source_time_s, q, target_time_s):
    source_time_s = np.asarray(source_time_s, float)
    target_time_s = np.asarray(target_time_s, float)
    q = np.asarray(q, float)
    if len(q) > 1:
        delta = (np.diff(q, axis=0) + np.pi) % (2 * np.pi) - np.pi
        q = q[0:1] + np.vstack((np.zeros((1, q.shape[1])),
                                np.cumsum(delta, axis=0)))
    return np.column_stack([
        np.interp(target_time_s, source_time_s, q[:, joint])
        for joint in range(q.shape[1])])


def _full_pose_errors(model, qpos, task, mapped):
    data = mujoco.MjData(model)
    actual = {}; position_error = {}; orientation_error = {}
    for side in ("left", "right"):
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        reached = np.zeros((len(qpos), 7)); pe = np.zeros(len(qpos))
        oe = np.zeros(len(qpos))
        for row, q in enumerate(qpos):
            data.qpos[:] = q; mujoco.mj_forward(model, data)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
            reached[row] = np.r_[data.site_xpos[site_id], quat]
            pe[row] = np.linalg.norm(
                getattr(task, f"{side}_position_m")[row]
                - data.site_xpos[site_id])
            residual = np.empty(3)
            mujoco.mju_subQuat(residual, mapped[side][row], quat)
            oe[row] = np.linalg.norm(residual)
        actual[side] = reached
        position_error[side] = pe
        orientation_error[side] = oe
    return actual, position_error, orientation_error


def evaluate_full_pair(task, mount, serial):
    """Run and independently audit every source row for one finalist mount."""
    if not _valid_mount(mount):
        raise ValueError("full-audit mount is invalid")
    xy = mount["xy"]; yaw = mount["yaw"]
    distance = float(np.linalg.norm(
        np.asarray(xy["left"]) - np.asarray(xy["right"])))
    scene = WORK / f"full_pair_{serial:04d}.xml"
    build_same_model_scene(
        CONTRACT, distance, scene, table_height_m=TABLE_HEIGHT_M,
        mount_xy_m=xy, mount_yaw_deg=yaw,
        **_scene_mount_kwargs(task, mount))
    model = mujoco.MjModel.from_xml_path(str(scene))
    indices = bounded_full_planning_indices(len(task.time_s))
    prepared = _prepare_targets(model, task, indices)
    # Solve on the same bounded/smoothed target representation that the
    # sparse evaluator uses.  The final audit must compare against the
    # corresponding representation over every source row; comparing against
    # the unsmoothed raw CSV can turn an otherwise valid solution into an
    # artificial zero-coverage result at the 1 mm / 1.5 deg gate.
    full_indices = np.arange(len(task.time_s), dtype=int)
    full_prepared = _prepare_targets(model, task, full_indices)
    planned_mapped = {side: getattr(prepared, f"{side}_quaternion_wxyz")
              for side in ("left", "right")}
    (qpos, _actual, position_error, orientation_error, _strict,
     _discontinuity, _velocity, diagnostics) = \
        solve_collision_safe_bimanual_method(
            model, prepared, planned_mapped, robot_name="piperx")
    full_qpos = np.repeat(model.qpos0[None, :], len(task.time_s), axis=0)
    for side in ("left", "right"):
        joint_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{i}")
            for i in range(1, 7)]
        qids = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
        full_qpos[:, qids] = _interpolate_joint_path(
            prepared.time_s, qpos[:, qids], task.time_s)
    mapped = {side: getattr(full_prepared, f"{side}_quaternion_wxyz")
              for side in ("left", "right")}
    _actual, position_error, orientation_error = _full_pose_errors(
        model, full_qpos, full_prepared, mapped)
    qpos = full_qpos
    collision, state_classes, edge_classes = audit_bimanual_collisions(
        model, qpos, robot_name="piperx")
    names = {side: {"joints": CONTRACT.prefixed_joint_names(side),
                    "site": f"{side}_tcp"} for side in ("left", "right")}
    topology = MuJoCoMountTopologyChecker(
        model, mujoco.MjData(model), names,
        config=MountTopologyConfig(transition_steps=5))
    qids = {}
    for side in ("left", "right"):
        joint_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names[side]["joints"]]
        qids[side] = np.asarray(model.jnt_qposadr[joint_ids], int)
    topology_states = []
    topology_edges = []
    previous = None
    for row in qpos:
        current = (row[qids["left"]], row[qids["right"]])
        topology_states.append(topology.state(*current))
        topology_edges.append(None if previous is None else
                              topology.transition(previous, current))
        previous = current
    paired = diagnostics["paired"]
    full_followed = (
        (position_error["left"] <= .001)
        & (orientation_error["left"] <= np.deg2rad(1.5))
        & (position_error["right"] <= .001)
        & (orientation_error["right"] <= np.deg2rad(1.5)))
    structural_state = sum(
        report.structural_crossing_count > 0 for report in topology_states)
    structural_edge = sum(
        report is not None and report.structural_crossing_count > 0
        for report in topology_edges)
    gripper_violation = sum(
        report.gripper_overlap_m > topology.config.gripper_overlap_limit_m
        for report in topology_states)
    gripper_edge_violation = sum(
        report is not None
        and report.gripper_overlap_m > topology.config.gripper_overlap_limit_m
        for report in topology_edges)
    gripper_frames = sum(
        report.gripper_overlap_count > 0 for report in topology_states)
    relaxed_frames = int(np.count_nonzero(
        (paired.left_tier != "exact") | (paired.right_tier != "exact")))
    payload = json.dumps(mount, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(payload)
    for array in (task.time_s, task.left_position_m, task.right_position_m,
                  qpos):
        digest.update(np.ascontiguousarray(array).tobytes())
    state_collision = int(sum(bool(classes) for classes in state_classes))
    edge_collision = int(sum(bool(classes) for classes in edge_classes))
    # A collision-free hold is not a successful mount search result.  Require
    # at least one strict followed row so zero-coverage artifacts cannot be
    # selected or later mistaken for a valid layout.
    valid = (not any((state_collision, edge_collision))
             and bool(np.any(full_followed)))
    count = len(task.time_s)
    quality = summarize_quality_arrays(
        position_error=position_error, orientation_error=orientation_error,
        joint_margins={side: np.interp(
            task.time_s, prepared.time_s,
            paired.selected_joint_limit_margin_rad[side])
            for side in ("left", "right")},
        singularity_margins={side: np.interp(
            task.time_s, prepared.time_s,
            paired.selected_singularity_margin[side])
            for side in ("left", "right")})
    return {
        "mount": mount,
        "status": "valid_selection" if valid else "rejected_full_audit",
        "audit_scope": "full_timeline",
        "source_row_count": count,
        "audited_source_rows": count,
        "audit_fingerprint": digest.hexdigest(),
        "base_distance_m": distance,
        "left_coverage": float(np.mean(full_followed)),
        "right_coverage": float(np.mean(full_followed)),
        "synchronous_pair_coverage": float(np.mean(full_followed)),
        "continuous_pair_coverage": float(np.mean(full_followed)),
        "pair_collision_frames": state_collision,
        "pair_edge_collision_frames": edge_collision,
        "structural_crossing_frames": int(structural_state),
        "structural_edge_crossing_frames": int(structural_edge),
        "gripper_overlap_violation_frames": int(gripper_violation),
        "gripper_overlap_edge_violation_frames": int(gripper_edge_violation),
        "gripper_overlap_frames": int(gripper_frames),
        "maximum_structural_crossing_m": max(
            report.maximum_structural_crossing_m
            for report in topology_states),
        "maximum_gripper_overlap_m": max(
            [report.gripper_overlap_m for report in topology_states]
            + [report.gripper_overlap_m for report in topology_edges
               if report is not None]),
        "longest_hold_frames": _longest_false_run(full_followed),
        "relaxed_tier_frames": relaxed_frames,
        **quality,
        "collision_audit_frames": int(np.count_nonzero(collision)),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    task = _registered_task()
    coarse_mounts = deterministic_pair_mounts(task, maximum=72)
    records = []
    serial = 0
    for stage, mount in enumerate(coarse_mounts, 1):
        record = evaluate_pair(
            task, mount, serial, uniform_count=8, global_seed_count=1,
            max_iterations=45, maximum_candidates=1,
            constrained_fallback_enabled=False)
        serial += 1; records.append(record)
        print("coarse", stage, len(coarse_mounts),
              record["synchronous_pair_coverage"],
              record["pair_collision_frames"],
              round(record["base_distance_m"], 3), flush=True)
    safe_coarse = sorted(
        (record for record in records if _sparse_safe(record)),
        key=rank_paired_mount_candidate)[:8]
    dense_seeds = []
    for record in safe_coarse:
        dense = evaluate_pair(
            task, record["mount"], serial, uniform_count=30,
            global_seed_count=6, max_iterations=100, maximum_candidates=3)
        serial += 1; records.append(dense); dense_seeds.append(dense)
        print("dense-seed", dense["synchronous_pair_coverage"],
              dense["pair_collision_frames"],
              dense["pair_edge_collision_frames"], flush=True)
    best_record = min(
        (record for record in dense_seeds if _sparse_safe(record)),
        key=rank_paired_mount_candidate)
    best = _sparse_mount(best_record)
    for side in ("left", "right"):
        local_records = [best_record]
        for mount in local_paired_refinements(
                best, side=side, xy_step_m=.05, yaw_step_deg=15.0):
            if not _valid_mount(mount):
                continue
            record = evaluate_pair(
                task, mount, serial, uniform_count=30,
                global_seed_count=6, max_iterations=100,
                maximum_candidates=3)
            serial += 1; records.append(record); local_records.append(record)
            print("refine", side, record["synchronous_pair_coverage"],
                  record["pair_collision_frames"],
                  round(record["base_distance_m"], 3), flush=True)
        best_record = min(
            (record for record in local_records if _sparse_safe(record)),
            key=rank_paired_mount_candidate)
        best = _sparse_mount(best_record)
    height_records = []
    for z in (.76, .81, .86):
        mount = {"xy": best["xy"], "yaw": best["yaw"],
                 "shared_base_z_m": z}
        record = evaluate_pair(
            task, mount, serial, uniform_count=40,
            global_seed_count=8, max_iterations=120,
            maximum_candidates=4)
        serial += 1; records.append(record); height_records.append(record)
        print("height", z, record["synchronous_pair_coverage"],
              record["pair_collision_frames"], flush=True)
    finalists = sorted(
        (record for record in height_records if _sparse_safe(record)),
        key=rank_paired_mount_candidate)[:2]
    full_records = []
    for finalist in finalists:
        record = evaluate_full_pair(task, finalist["mount"], serial)
        serial += 1; records.append(record); full_records.append(record)
        print("full-audit", record["status"],
              record["continuous_pair_coverage"],
              record["pair_collision_frames"],
              record["pair_edge_collision_frames"],
              record["structural_crossing_frames"], flush=True)
    selected = select_paired_mount(full_records)
    payload = {"robot": "piperx", "task": "fold_box",
               "search_method": "Seal Bag paired synchronous mount search",
               "selected_mount": selected, "records": records}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("selected", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
