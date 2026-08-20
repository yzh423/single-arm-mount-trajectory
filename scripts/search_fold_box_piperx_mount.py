"""Deterministic upright mount search for dual-PiperX Fold Box following."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig, MuJoCoCandidateGenerator,
)
from factory_bimanual.mujoco_collision_adapter import MuJoCoPairedCollisionChecker
from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.source_data import load_factory_task
from scripts.render_factory_dual_xarm6_fold_box import (
    CSV, bounded_smooth_pose_series, mapped_quaternions_at_joint_midpoint,
    subset,
)
from factory_bimanual.quaternion_trajectory import reconstruct_held_quaternions


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/factory_bimanual/fold_box_dual_piperx/fold_box_piperx_mount_search.json"
BASE_Z_M = .87
TABLE_HEIGHT_M = .75
ROBOT = "piperx"


def layered_sample_indices(frame_count, positions, *, uniform_count=14):
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (frame_count, 3) or frame_count < 2 or uniform_count < 2:
        raise ValueError("invalid trajectory or sample count")
    uniform = np.linspace(0, frame_count - 1, min(frame_count, uniform_count))
    extrema = [0, frame_count - 1]
    for axis in range(3):
        extrema.extend((int(np.argmin(positions[:, axis])),
                        int(np.argmax(positions[:, axis]))))
    return np.unique(np.r_[np.rint(uniform).astype(int), extrema]).astype(int)


def _normalize_yaw(degrees):
    return float((degrees + 180.0) % 360.0 - 180.0)


def coarse_mount_candidates(target_center_xy):
    center = np.asarray(target_center_xy, dtype=float)
    records = []
    for radius in (.26, .34, .42):
        for angle_deg in np.arange(0.0, 360.0, 45.0):
            angle = np.deg2rad(angle_deg)
            xy = center + radius * np.asarray((np.cos(angle), np.sin(angle)))
            if abs(xy[0]) > .825 or abs(xy[1]) > .625:
                continue
            yaw = np.rad2deg(np.arctan2(center[1] - xy[1], center[0] - xy[0]))
            records.append({
                "xy": [float(xy[0]), float(xy[1])],
                "yaw_deg": _normalize_yaw(yaw),
                "base_z_m": BASE_Z_M, "roll_deg": 0.0, "pitch_deg": 0.0,
            })
    return records


def rank_mount_candidate(record):
    return (
        -float(record["coverage"]),
        int(record["collision_frames"]),
        float(record["required_time_s"]),
        -float(record["p10_singularity_margin"]),
        float(record["mean_pose_error"]),
    )


def rank_paired_mount_candidate(record):
    """Rank only after hard gates; following quality cannot buy safety."""
    return (
        -float(record.get("continuous_pair_coverage",
                          record["synchronous_pair_coverage"])),
        int(record.get("longest_hold_frames", 0)),
        int(record.get("relaxed_tier_frames", 0)),
        float(record.get("maximum_structural_crossing_m", 0.0)),
        float(record.get("maximum_gripper_overlap_m", 0.0)),
        -min(float(record["left_coverage"]), float(record["right_coverage"])),
        float(record["mean_pair_pose_error"]),
        -float(record["p10_pair_singularity_margin"]),
        -float(record.get("base_distance_m", 0.0)),
    )


def _is_full_audited_safe(record):
    expected_rows = int(record.get("source_row_count", 0))
    fingerprint = str(record.get("audit_fingerprint", ""))
    zero_fields = (
        "pair_collision_frames", "pair_edge_collision_frames",
    )
    return (
        record.get("status") == "valid_selection"
        and record.get("audit_scope") == "full_timeline"
        and expected_rows > 0
        and int(record.get("audited_source_rows", -1)) == expected_rows
        and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
        and all(int(record.get(field, -1)) == 0 for field in zero_fields))


def select_paired_mount(records):
    """Return the best mount backed by a complete collision audit."""
    if not records:
        raise ValueError("paired mount records may not be empty")
    eligible = [record for record in records if _is_full_audited_safe(record)]
    if not eligible:
        raise RuntimeError("no full-audited collision-free paired mount candidate")
    best = min(eligible, key=rank_paired_mount_candidate)
    result = dict(best["mount"])
    result["selection_method"] = (
        "Seal Bag method: paired strict collision-free synchronous 6D IK "
        "coarse search followed by local XY/yaw/shared-height refinement")
    result["paired_sparse_metrics"] = {
        key: value for key, value in best.items() if key != "mount"}
    result["full_audit_metrics"] = dict(result["paired_sparse_metrics"])
    return result


def local_paired_refinements(mount, *, side, xy_step_m=.05,
                             yaw_step_deg=15.0):
    """Refine one base while holding its partner and shared height fixed."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    records = []
    for dx in (-xy_step_m, 0.0, xy_step_m):
        for dy in (-xy_step_m, 0.0, xy_step_m):
            for dyaw in (-yaw_step_deg, 0.0, yaw_step_deg):
                result = {
                    "xy": {arm: list(mount["xy"][arm])
                           for arm in ("left", "right")},
                    "yaw": {arm: float(mount["yaw"][arm])
                            for arm in ("left", "right")},
                    "shared_base_z_m": float(mount["shared_base_z_m"]),
                }
                result["xy"][side][0] += float(dx)
                result["xy"][side][1] += float(dy)
                result["yaw"][side] = _normalize_yaw(
                    result["yaw"][side] + float(dyaw))
                records.append(result)
    return records


def select_mount_pair(left_records, right_records, *, minimum_separation_m=.18):
    best = None
    for left in left_records:
        for right in right_records:
            if not np.isclose(float(left.get("base_z_m", BASE_Z_M)),
                              float(right.get("base_z_m", BASE_Z_M))):
                continue
            distance = float(np.linalg.norm(
                np.asarray(left["xy"]) - np.asarray(right["xy"])))
            if distance < minimum_separation_m:
                continue
            score = (
                -min(float(left["coverage"]), float(right["coverage"])),
                int(left["collision_frames"]) + int(right["collision_frames"]),
                float(left["required_time_s"]) + float(right["required_time_s"]),
                -min(float(left["p10_singularity_margin"]),
                     float(right["p10_singularity_margin"])),
                float(left["mean_pose_error"]) + float(right["mean_pose_error"]),
            )
            if best is None or score < best[0]:
                best = (score, left, right, distance)
    if best is None:
        raise RuntimeError("no non-overlapping PiperX mount pair")
    _, left, right, distance = best
    return {
        "xy": {"left": left["xy"], "right": right["xy"]},
        "yaw": {"left": left["yaw_deg"], "right": right["yaw_deg"]},
        "shared_base_z_m": float(left.get("base_z_m", BASE_Z_M)),
        "base_distance_m": distance,
        "selection_method": "deterministic sparse strict-6D coarse plus local mount search",
        "sparse_metrics": {"left": left, "right": right},
    }


def _registered_task():
    source = load_factory_task(CSV, "fold_box", max_translation_jump_m=.07)
    points = np.vstack((source.left_position_m, source.right_position_m))
    translation = np.asarray((
        -points[:, 0].mean(), -points[:, 1].mean(),
        .90 - points[:, 2].min(),
    ))
    return register_task(source, RigidTaskRegistration(np.eye(3), translation))


def _refined_candidates(record):
    result = []
    for dx in (-.04, 0.0, .04):
        for dy in (-.04, 0.0, .04):
            for dyaw in (0.0,):
                xy = np.asarray(record["xy"]) + (dx, dy)
                if abs(xy[0]) > .825 or abs(xy[1]) > .625:
                    continue
                result.append({
                    "xy": xy.tolist(),
                    "yaw_deg": _normalize_yaw(record["yaw_deg"] + dyaw),
                    "base_z_m": BASE_Z_M, "roll_deg": 0.0,
                    "pitch_deg": 0.0,
                })
    return result


def _evaluate_side(
    task, side, mount, index, workdir, *, max_iterations=45,
    global_seed_count=1, uniform_count=14, position_tolerance_m=.001,
    orientation_tolerance_rad=np.deg2rad(1.5),
):
    other = "right" if side == "left" else "left"
    base_z = float(mount.get("base_z_m", BASE_Z_M))
    placeholder = [float(.76 if side == "left" else -.76), .55]
    xy = {side: tuple(mount["xy"]), other: tuple(placeholder)}
    yaw = {side: float(mount["yaw_deg"]), other: 0.0}
    distance = float(np.linalg.norm(np.asarray(xy["left"]) - np.asarray(xy["right"])))
    scene = workdir / f"{side}_{index:03d}.xml"
    build_same_model_scene(
        ROBOT_CONTRACTS[ROBOT], distance, scene,
        table_height_m=TABLE_HEIGHT_M, mount_xy_m=xy, mount_yaw_deg=yaw,
        mount_adapter_height_m=base_z - TABLE_HEIGHT_M)
    model = mujoco.MjModel.from_xml_path(str(scene))
    mapped = mapped_quaternions_at_joint_midpoint(model, task, robot_name=ROBOT)
    reconstructed = reconstruct_held_quaternions(task.time_s, mapped[side])
    smoothed_position, smoothed_quaternion = bounded_smooth_pose_series(
        getattr(task, f"{side}_position_m"), reconstructed.quaternion_wxyz,
        position_cap_m=.003, orientation_cap_rad=np.deg2rad(1.0),
        sigma_frames=1.0)
    indices = layered_sample_indices(
        len(task.time_s), smoothed_position, uniform_count=uniform_count)
    sampled = subset(replace(
        task, **{f"{side}_position_m": smoothed_position}), indices)
    setattr(sampled, f"{side}_quaternion_wxyz", smoothed_quaternion[indices])

    contract = ROBOT_CONTRACTS[ROBOT]
    names = {arm: {"joints": contract.prefixed_joint_names(arm),
                   "site": f"{arm}_tcp"} for arm in ("left", "right")}
    data = mujoco.MjData(model)
    generator = MuJoCoCandidateGenerator(
        model, data, contract, name_map={side: names[side]},
        config=CandidateGeneratorConfig(
            max_iterations=max_iterations, maximum_candidates=4,
            global_seed_count=global_seed_count,
            dedup_rad=np.deg2rad(1.0),
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad))
    checker = MuJoCoPairedCollisionChecker(
        model, data, names, transition_steps=3)
    successful = 0
    collision_frames = 0
    sigmas = []
    residuals = []
    required_time = 0.0
    previous_q = None
    previous_time = None
    original_contype = model.geom_contype.copy()
    original_conaffinity = model.geom_conaffinity.copy()
    failed_ik_rows = []
    collision_only_rows = []
    for row in range(len(indices)):
        # Dense PiperX triangle meshes make collision broadphase dominate every
        # DLS iteration.  Generate kinematic candidates without contacts, then
        # restore the exact model before accepting any candidate.
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        proposals = generator(model, contract, sampled, row, side)
        model.geom_contype[:] = original_contype
        model.geom_conaffinity[:] = original_conaffinity
        collision_free = [item for item in proposals
                          if checker.side_state(side, item.q).valid]
        collision_frames += int(bool(proposals) and not collision_free)
        if not collision_free:
            source_row = int(indices[row])
            (collision_only_rows if proposals else failed_ik_rows).append(source_row)
            previous_q = None
            previous_time = None
            continue
        item = max(collision_free, key=lambda value: (
            value.singularity_margin, value.joint_limit_margin_rad,
            -value.position_error_m - value.orientation_error_rad))
        successful += 1
        sigmas.append(float(item.singularity_margin))
        residuals.append(float(item.position_error_m + item.orientation_error_rad))
        if previous_q is not None:
            source_dt = float(sampled.time_s[row] - previous_time)
            edge_dt = float(np.max(np.abs(item.q - previous_q)) / 3.0)
            required_time += max(0.0, edge_dt - source_dt)
        previous_q = item.q.copy()
        previous_time = float(sampled.time_s[row])
    result = dict(mount)
    result.update(
        side=side, sampled_frames=len(indices),
        coverage=successful / len(indices), collision_frames=collision_frames,
        required_time_s=required_time,
        p10_singularity_margin=(float(np.percentile(sigmas, 10))
                                if sigmas else 0.0),
        mean_pose_error=(float(np.mean(residuals))
                         if residuals else float("inf")),
        sampled_source_rows=indices.tolist(),
        failed_ik_rows=failed_ik_rows,
        collision_only_rows=collision_only_rows,
        position_tolerance_m=float(position_tolerance_m),
        orientation_tolerance_deg=float(np.rad2deg(orientation_tolerance_rad)),
    )
    return result


def search(output_json=OUTPUT):
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    workdir = ROOT / ".tmp/piperx_fold_box_mount_search"
    workdir.mkdir(parents=True, exist_ok=True)
    task = _registered_task()
    all_records = {}
    selected_records = {}
    serial = 0
    for side in ("left", "right"):
        center = getattr(task, f"{side}_position_m")[:, :2].mean(axis=0)
        coarse = []
        for mount in coarse_mount_candidates(center):
            record = _evaluate_side(task, side, mount, serial, workdir)
            serial += 1
            coarse.append(record)
            print(side, "coarse", len(coarse), record["coverage"], mount["xy"],
                  mount["yaw_deg"], flush=True)
        coarse.sort(key=rank_mount_candidate)
        refined = []
        for mount in _refined_candidates(coarse[0]):
            record = _evaluate_side(task, side, mount, serial, workdir)
            serial += 1
            refined.append(record)
            print(side, "refine", len(refined), record["coverage"], mount["xy"],
                  mount["yaw_deg"], flush=True)
        records = coarse + refined
        records.sort(key=rank_mount_candidate)
        all_records[side] = records
        selected_records[side] = records[:10]
    selected = select_mount_pair(
        selected_records["left"], selected_records["right"])
    payload = {
        "robot": ROBOT, "task": "fold_box", "shared_base_z_m": BASE_Z_M,
        "source_csv": str(CSV), "selected_mount": selected,
        "candidates": all_records,
    }
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(selected, ensure_ascii=False), flush=True)
    return payload


def main():
    search()


if __name__ == "__main__":
    main()
