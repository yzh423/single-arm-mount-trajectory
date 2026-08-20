"""Bounded xArm6-style upright Mount search for PiperX fixed-time tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from factory_bimanual.fixed_time_run_contract import validate_synchronized_mount
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.staged_mount_search import (
    MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M,
    rank_full_fixed_time_mount,
    targeted_source_indices,
)
from scripts import render_factory_dual_piperx_fixed_time as runner
from scripts.render_factory_dual_xarm6_se3_follow import (
    audit_bimanual_collisions,
    prepare_follow_targets,
    solve_collision_safe_bimanual_method,
)


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".tmp/piperx_xarm6_style_mount_search"
SEARCH_VERSION = 4


def _mount_key(mount):
    payload = json.dumps(mount, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def xarm6_style_mount_candidates(*, maximum=32):
    """Scan independent upright XY/yaw and shared Z around the 0.702 m mount."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    raw = []
    yaw_pairs = ((-15., 15.), (0., 15.), (0., 30.), (15., 15.),
                 (15., 30.), (15., 45.), (30., 0.), (30., 30.))
    left_positions = [(x, y) for x in (-.42, -.35, -.28)
                      for y in (.18, .25, .32)]
    right_positions = [(x, y) for x in (-.40, -.30, -.20)
                       for y in (-.52, -.45, -.35, -.25, -.15)]
    for left_xy in left_positions:
        for right_xy in right_positions:
            separation = float(np.linalg.norm(
                np.asarray(right_xy) - np.asarray(left_xy)))
            if not (.40 <= separation <= .80):
                continue
            for left_yaw, right_yaw in yaw_pairs:
                for shared_z in (.79, .81, .90, 1.0):
                    mount = {
                        "xy": {
                            "left": list(left_xy),
                            "right": list(right_xy),
                        },
                        "yaw": {"left": left_yaw,
                                "right": right_yaw},
                        "shared_base_z_m": shared_z,
                        "roll": {"left": 0.0, "right": 0.0},
                        "pitch": {"left": 0.0, "right": 0.0},
                    }
                    validate_synchronized_mount(
                        mount,
                        minimum_separation_m=
                        MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M)
                    raw.append(mount)
    raw.sort(key=lambda mount: (
        np.linalg.norm(np.asarray(mount["xy"]["left"]) -
                       np.asarray([-.35, .25])),
        np.linalg.norm(np.asarray(mount["xy"]["right"]) -
                       np.asarray([-.30, -.45])),
        abs(validate_synchronized_mount(
            mount, minimum_separation_m=
            MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M) - .702),
        abs(mount["yaw"]["left"] - 15.) +
        abs(mount["yaw"]["right"] - 15.),
        abs(mount["shared_base_z_m"] - .81),
        _mount_key(mount),
    ))
    if len(raw) <= maximum:
        return raw
    features = np.asarray([[
        mount["xy"]["left"][0] / .2,
        mount["xy"]["left"][1] / .2,
        mount["xy"]["right"][0] / .2,
        mount["xy"]["right"][1] / .25,
        np.sin(np.deg2rad(mount["yaw"]["left"])),
        np.cos(np.deg2rad(mount["yaw"]["left"])),
        np.sin(np.deg2rad(mount["yaw"]["right"])),
        np.cos(np.deg2rad(mount["yaw"]["right"])),
        (mount["shared_base_z_m"] - .79) / .21,
    ] for mount in raw])
    chosen = [0]
    nearest = np.sum((features - features[0]) ** 2, axis=1)
    nearest[0] = -np.inf
    while len(chosen) < maximum:
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(
            nearest, np.sum((features - features[index]) ** 2, axis=1))
        nearest[chosen] = -np.inf
    return [raw[index] for index in chosen]


def local_mount_candidates(center, *, maximum=16):
    """Refine left/right XY, yaw and shared Z around one coarse finalist."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    center = json.loads(json.dumps(center))
    offsets = ((0., 0.), (-.05, 0.), (.05, 0.),
               (0., -.05), (0., .05))
    raw = {}
    for left_offset in offsets:
        for right_offset in offsets:
            left_xy = [round(center["xy"]["left"][axis] +
                             left_offset[axis], 6) for axis in range(2)]
            right_xy = [round(center["xy"]["right"][axis] +
                              right_offset[axis], 6) for axis in range(2)]
            distance = float(np.linalg.norm(
                np.asarray(right_xy) - np.asarray(left_xy)))
            if not (.40 <= distance <= .80):
                continue
            for left_delta in (-15., 0., 15.):
                for right_delta in (-15., 0., 15.):
                    for z_delta in (-.05, 0., .05):
                        z = round(float(np.clip(
                            center["shared_base_z_m"] + z_delta,
                            .79, 1.0)), 6)
                        mount = {
                            "xy": {"left": left_xy,
                                   "right": right_xy},
                            "yaw": {
                                "left": center["yaw"]["left"] + left_delta,
                                "right": center["yaw"]["right"] + right_delta,
                            },
                            "shared_base_z_m": z,
                            "roll": {"left": 0.0, "right": 0.0},
                            "pitch": {"left": 0.0, "right": 0.0},
                        }
                        raw[_mount_key(mount)] = mount
    raw[_mount_key(center)] = center
    center_key = _mount_key(center)
    ordered = sorted(raw.values(), key=lambda item: (
        0 if _mount_key(item) == center_key else 1,
        sum(np.linalg.norm(
            np.asarray(item["xy"][side]) -
            np.asarray(center["xy"][side])) for side in ("left", "right")),
        abs(item["yaw"]["left"] - center["yaw"]["left"]) +
        abs(item["yaw"]["right"] - center["yaw"]["right"]),
        abs(item["shared_base_z_m"] - center["shared_base_z_m"]),
        _mount_key(item),
    ))
    if len(ordered) <= maximum:
        return ordered
    features = np.asarray([[
        (item["xy"]["left"][0] - center["xy"]["left"][0]) / .05,
        (item["xy"]["left"][1] - center["xy"]["left"][1]) / .05,
        (item["xy"]["right"][0] - center["xy"]["right"][0]) / .05,
        (item["xy"]["right"][1] - center["xy"]["right"][1]) / .05,
        (item["yaw"]["left"] - center["yaw"]["left"]) / 15.,
        (item["yaw"]["right"] - center["yaw"]["right"]) / 15.,
        (item["shared_base_z_m"] - center["shared_base_z_m"]) / .05,
    ] for item in ordered])
    chosen = [0]
    nearest = np.sum((features - features[0]) ** 2, axis=1)
    nearest[0] = -np.inf
    while len(chosen) < maximum:
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(
            nearest, np.sum((features - features[index]) ** 2, axis=1))
        nearest[chosen] = -np.inf
    return [ordered[index] for index in chosen]


def select_stage_finalists(records, *, stage, maximum):
    eligible = [record for record in records
                if record.get("stage") == stage
                and record.get("status") == "complete"
                and int(record.get("collision_frame_count", -1)) == 0
                and int(record.get(
                    "clearance_violation_frame_count", 0)) == 0
                and float(record.get("base_distance_m", 0.0)) >=
                MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M]
    eligible.sort(key=rank_full_fixed_time_mount)
    output = []
    used = set()
    for record in eligible:
        key = record["mount_key"]
        if key in used:
            continue
        used.add(key)
        output.append(record)
        if len(output) >= maximum:
            break
    return output


def _subset(task, indices):
    total = len(task.time_s)
    values = {}
    for name, value in task.__dict__.items():
        array = np.asarray(value)
        values[name] = array[indices].copy() if (
            array.ndim and len(array) == total) else value
    values["source_row_indices"] = np.asarray(indices, dtype=int)
    return SimpleNamespace(**values)


def _checkpoint_path(task_name):
    spec = runner.task_spec(task_name)
    return (spec.report_directory /
            f"{task_name}_piperx_xarm6_style_mount_search.json")


def _load_failure_mask(task_name, row_count):
    spec = runner.task_spec(task_name)
    path = spec.report_directory / f"{spec.output_stem}.trajectory.npz"
    if not path.exists():
        return np.ones(row_count, dtype=bool)
    with np.load(path, allow_pickle=True) as payload:
        values = ~np.asarray(payload["synchronous_success"], dtype=bool)
    return values if values.shape == (row_count,) else np.ones(row_count, bool)


def _evaluate_sampled(task_name, mount, indices, *, stage, serial):
    spec = runner.task_spec(task_name)
    task, _ = runner._registered_task(spec)
    separation = validate_synchronized_mount(
        mount, minimum_separation_m=
        MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M)
    WORK.mkdir(parents=True, exist_ok=True)
    scene = WORK / f"{task_name}_{stage}_{serial:04d}.scene.xml"
    runner.build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], separation, scene,
        table_height_m=runner.TABLE_HEIGHT_M,
        mount_xy_m=mount["xy"], mount_yaw_deg=mount["yaw"],
        mount_adapter_height_m=(
            float(mount["shared_base_z_m"]) - runner.TABLE_HEIGHT_M),
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    prepared, mapped = prepare_follow_targets(
        model, task, calibration=runner.load_locked_piperx_calibration())
    window = _subset(prepared, np.asarray(indices, dtype=int))
    mapped_window = {side: np.asarray(mapped[side])[indices]
                     for side in ("left", "right")}
    sparse_stage = stage in ("sparse", "refine_sparse")
    (qpos, _actual, position_error, orientation_error, strict,
     discontinuity, velocity, diagnostics) = \
        solve_collision_safe_bimanual_method(
            model, window, mapped_window,
            candidate_iterations=(35 if sparse_stage else 60),
            velocity_limit_rad_s=runner.VELOCITY_LIMIT_RAD_S,
            robot_name="piperx",
            global_seed_count=(0 if sparse_stage else 2),
            maximum_candidates_per_tier=(3 if sparse_stage else 8),
            constrained_fallback_enabled=not sparse_stage,
            strict_pose_only=True,
            stratified_refresh_interval=(4 if sparse_stage else 10))
    paired = diagnostics["paired"]
    collision, _, _ = audit_bimanual_collisions(
        model, qpos, robot_name="piperx")
    executed_synchronous = (
        strict["left"] & strict["right"] & paired.followed & ~collision)
    safe_6d = paired.target_state_safe
    failure = ~safe_6d
    changes = np.diff(np.r_[False, failure, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    longest_failure = int(max(
        (stop-start for start, stop in zip(starts, stops)), default=0))
    joint_margin = np.r_[
        paired.selected_joint_limit_margin_rad["left"],
        paired.selected_joint_limit_margin_rad["right"]]
    singularity = np.r_[
        paired.selected_singularity_margin["left"],
        paired.selected_singularity_margin["right"]]
    finite_joint_margin = joint_margin[np.isfinite(joint_margin)]
    finite_singularity = singularity[np.isfinite(singularity)]
    joint_travel = float(
        np.sum(np.abs(np.diff(paired.left_q, axis=0))) +
        np.sum(np.abs(np.diff(paired.right_q, axis=0))))
    clearance_violations = int(
        np.sum(paired.state_clearance_m < .015) +
        np.sum(paired.swept_clearance_m < .015))
    return {
        "stage": stage,
        "status": "complete",
        "task": task_name,
        "mount": mount,
        "mount_key": _mount_key(mount),
        "sampled_source_rows": np.asarray(indices, int).tolist(),
        "sampled_frame_count": int(len(indices)),
        "synchronous_strict_coverage": float(safe_6d.mean()),
        "collision_frame_count": int(collision.sum()),
        "cannot_follow_frame_count": int((~safe_6d).sum()),
        "longest_failure_run_frames": longest_failure,
        "connectable_safe_branch_ratio": float(
            paired.target_connectable.mean()),
        "executed_fixed_time_coverage": float(
            executed_synchronous.mean()),
        "minimum_clearance_m": float(paired.minimum_clearance_m),
        "clearance_violation_frame_count": clearance_violations,
        "clearance_limiting_pair": paired.limiting_clearance_pair,
        "minimum_singularity_margin": (
            float(np.min(finite_singularity))
            if len(finite_singularity) else 0.0),
        "minimum_joint_limit_margin_rad": (
            float(np.min(finite_joint_margin))
            if len(finite_joint_margin) else 0.0),
        "joint_travel_rad": joint_travel,
        "required_retime_frame_count": int(np.sum(
            paired.failure_reason == "dynamic_limit")),
        "position_mean_mm": float(500. * (
            np.mean(position_error["left"]) +
            np.mean(position_error["right"]))),
        "orientation_mean_deg": float(.5 * np.rad2deg(
            np.mean(orientation_error["left"]) +
            np.mean(orientation_error["right"]))),
        "left_strict_coverage": float(strict["left"].mean()),
        "right_strict_coverage": float(strict["right"].mean()),
        "discontinuity_frame_count": int(
            discontinuity["left"].sum() + discontinuity["right"].sum()),
        "velocity_infeasible_frame_count": int(
            velocity["left"].sum() + velocity["right"].sum()),
        "base_distance_m": separation,
        "failure_reason_counts": {
            str(reason): int(np.sum(paired.failure_reason == reason))
            for reason in np.unique(paired.failure_reason)
            if str(reason) != "ok"
        },
    }


def _write_checkpoint(task_name, records, full_finalists=None):
    path = _checkpoint_path(task_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "robot": "piperx",
        "task": task_name,
        "search_version": SEARCH_VERSION,
        "tool_frame_calibration": str(runner.CALIBRATION_PATH.resolve()),
        "search_method": (
            "xArm6-style upright Mount search with locked fixed tool "
            "calibration and PiperX joint collision/clearance planning"),
        "timing_mode": "fixed_source_time",
        "records": records,
        "full_finalists": full_finalists or [],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def screen_task(task_name):
    spec = runner.task_spec(task_name)
    task, _ = runner._registered_task(spec)
    failure = _load_failure_mask(task_name, len(task.time_s))
    sparse_indices = targeted_source_indices(
        len(task.time_s), failure, maximum=16)
    dense_indices = targeted_source_indices(
        len(task.time_s), failure, maximum=64)
    path = _checkpoint_path(task_name)
    records = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("search_version") == SEARCH_VERSION:
            records = payload.get("records", [])
    completed = {(record.get("stage"), record.get("mount_key"))
                 for record in records if record.get("status") == "complete"}
    candidates = xarm6_style_mount_candidates(maximum=24)
    for serial, mount in enumerate(candidates):
        key = _mount_key(mount)
        if ("sparse", key) in completed:
            continue
        record = _evaluate_sampled(
            task_name, mount, sparse_indices, stage="sparse", serial=serial)
        records.append(record)
        _write_checkpoint(task_name, records)
        print(task_name, "sparse", serial + 1, len(candidates),
              f"coverage={record['synchronous_strict_coverage']:.4f}",
              f"collision={record['collision_frame_count']}", flush=True)
    sparse = select_stage_finalists(records, stage="sparse", maximum=4)
    for serial, finalist in enumerate(sparse):
        mount = finalist["mount"]
        key = finalist["mount_key"]
        if ("dense", key) in completed:
            continue
        record = _evaluate_sampled(
            task_name, mount, dense_indices, stage="dense", serial=100 + serial)
        records.append(record)
        _write_checkpoint(task_name, records)
        print(task_name, "dense", serial + 1, len(sparse),
              f"coverage={record['synchronous_strict_coverage']:.4f}",
              f"collision={record['collision_frame_count']}", flush=True)
    dense = select_stage_finalists(records, stage="dense", maximum=2)
    _write_checkpoint(task_name, records, [record["mount"] for record in dense])
    print(json.dumps({"task": task_name, "full_finalists": dense},
                     ensure_ascii=False), flush=True)
    return dense


def refine_task(task_name):
    """Run a bounded local search around the best two coarse Mounts."""
    path = _checkpoint_path(task_name)
    if not path.exists():
        raise RuntimeError("coarse Mount search must complete before refinement")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    coarse = select_stage_finalists(records, stage="dense", maximum=2)
    if not coarse:
        raise RuntimeError("no collision-safe dense Mount finalist to refine")
    candidates = []
    seen = set()
    for record in coarse:
        for mount in local_mount_candidates(record["mount"], maximum=8):
            key = _mount_key(mount)
            if key not in seen:
                seen.add(key)
                candidates.append(mount)
    spec = runner.task_spec(task_name)
    task, _ = runner._registered_task(spec)
    failure = _load_failure_mask(task_name, len(task.time_s))
    sparse_indices = targeted_source_indices(
        len(task.time_s), failure, maximum=32)
    dense_indices = targeted_source_indices(
        len(task.time_s), failure, maximum=96)
    completed = {(record.get("stage"), record.get("mount_key"))
                 for record in records if record.get("status") == "complete"}
    for serial, mount in enumerate(candidates):
        key = _mount_key(mount)
        if ("refine_sparse", key) in completed:
            continue
        record = _evaluate_sampled(
            task_name, mount, sparse_indices,
            stage="refine_sparse", serial=200 + serial)
        records.append(record)
        _write_checkpoint(task_name, records)
        print(task_name, "refine_sparse", serial + 1, len(candidates),
              f"coverage={record['synchronous_strict_coverage']:.4f}",
              f"clearance={record['minimum_clearance_m']:.4f}", flush=True)
    refined = select_stage_finalists(
        records, stage="refine_sparse", maximum=3)
    for serial, finalist in enumerate(refined):
        mount = finalist["mount"]
        key = finalist["mount_key"]
        if ("refine_dense", key) in completed:
            continue
        record = _evaluate_sampled(
            task_name, mount, dense_indices,
            stage="refine_dense", serial=300 + serial)
        records.append(record)
        _write_checkpoint(task_name, records)
        print(task_name, "refine_dense", serial + 1, len(refined),
              f"coverage={record['synchronous_strict_coverage']:.4f}",
              f"clearance={record['minimum_clearance_m']:.4f}", flush=True)
    final = select_stage_finalists(
        records, stage="refine_dense", maximum=2)
    _write_checkpoint(task_name, records, [record["mount"] for record in final])
    print(json.dumps({"task": task_name, "full_finalists": final},
                     ensure_ascii=False), flush=True)
    return final


def refine_again_task(task_name):
    """Expand once more around the first local optimum for difficult tasks."""
    path = _checkpoint_path(task_name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    centers = select_stage_finalists(
        records, stage="refine_dense", maximum=2)
    if not centers:
        raise RuntimeError("first local refinement must complete")
    candidates = []
    seen = set()
    for record in centers:
        for mount in local_mount_candidates(record["mount"], maximum=8):
            key = _mount_key(mount)
            if key not in seen:
                seen.add(key)
                candidates.append(mount)
    spec = runner.task_spec(task_name)
    task, _ = runner._registered_task(spec)
    failure = _load_failure_mask(task_name, len(task.time_s))
    sparse_indices = targeted_source_indices(
        len(task.time_s), failure, maximum=48)
    dense_indices = targeted_source_indices(
        len(task.time_s), failure, maximum=128)
    completed = {(record.get("stage"), record.get("mount_key"))
                 for record in records if record.get("status") == "complete"}
    for serial, mount in enumerate(candidates):
        key = _mount_key(mount)
        if ("refine2_sparse", key) in completed:
            continue
        record = _evaluate_sampled(
            task_name, mount, sparse_indices,
            stage="refine_sparse", serial=500 + serial)
        record["stage"] = "refine2_sparse"
        records.append(record)
        _write_checkpoint(task_name, records)
        print(task_name, "refine2_sparse", serial + 1, len(candidates),
              f"coverage={record['synchronous_strict_coverage']:.4f}",
              f"connected={record['connectable_safe_branch_ratio']:.4f}",
              flush=True)
    refined = select_stage_finalists(
        records, stage="refine2_sparse", maximum=3)
    for serial, finalist in enumerate(refined):
        key = finalist["mount_key"]
        if ("refine2_dense", key) in completed:
            continue
        record = _evaluate_sampled(
            task_name, finalist["mount"], dense_indices,
            stage="refine2_dense", serial=600 + serial)
        records.append(record)
        _write_checkpoint(task_name, records)
        print(task_name, "refine2_dense", serial + 1, len(refined),
              f"coverage={record['synchronous_strict_coverage']:.4f}",
              f"connected={record['connectable_safe_branch_ratio']:.4f}",
              flush=True)
    final = select_stage_finalists(
        records, stage="refine2_dense", maximum=2)
    _write_checkpoint(task_name, records, [record["mount"] for record in final])
    print(json.dumps({"task": task_name, "full_finalists": final},
                     ensure_ascii=False), flush=True)
    return final


def validate_task_finalists(task_name):
    """Recheck refined Mounts with eight strict candidates per arm."""
    path = _checkpoint_path(task_name)
    if not path.exists():
        raise RuntimeError("Mount search checkpoint does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    finalists = select_stage_finalists(
        records, stage="refine_dense", maximum=2)
    if not finalists:
        finalists = select_stage_finalists(
            records, stage="dense", maximum=2)
    spec = runner.task_spec(task_name)
    task, _ = runner._registered_task(spec)
    indices = targeted_source_indices(
        len(task.time_s), _load_failure_mask(task_name, len(task.time_s)),
        maximum=192)
    completed = {(record.get("stage"), record.get("mount_key"))
                 for record in records if record.get("status") == "complete"}
    for serial, finalist in enumerate(finalists):
        key = finalist["mount_key"]
        if ("validation", key) in completed:
            continue
        record = _evaluate_sampled(
            task_name, finalist["mount"], indices,
            stage="validation", serial=400 + serial)
        records.append(record)
        _write_checkpoint(task_name, records)
        print(task_name, "validation", serial + 1, len(finalists),
              f"coverage={record['synchronous_strict_coverage']:.4f}",
              f"clearance={record['minimum_clearance_m']:.4f}", flush=True)
    final = select_stage_finalists(records, stage="validation", maximum=2)
    _write_checkpoint(task_name, records, [record["mount"] for record in final])
    print(json.dumps({"task": task_name, "full_finalists": final},
                     ensure_ascii=False), flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("fold_box", "seal_bag"))
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refine2", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if sum((args.refine, args.refine2, args.validate)) > 1:
        parser.error("refinement/validation modes are mutually exclusive")
    if args.validate:
        validate_task_finalists(args.task)
    elif args.refine2:
        refine_again_task(args.task)
    elif args.refine:
        refine_task(args.task)
    else:
        screen_task(args.task)


if __name__ == "__main__":
    main()
