"""Build and validate evidence for the multitask fixed-time mount study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.mount_topology import (
    MountTopologyConfig,
    MuJoCoMountTopologyChecker,
)
from factory_bimanual.multitask_fixed_time_study import (
    SHARD_SCHEMA,
    STUDY_MODES,
    discover_dual_hand_trajectories,
    validate_shard,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from scripts import render_factory_dual_piperx_fixed_time as fixed_runner
from scripts import run_piperx_multitask_fixed_time_mount_study as search_runner
from scripts.render_factory_dual_xarm6_se3_follow import (
    audit_bimanual_collisions,
    prepare_follow_targets,
)
from scripts.search_fold_box_piperx_paired_mount import _scene_mount_kwargs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/piperx_multitask_fixed_time_mount_study"
BUNDLE_SCHEMA = "piperx-multitask-fixed-time-bundle-v1"


def build_shard_arrays(*, mode, source_time_s, qpos, position_error_m,
                       orientation_error_rad, collision, edge_collision,
                       topology_valid, velocity_rad_s,
                       acceleration_rad_s2):
    """Assemble exact-source-time arrays while keeping dynamics separate."""
    if mode not in STUDY_MODES:
        raise ValueError("unsupported fixed-time mount mode")
    source_time = np.asarray(source_time_s, dtype=float)
    qpos = np.asarray(qpos, dtype=float)
    position = np.asarray(position_error_m, dtype=float)
    orientation = np.asarray(orientation_error_rad, dtype=float)
    count = len(source_time)
    if source_time.shape != (count,) or not np.all(np.diff(source_time) > 0):
        raise ValueError("source timestamps must be strictly increasing")
    if qpos.ndim != 2 or qpos.shape[0] != count:
        raise ValueError("qpos must have one row per source timestamp")
    if position.shape != (count, 2) or orientation.shape != (count, 2):
        raise ValueError("pose errors must have shape (frames, 2)")
    left = ((position[:, 0] <= 0.001 + 1e-12)
            & (orientation[:, 0] <= np.deg2rad(0.5) + 1e-12))
    right = ((position[:, 1] <= 0.001 + 1e-12)
             & (orientation[:, 1] <= np.deg2rad(0.5) + 1e-12))
    payload = {
        "schema": "piperx-multitask-fixed-time-shard-v1",
        "mode": mode,
        "source_time_s": source_time.copy(),
        "fixed_time_s": source_time.copy(),
        "retiming_applied": False,
        "qpos": qpos.copy(),
        "left_accept": left,
        "right_accept": right,
        "both_accept": left & right,
        "position_error_m": position.copy(),
        "orientation_error_rad": orientation.copy(),
        "collision": np.asarray(collision, dtype=bool).copy(),
        "edge_collision": np.asarray(edge_collision, dtype=bool).copy(),
        "topology_valid": np.asarray(topology_valid, dtype=bool).copy(),
        "velocity_rad_s": np.asarray(velocity_rad_s, dtype=float).copy(),
        "acceleration_rad_s2": np.asarray(
            acceleration_rad_s2, dtype=float).copy(),
    }
    aggregate_shard(payload)
    return payload


def _longest_false_run(values) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _joint_derivatives(model, qpos, time_s):
    addresses = []
    for side in ("left", "right"):
        for name in ROBOT_CONTRACTS["piperx"].prefixed_joint_names(side):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"missing PiperX joint {name}")
            addresses.append(model.jnt_qposadr[joint_id])
    q = np.asarray(qpos, dtype=float)[:, addresses]
    time_s = np.asarray(time_s, dtype=float)
    edge_order = 2 if len(time_s) >= 3 else 1
    velocity = np.gradient(q, time_s, axis=0, edge_order=edge_order)
    acceleration = np.gradient(
        velocity, time_s, axis=0, edge_order=edge_order)
    return velocity, acceleration


def _topology_valid(model, qpos):
    names = {side: {
        "joints": ROBOT_CONTRACTS["piperx"].prefixed_joint_names(side),
        "site": f"{side}_tcp",
    } for side in ("left", "right")}
    checker = MuJoCoMountTopologyChecker(
        model, mujoco.MjData(model), names,
        config=MountTopologyConfig(transition_steps=5))
    qids = {}
    for side in ("left", "right"):
        joint_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names[side]["joints"]]
        qids[side] = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
    valid = np.ones(len(qpos), dtype=bool)
    previous = None
    for index, row in enumerate(qpos):
        current = (row[qids["left"]], row[qids["right"]])
        state = checker.state(*current)
        edge = None if previous is None else checker.transition(previous, current)
        valid[index] = state.valid and (edge is None or edge.valid)
        previous = current
    return valid


def _checkpoint_path(root, spec, mode):
    return (Path(root) / "checkpoints" / spec.family.date
            / spec.family.task / spec.take / f"{mode}.json")


def solve_selected_shard(spec, mode, output_root=DEFAULT_OUTPUT, *,
                         source_prefix=None):
    """Run strict paired IK on the native timestamps of one selected mount."""
    checkpoint = _checkpoint_path(output_root, spec, mode)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (state.get("status") not in {"complete", "infeasible"}
            or not state.get("selected_mount")):
        raise ValueError(f"{spec.key}/{mode}: selected mount is unavailable")
    mount = state["selected_mount"]
    task, _registration = search_runner._load_registered_spec(spec)
    task = search_runner._prefix_task(task, source_prefix)
    shard_dir = (Path(output_root) / "shards" / spec.family.date
                 / spec.family.task / spec.take / mode)
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.family.date}_{spec.family.task}_{spec.take}_{mode}_fixed_time"
    scene = shard_dir / f"{stem}.scene.xml"
    separation = float(np.linalg.norm(
        np.asarray(mount["xy"]["left"], float)
        - np.asarray(mount["xy"]["right"], float)))
    build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], separation, scene,
        table_height_m=fixed_runner.TABLE_HEIGHT_M,
        mount_xy_m=mount["xy"], mount_yaw_deg=mount["yaw"],
        **_scene_mount_kwargs(task, mount))
    model = mujoco.MjModel.from_xml_path(str(scene))
    prepared, mapped = prepare_follow_targets(
        model, task, calibration=fixed_runner.load_locked_piperx_calibration())
    (qpos, actual, position_error, orientation_error, _strict,
     discontinuity, _planner_velocity, diagnostics) = \
        fixed_runner.solve_fixed_time_motion(
            model, prepared, mapped, solver_method="paired",
            solver_profile="default")
    source_time = np.asarray(prepared.time_s, dtype=float)
    source_time = source_time - source_time[0]
    velocity, acceleration = _joint_derivatives(model, qpos, source_time)
    _collision_union, state_classes, edge_classes = audit_bimanual_collisions(
        model, qpos, robot_name="piperx")
    collision = np.asarray([
        np.asarray(value).size > 0 for value in state_classes])
    edge_collision = np.asarray([
        np.asarray(value).size > 0 for value in edge_classes])
    topology_valid = _topology_valid(model, qpos)
    position = np.column_stack((
        position_error["left"], position_error["right"]))
    orientation = np.column_stack((
        orientation_error["left"], orientation_error["right"]))
    payload = build_shard_arrays(
        mode=mode, source_time_s=source_time, qpos=qpos,
        position_error_m=position, orientation_error_rad=orientation,
        collision=collision, edge_collision=edge_collision,
        topology_valid=topology_valid, velocity_rad_s=velocity,
        acceleration_rad_s2=acceleration)
    payload.update({
        "left_actual_tcp": np.asarray(actual["left"]),
        "right_actual_tcp": np.asarray(actual["right"]),
        "left_target_position_m": np.asarray(prepared.left_position_m),
        "right_target_position_m": np.asarray(prepared.right_position_m),
        "left_target_quaternion_wxyz": np.asarray(mapped["left"]),
        "right_target_quaternion_wxyz": np.asarray(mapped["right"]),
        "left_discontinuity": np.asarray(discontinuity["left"]),
        "right_discontinuity": np.asarray(discontinuity["right"]),
        "state_collision_classes": np.asarray([
            ";".join(map(str, value)) for value in state_classes], dtype=np.str_),
        "edge_collision_classes": np.asarray([
            ";".join(map(str, value)) for value in edge_classes], dtype=np.str_),
        "paired_failure_reason": np.asarray(
            diagnostics["paired"].failure_reason, dtype=np.str_),
    })
    trajectory = shard_dir / f"{stem}.trajectory.npz"
    np.savez_compressed(trajectory, **payload)
    validation_spec = spec
    if source_prefix is not None:
        from dataclasses import replace
        validation_spec = replace(spec, row_count=len(source_time))
    validate_shard(payload, validation_spec, mode)
    metrics = aggregate_shard(payload)
    summary = {
        "schema": "piperx-multitask-fixed-time-summary-v1",
        "trajectory": spec.key, "family": spec.family.key,
        "take": spec.take, "mode": mode,
        "timing_mode": "fixed_source_time",
        "retiming_applied": False,
        "source_sha256": spec.source_sha256,
        "mount": mount, "metrics": metrics,
        "dynamics": {
            "maximum_velocity_rad_s": float(np.max(np.abs(velocity))),
            "maximum_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
            "limits_passed": bool(
                np.max(np.abs(velocity)) <= 1.0 + 1e-12
                and np.max(np.abs(acceleration)) <= 4.0 + 1e-12),
        },
        "artifacts": {
            "trajectory_npz": {"path": str(trajectory),
                               "sha256": _sha256(trajectory)},
            "scene_xml": {"path": str(scene), "sha256": _sha256(scene)},
        },
    }
    summary_path = shard_dir / f"{stem}.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return summary_path


def aggregate_shard(payload):
    left = np.asarray(payload["left_accept"], dtype=bool)
    right = np.asarray(payload["right_accept"], dtype=bool)
    both = np.asarray(payload["both_accept"], dtype=bool)
    count = len(both)
    if not (left.shape == right.shape == both.shape == (count,)):
        raise ValueError("accept arrays must share one frame dimension")
    if not np.array_equal(both, left & right):
        raise ValueError("both_accept must equal the conjunction of both arms")
    position = np.asarray(payload["position_error_m"], dtype=float)
    orientation = np.asarray(payload["orientation_error_rad"], dtype=float)
    if position.shape != (count, 2) or orientation.shape != (count, 2):
        raise ValueError("pose error arrays must have shape (frames, 2)")
    collision = np.asarray(payload["collision"], dtype=bool)
    edge = np.asarray(payload["edge_collision"], dtype=bool)
    topology = np.asarray(payload["topology_valid"], dtype=bool)
    if collision.shape != (count,) or edge.shape != (count,) or topology.shape != (count,):
        raise ValueError("safety arrays must have one entry per frame")
    return {
        "source_frames": count,
        "left_accept_frames": int(np.count_nonzero(left)),
        "right_accept_frames": int(np.count_nonzero(right)),
        "both_accept_frames": int(np.count_nonzero(both)),
        "left_accept_coverage": float(np.mean(left)),
        "right_accept_coverage": float(np.mean(right)),
        "both_accept_coverage": float(np.mean(both)),
        "collision_frames": int(np.count_nonzero(collision)),
        "edge_collision_frames": int(np.count_nonzero(edge)),
        "topology_invalid_frames": int(np.count_nonzero(~topology)),
        "longest_hold_frames": _longest_false_run(both),
        "maximum_position_error_mm": float(np.max(position) * 1000.0),
        "maximum_orientation_error_deg": float(np.rad2deg(np.max(orientation))),
    }


def validate_manifest(manifest):
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unexpected multitask bundle schema")
    if (manifest.get("trajectory_count") != 27
            or manifest.get("family_count") != 12
            or manifest.get("mode_count") != 4):
        raise ValueError("bundle must describe 27 trajectories and four modes")
    shards = manifest.get("shards", [])
    if len(shards) != 108:
        raise ValueError("bundle must contain exactly 108 shards")
    pairs = {(item.get("trajectory"), item.get("mode")) for item in shards}
    if len(pairs) != 108:
        raise ValueError("bundle contains duplicate trajectory/mode shards")
    if {mode for _, mode in pairs} != set(STUDY_MODES):
        raise ValueError("bundle must contain every published mount mode")
    return manifest


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _resolve_artifact(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_bundle_artifacts(manifest_path, *, require_complete=True):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_complete:
        validate_manifest(manifest)
    specs = {item.key: item for item in discover_dual_hand_trajectories(
        ROOT / "data/factory")}
    for shard in manifest.get("shards", []):
        spec = specs[shard["trajectory"]]
        summary_path = _resolve_artifact(shard["summary_json"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (manifest.get("status") == "partial"
                and summary["metrics"]["source_frames"] != spec.row_count):
            from dataclasses import replace
            spec = replace(
                spec, row_count=int(summary["metrics"]["source_frames"]))
        trajectory = _resolve_artifact(
            summary["artifacts"]["trajectory_npz"]["path"])
        if _sha256(trajectory) != summary["artifacts"]["trajectory_npz"]["sha256"]:
            raise ValueError(f"{shard['trajectory']}/{shard['mode']}: trajectory hash mismatch")
        with np.load(trajectory, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        payload["schema"] = str(payload["schema"].item())
        payload["mode"] = str(payload["mode"].item())
        payload["retiming_applied"] = bool(payload["retiming_applied"].item())
        validate_shard(payload, spec, shard["mode"])
        recomputed = aggregate_shard(payload)
        if recomputed != summary["metrics"]:
            raise ValueError(f"{shard['trajectory']}/{shard['mode']}: summary drift")
    return manifest


def build_bundle(output_root=DEFAULT_OUTPUT, *, trajectory=None, mode=None,
                 source_prefix=None, require_complete=True):
    output_root = Path(output_root)
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    jobs = [(spec, job_mode) for spec in specs
            for job_mode in STUDY_MODES]
    if trajectory:
        jobs = [job for job in jobs if job[0].key == trajectory]
    if mode:
        jobs = [job for job in jobs if job[1] == mode]
    if not jobs:
        raise ValueError("no bundle jobs matched the requested filters")
    shards = []
    aggregate_rows = []
    for index, (spec, job_mode) in enumerate(jobs, 1):
        checkpoint = _checkpoint_path(output_root, spec, job_mode)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"missing search checkpoint: {spec.key}/{job_mode}")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("status") not in {"complete", "infeasible"}:
            raise ValueError(
                f"unfinished search checkpoint: {spec.key}/{job_mode}")
        shard_dir = (output_root / "shards" / spec.family.date
                     / spec.family.task / spec.take / job_mode)
        summaries = list(shard_dir.glob("*.summary.json"))
        summary_path = (summaries[0] if len(summaries) == 1 else
                        solve_selected_shard(
                            spec, job_mode, output_root,
                            source_prefix=source_prefix))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_link = str(summary_path.resolve().relative_to(ROOT.resolve()))
        shards.append({
            "trajectory": spec.key, "family": spec.family.key,
            "take": spec.take, "mode": job_mode,
            "search_status": state["status"],
            "summary_json": summary_link.replace("\\", "/"),
        })
        aggregate_rows.append({
            "trajectory": spec.key, "family": spec.family.key,
            "take": spec.take, "mode": job_mode,
            "search_status": state["status"],
            **summary["metrics"], **summary["dynamics"],
        })
        partial = {
            "schema": BUNDLE_SCHEMA,
            "trajectory_count": len(specs),
            "family_count": len({spec.family.key for spec in specs}),
            "mode_count": len(STUDY_MODES),
            "status": "running", "completed_shards": index,
            "shards": shards,
        }
        _atomic_json(output_root / "bundle_manifest.partial.json", partial)
    aggregate_path = output_root / "aggregate.csv"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "trajectory_count": len(specs),
        "family_count": len({spec.family.key for spec in specs}),
        "mode_count": len(STUDY_MODES),
        "status": "complete" if len(shards) == 108 else "partial",
        "retiming_applied": False,
        "aggregate_csv": str(aggregate_path.resolve().relative_to(
            ROOT.resolve())).replace("\\", "/"),
        "shards": shards,
    }
    if require_complete:
        validate_manifest(manifest)
    manifest_path = output_root / "bundle_manifest.json"
    _atomic_json(manifest_path, manifest)
    validate_bundle_artifacts(
        manifest_path, require_complete=require_complete)
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--trajectory")
    parser.add_argument("--mode", choices=STUDY_MODES)
    parser.add_argument("--source-prefix", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.output / "bundle_manifest.json"
    if args.validate_only:
        validate_bundle_artifacts(
            manifest_path, require_complete=not args.allow_partial)
        print(manifest_path)
        return
    print(build_bundle(
        args.output, trajectory=args.trajectory, mode=args.mode,
        source_prefix=args.source_prefix,
        require_complete=not args.allow_partial))


if __name__ == "__main__":
    main()
