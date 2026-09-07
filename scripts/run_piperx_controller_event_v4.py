"""Solve auditable raw PiperX targets on synchronized controller event time."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.fixed_time_tracking import fixed_time_derivatives
from factory_bimanual.multitask_fixed_time_study import (
    STUDY_MODES,
    discover_dual_hand_trajectories,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from scripts import build_piperx_multitask_fixed_time_bundle as bundle
from scripts import render_factory_dual_piperx_fixed_time as fixed_runner
from scripts import run_piperx_multitask_fixed_time_mount_study as study
from scripts.render_factory_dual_xarm6_se3_follow import (
    audit_bimanual_collisions,
)
from scripts.search_fold_box_piperx_paired_mount import _scene_mount_kwargs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "reports/piperx_multitask_fixed_time_mount_study"
DEFAULT_OUTPUT = ROOT / "reports/piperx_controller_event_v4"
EVENT_SOLVER_PROTOCOL = "piperx-controller-event-raw-fixed-time-v4"


def validate_enforced_dynamics(
        velocity_rad_s, acceleration_rad_s2, *,
        velocity_limit_rad_s=3.0, acceleration_limit_rad_s2=5.0):
    """Reject a purported dynamics-enforced result that exceeds its gates."""
    velocity = np.asarray(velocity_rad_s, dtype=float)
    acceleration = np.asarray(acceleration_rad_s2, dtype=float)
    if (np.any(~np.isfinite(velocity))
            or np.any(~np.isfinite(acceleration))):
        raise ValueError("fixed-time dynamics evidence must be finite")
    maximum_velocity = float(np.max(np.abs(velocity), initial=0.0))
    maximum_acceleration = float(
        np.max(np.abs(acceleration), initial=0.0))
    tolerance = 1e-9
    if maximum_velocity > float(velocity_limit_rad_s) + tolerance:
        raise ValueError(
            "dynamics-enforced result exceeds the velocity limit")
    if (maximum_acceleration
            > float(acceleration_limit_rad_s2) + tolerance):
        raise ValueError(
            "dynamics-enforced result exceeds the acceleration limit")
    return True


def validate_zero_safety_evidence(payload):
    """Require every saved state, swept edge, and mount topology to be safe."""
    for key in ("collision", "edge_collision"):
        values = np.asarray(payload[key], dtype=bool)
        if np.any(values):
            raise ValueError(f"event shard contains {key} violations")
    topology = np.asarray(payload["topology_valid"], dtype=bool)
    if not np.all(topology):
        raise ValueError("event shard contains topology violations")
    return True


def validate_acceptance_evidence(payload):
    """Recompute the strict 1 mm / 0.5 degree paired acceptance masks."""
    position = np.asarray(payload["position_error_m"], dtype=float)
    orientation = np.asarray(payload["orientation_error_rad"], dtype=float)
    if (position.ndim != 2 or position.shape[1] != 2
            or orientation.shape != position.shape):
        raise ValueError("event shard pose errors must have shape (frames, 2)")
    left_valid = np.asarray(payload["left_source_valid"], dtype=bool)
    right_valid = np.asarray(payload["right_source_valid"], dtype=bool)
    expected_left = (
        left_valid
        & (position[:, 0] <= .001 + 1e-12)
        & (orientation[:, 0] <= np.deg2rad(.5) + 1e-12))
    expected_right = (
        right_valid
        & (position[:, 1] <= .001 + 1e-12)
        & (orientation[:, 1] <= np.deg2rad(.5) + 1e-12))
    if not np.array_equal(
            np.asarray(payload["left_accept"], dtype=bool), expected_left):
        raise ValueError("event shard left acceptance mask drift")
    if not np.array_equal(
            np.asarray(payload["right_accept"], dtype=bool), expected_right):
        raise ValueError("event shard right acceptance mask drift")
    if not np.array_equal(
            np.asarray(payload["both_accept"], dtype=bool),
            expected_left & expected_right):
        raise ValueError("event shard paired acceptance mask drift")
    return True


def event_validation_spec(spec, event_count):
    return replace(spec, row_count=int(event_count))


def validate_event_shard(payload, spec, mode, *, source_prefix=None):
    """Bind a v4 shard to paired controller frames in the source CSV."""
    if payload.get("schema") != bundle.SHARD_SCHEMA:
        raise ValueError("event shard schema mismatch")
    if payload.get("solver_protocol") != EVENT_SOLVER_PROTOCOL:
        raise ValueError("event shard solver protocol mismatch")
    if payload.get("mode") != mode:
        raise ValueError("event shard mode mismatch")
    if bool(payload.get("retiming_applied")):
        raise ValueError("event shard cannot apply retiming")
    source, _registration = study._load_registered_spec(
        spec, timing_mode="controller_updates")
    source, mapped, _target_audit = study.prepare_family_follow_targets(
        spec, source, apply_conditioning=False,
        apply_wrist_adaptation=False)
    count = (len(source.time_s) if source_prefix is None else
             min(len(source.time_s), max(2, int(source_prefix))))
    source = study._prefix_task(source, count)
    mapped = {
        side: np.asarray(mapped[side], dtype=float)[:count]
        for side in ("left", "right")}
    expected_time = np.asarray(source.time_s[:count], dtype=float)
    expected_time -= expected_time[0]
    if not np.array_equal(
            np.asarray(payload["source_poll_row_index"], dtype=int),
            np.asarray(source.source_row_index[:count], dtype=int)):
        raise ValueError("event shard source poll-row mapping drift")
    if int(payload["source_poll_row_count"]) != int(
            source.source_poll_row_count):
        raise ValueError("event shard source poll-row count drift")
    if not np.allclose(
            np.asarray(payload["source_time_s"], dtype=float),
            expected_time, rtol=0.0, atol=1e-9):
        raise ValueError("event shard controller receive timeline drift")
    if not np.array_equal(
            np.asarray(payload["fixed_time_s"], dtype=float),
            np.asarray(payload["source_time_s"], dtype=float)):
        raise ValueError("event shard fixed time differs from source events")
    for side in ("left", "right"):
        if not np.allclose(
                np.asarray(payload[f"{side}_target_position_m"], dtype=float),
                np.asarray(getattr(source, f"{side}_position_m"), dtype=float),
                rtol=0.0, atol=1e-12):
            raise ValueError(f"event shard {side} target position drift")
        if not np.allclose(
                np.asarray(payload[f"{side}_target_quaternion_wxyz"], dtype=float),
                mapped[side], rtol=0.0, atol=1e-12):
            raise ValueError(f"event shard {side} target orientation drift")
        if not np.array_equal(
                np.asarray(payload[f"{side}_source_valid"], dtype=bool),
                np.asarray(getattr(source, f"{side}_valid"), dtype=bool)):
            raise ValueError(f"event shard {side} source-valid mask drift")
    for key in ("qpos", "position_error_m", "orientation_error_rad",
                "collision", "edge_collision", "topology_valid"):
        if len(np.asarray(payload[key])) != count:
            raise ValueError(f"event shard {key} row count drift")
    validate_zero_safety_evidence(payload)
    validate_acceptance_evidence(payload)
    bundle._validate_formal_evidence(payload, count)
    bundle.aggregate_shard(payload)
    return True


def event_initializer_rows(
        selected_result, source_poll_rows, *, source_poll_row_count):
    """Map search probes expressed in poll rows onto controller events."""
    event_rows = np.asarray(source_poll_rows, dtype=int)
    if event_rows.ndim != 1 or not len(event_rows):
        raise ValueError("source_poll_rows must contain controller events")
    raw_rows = bundle._initializer_probe_rows(
        selected_result, source_count=int(source_poll_row_count))
    mapped = np.searchsorted(event_rows, raw_rows, side="left")
    mapped = np.clip(mapped, 0, len(event_rows) - 1)
    return np.unique(mapped).astype(int).tolist()


def _checkpoint(source_root, spec, mode):
    return (Path(source_root) / "checkpoints" / spec.family.date
            / spec.family.task / spec.take / f"{mode}.json")


def _joint_addresses(model):
    addresses = []
    for side in ("left", "right"):
        for name in ROBOT_CONTRACTS["piperx"].prefixed_joint_names(side):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"missing PiperX joint {name}")
            addresses.append(model.jnt_qposadr[joint_id])
    return np.asarray(addresses, dtype=int)


def _fingerprint(spec, mode, mount, *, source_prefix, enforce_dynamics):
    value = {
        "protocol": EVENT_SOLVER_PROTOCOL,
        "source_sha256": spec.source_sha256,
        "mode": mode,
        "mount": mount,
        "source_prefix": source_prefix,
        "timing_source": "paired_controller_receive",
        "target": "fixed_tool_raw_no_conditioning_no_wrist_adaptation",
        "retiming_applied": False,
        "position_tolerance_m": .001,
        "orientation_tolerance_deg": .5,
        "enforce_dynamics": bool(enforce_dynamics),
        "velocity_limit_rad_s": 3.0 if enforce_dynamics else None,
        "acceleration_limit_rad_s2": 5.0 if enforce_dynamics else None,
    }
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def solve_event_shard(
        spec, mode, *, source_root=DEFAULT_SOURCE, output=DEFAULT_OUTPUT,
        source_prefix=None, enforce_dynamics=False):
    checkpoint = _checkpoint(source_root, spec, mode)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    mount = state.get("selected_mount")
    if not mount:
        raise ValueError(f"{spec.key}/{mode}: selected mount is unavailable")
    task, _registration = study._load_registered_spec(
        spec, timing_mode="controller_updates")
    task, mapped, target_audit = study.prepare_family_follow_targets(
        spec, task, apply_conditioning=False,
        apply_wrist_adaptation=False)
    controller_event_rows_total = len(task.time_s)
    task = study._prefix_task(task, source_prefix)
    mapped = {
        side: np.asarray(mapped[side], dtype=float)[:len(task.time_s)]
        for side in ("left", "right")}
    shard_dir = (Path(output) / "shards" / spec.family.date
                 / spec.family.task / spec.take / mode)
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.family.date}_{spec.family.task}_{spec.take}_{mode}_event_v4"
    scene = shard_dir / f"{stem}.scene.xml"
    separation = float(np.linalg.norm(
        np.asarray(mount["xy"]["left"], dtype=float)
        - np.asarray(mount["xy"]["right"], dtype=float)))
    build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], separation, scene,
        table_height_m=fixed_runner.TABLE_HEIGHT_M,
        mount_xy_m=mount["xy"], mount_yaw_deg=mount["yaw"],
        **_scene_mount_kwargs(task, mount))
    model = mujoco.MjModel.from_xml_path(str(scene))
    initializer_rows = event_initializer_rows(
        state.get("selected_result"), task.source_row_index,
        source_poll_row_count=(task.source_poll_row_count
                               or spec.row_count))
    dynamics = ({
        "velocity_limit_rad_s": 3.0,
        "acceleration_limit_rad_s2": 5.0,
    } if enforce_dynamics else {})
    (qpos, actual, position_error, orientation_error, _strict,
     discontinuity, diagnostics) = bundle._solve_candidate_dls_hold(
        model, task, mapped, branch_guard_rad=.30,
        initializer_rows=initializer_rows,
        absolute_reach_prefilter=True, **dynamics)
    source_time = np.asarray(task.time_s, dtype=float)
    source_time -= source_time[0]
    velocity, acceleration = fixed_time_derivatives(
        qpos[:, _joint_addresses(model)], source_time,
        initial_velocity_rad_s=0.0)
    if enforce_dynamics:
        validate_enforced_dynamics(velocity, acceleration)
    _collision_union, state_classes, edge_classes = (
        audit_bimanual_collisions(model, qpos, robot_name="piperx"))
    collision = np.asarray([bool(len(value)) for value in state_classes])
    edge_collision = np.asarray([bool(len(value)) for value in edge_classes])
    topology_valid = bundle._topology_valid(model, qpos)
    position = np.column_stack((
        position_error["left"], position_error["right"]))
    orientation = np.column_stack((
        orientation_error["left"], orientation_error["right"]))
    payload = bundle.build_shard_arrays(
        mode=mode, source_time_s=source_time, qpos=qpos,
        position_error_m=position, orientation_error_rad=orientation,
        collision=collision, edge_collision=edge_collision,
        topology_valid=topology_valid,
        left_source_valid=np.asarray(task.left_valid, dtype=bool),
        right_source_valid=np.asarray(task.right_valid, dtype=bool),
        velocity_rad_s=velocity, acceleration_rad_s2=acceleration)
    payload.update({
        "solver_protocol": EVENT_SOLVER_PROTOCOL,
        "timing_source": task.timing_source,
        "source_poll_row_index": np.asarray(task.source_row_index, dtype=int),
        "source_poll_row_count": int(task.source_poll_row_count),
        "left_actual_tcp": np.asarray(actual["left"]),
        "right_actual_tcp": np.asarray(actual["right"]),
        "left_target_position_m": np.asarray(task.left_position_m),
        "right_target_position_m": np.asarray(task.right_position_m),
        "left_target_quaternion_wxyz": np.asarray(mapped["left"]),
        "right_target_quaternion_wxyz": np.asarray(mapped["right"]),
        "left_discontinuity": np.asarray(discontinuity["left"]),
        "right_discontinuity": np.asarray(discontinuity["right"]),
        "state_collision_classes": np.asarray([
            ";".join(map(str, value)) for value in state_classes],
            dtype=np.str_),
        "edge_collision_classes": np.asarray([
            ";".join(map(str, value)) for value in edge_classes],
            dtype=np.str_),
        "paired_failure_reason": np.asarray([
            "ok" if all(value == "ok" for value in row)
            else ";".join(value for value in row if value != "ok")
            for row in diagnostics["failure_reason"]], dtype=np.str_),
        "dls_solve_mode": diagnostics["solve_mode"],
    })
    trajectory = shard_dir / f"{stem}.trajectory.npz"
    np.savez_compressed(trajectory, **payload)
    validate_event_shard(
        payload, spec, mode, source_prefix=source_prefix)
    bundle._validate_formal_evidence(payload, len(source_time))
    metrics = bundle.aggregate_shard(payload)
    fingerprint = _fingerprint(
        spec, mode, mount, source_prefix=source_prefix,
        enforce_dynamics=enforce_dynamics)
    summary = {
        "schema": "piperx-controller-event-v4-summary-v1",
        "solver_protocol": EVENT_SOLVER_PROTOCOL,
        "formal_fingerprint": fingerprint,
        "trajectory": spec.key,
        "family": spec.family.key,
        "take": spec.take,
        "mode": mode,
        "source_sha256": spec.source_sha256,
        "timing_mode": "fixed_controller_event_time",
        "timing_source": task.timing_source,
        "retiming_applied": False,
        "source_poll_rows": int(task.source_poll_row_count),
        "controller_event_rows": len(source_time),
        "controller_event_rows_total": int(controller_event_rows_total),
        "duplicate_poll_rows_removed": int(
            task.source_poll_row_count - controller_event_rows_total),
        "controller_events_not_solved": int(
            controller_event_rows_total - len(source_time)),
        "source_poll_row_index_preserved": True,
        "target_contract": {
            "tool_transform": "configured fixed SE(3)",
            "conditioning": "none",
            "time_varying_wrist_adaptation": False,
            "position_tolerance_mm": 1.0,
            "orientation_tolerance_deg": .5,
        },
        "dynamics_enforced": bool(enforce_dynamics),
        "official_dynamics_limits": {
            "maximum_velocity_rad_s": 3.0,
            "maximum_acceleration_rad_s2": 5.0,
        },
        "target_audit": {
            "maximum_position_deviation_m": float(
                target_audit.maximum_position_deviation_m),
            "maximum_orientation_deviation_rad": float(
                target_audit.maximum_orientation_deviation_rad),
        },
        "mount": mount,
        "scene": str(scene.relative_to(ROOT)),
        "trajectory_artifact": str(trajectory.relative_to(ROOT)),
        "artifacts": {
            "scene_xml": {"path": str(scene.relative_to(ROOT))},
            "trajectory_npz": {"path": str(trajectory.relative_to(ROOT))},
        },
        **metrics,
        "maximum_velocity_rad_s": float(np.max(np.abs(velocity))),
        "maximum_acceleration_rad_s2": float(
            np.max(np.abs(acceleration))),
    }
    summary_path = shard_dir / f"{stem}.summary.json"
    study.atomic_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "coverage": metrics["both_accept_coverage"],
        "collision_frames": metrics["collision_frames"],
        "edge_collision_frames": metrics["edge_collision_frames"],
        "topology_invalid_frames": metrics["topology_invalid_frames"],
        "maximum_velocity_rad_s": summary["maximum_velocity_rad_s"],
        "maximum_acceleration_rad_s2": summary[
            "maximum_acceleration_rad_s2"],
    }, ensure_ascii=False), flush=True)
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--mode", choices=STUDY_MODES, required=True)
    parser.add_argument("--source-prefix", type=int)
    parser.add_argument("--enforce-official-dynamics", action="store_true")
    args = parser.parse_args(argv)
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    matches = [spec for spec in specs if spec.key == args.trajectory]
    if len(matches) != 1:
        raise ValueError("trajectory must match one discovered recording")
    solve_event_shard(
        matches[0], args.mode, source_root=args.source_root,
        output=args.output, source_prefix=args.source_prefix,
        enforce_dynamics=args.enforce_official_dynamics)


if __name__ == "__main__":
    main()
