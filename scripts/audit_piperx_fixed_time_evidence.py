"""Re-audit the formal PiperX study against raw mapped dual-hand targets.

This audit does not modify trajectories.  It separates three contracts that the
original aggregate mixed together: fidelity to the unconditioned source target,
fidelity to the solver's conditioned target, and fixed-time joint dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from factory_bimanual.fixed_time_evidence_audit import (
    compare_target_tracks,
    pose_tracking_errors,
    strict_pair_accept,
    taskspace_segment_rates,
)
from factory_bimanual.multitask_fixed_time_study import (
    STUDY_MODES,
    discover_dual_hand_trajectories,
)
from factory_bimanual.piperx_recommended import load_recommended_config
from scripts import run_piperx_multitask_fixed_time_mount_study as study
from scripts.run_piperx_recommended_v31 import condition_complete_follow_targets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "reports/piperx_multitask_fixed_time_mount_study"
DEFAULT_OUTPUT = DEFAULT_STUDY / "literature_audit"
POSITION_TOLERANCE_M = 0.001
ORIENTATION_TOLERANCE_RAD = float(np.deg2rad(0.5))
STUDY_VELOCITY_LIMIT_RAD_S = 1.0
STUDY_ACCELERATION_LIMIT_RAD_S2 = 4.0
OFFICIAL_VELOCITY_LIMIT_RAD_S = 3.0
OFFICIAL_ACCELERATION_LIMIT_RAD_S2 = 5.0


def _maximum(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.max(array, initial=0.0))


def _percentile(values, percentile) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, percentile)) if array.size else 0.0


def _coverage(mask, valid) -> float:
    mask = np.asarray(mask, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    count = int(np.count_nonzero(valid))
    return float(np.count_nonzero(mask & valid) / count) if count else 0.0


def _target_cache(spec):
    registered, _ = study._load_registered_spec(spec)
    mount = load_recommended_config().mounts[spec.family.key]
    fixed_mapped, fixed_mapped_quaternion, _ = condition_complete_follow_targets(
        registered,
        {
            "left": mount.left_tool_offset_quaternion_wxyz,
            "right": mount.right_tool_offset_quaternion_wxyz,
        },
        tool_translations={
            "left": mount.left_tool_translation_m,
            "right": mount.right_tool_translation_m,
        },
        wrist_adaptation=None,
        apply_conditioning=False,
    )
    adapted, adapted_quaternion, _ = study.prepare_family_follow_targets(
        spec, registered, apply_conditioning=False)
    conditioned, conditioned_quaternion, conditioning_audit = (
        study.prepare_family_follow_targets(
            spec, registered, apply_conditioning=True))
    return {
        "fixed_mapped": fixed_mapped,
        "fixed_mapped_quaternion": fixed_mapped_quaternion,
        "adapted": adapted,
        "adapted_quaternion": adapted_quaternion,
        "conditioned": conditioned,
        "conditioned_quaternion": conditioned_quaternion,
        "conditioning_audit": conditioning_audit,
    }


def _audit_shard(spec, manifest_row, study_root, targets):
    summary_path = ROOT / manifest_row["summary_json"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trajectory_path = ROOT / summary["artifacts"]["trajectory_npz"]["path"]
    with np.load(trajectory_path, allow_pickle=False) as payload:
        source_time = np.asarray(payload["source_time_s"], dtype=float)
        count = len(source_time)
        raw_task = targets["fixed_mapped"]
        adapted_task = targets["adapted"]
        conditioned_task = targets["conditioned"]
        pair_valid = (
            np.asarray(payload["left_source_valid"], dtype=bool)
            & np.asarray(payload["right_source_valid"], dtype=bool)
        )
        side_errors = {}
        target_differences = {}
        adaptation_differences = {}
        smoothing_differences = {}
        stored_target_differences = {}
        rates = {}
        for side in ("left", "right"):
            raw_position = np.asarray(
                getattr(raw_task, f"{side}_position_m"), dtype=float)[:count]
            raw_quaternion = np.asarray(
                targets["fixed_mapped_quaternion"][side], dtype=float)[:count]
            adapted_position = np.asarray(
                getattr(adapted_task, f"{side}_position_m"), dtype=float)[:count]
            adapted_quaternion = np.asarray(
                targets["adapted_quaternion"][side], dtype=float)[:count]
            conditioned_position = np.asarray(
                getattr(conditioned_task, f"{side}_position_m"), dtype=float)[:count]
            conditioned_quaternion = np.asarray(
                targets["conditioned_quaternion"][side], dtype=float)[:count]
            side_errors[side] = pose_tracking_errors(
                payload[f"{side}_actual_tcp"], raw_position, raw_quaternion)
            target_differences[side] = compare_target_tracks(
                raw_position, raw_quaternion,
                conditioned_position, conditioned_quaternion,
                position_budget_m=POSITION_TOLERANCE_M,
                orientation_budget_rad=ORIENTATION_TOLERANCE_RAD,
            )
            adaptation_differences[side] = compare_target_tracks(
                raw_position, raw_quaternion,
                adapted_position, adapted_quaternion,
                position_budget_m=POSITION_TOLERANCE_M,
                orientation_budget_rad=ORIENTATION_TOLERANCE_RAD,
            )
            smoothing_differences[side] = compare_target_tracks(
                adapted_position, adapted_quaternion,
                conditioned_position, conditioned_quaternion,
                position_budget_m=POSITION_TOLERANCE_M,
                orientation_budget_rad=ORIENTATION_TOLERANCE_RAD,
            )
            stored_target_differences[side] = compare_target_tracks(
                conditioned_position, conditioned_quaternion,
                payload[f"{side}_target_position_m"],
                payload[f"{side}_target_quaternion_wxyz"],
                position_budget_m=1e-10,
                orientation_budget_rad=1e-7,
            )
            rates[side] = taskspace_segment_rates(
                source_time, raw_position, raw_quaternion)
        raw_accept = strict_pair_accept(
            side_errors["left"].position_error_m,
            side_errors["right"].position_error_m,
            side_errors["left"].orientation_error_rad,
            side_errors["right"].orientation_error_rad,
            source_valid=pair_valid,
            position_tolerance_m=POSITION_TOLERANCE_M,
            orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
        )
        conditioned_accept = np.asarray(payload["both_accept"], dtype=bool)
        velocity = np.abs(np.asarray(payload["velocity_rad_s"], dtype=float))
        acceleration = np.abs(
            np.asarray(payload["acceleration_rad_s2"], dtype=float))
        maximum_velocity = _maximum(velocity)
        maximum_acceleration = _maximum(acceleration)
        return {
            "trajectory": spec.key,
            "family": spec.family.key,
            "take": spec.take,
            "mode": manifest_row["mode"],
            "frames": count,
            "valid_pair_frames": int(np.count_nonzero(pair_valid)),
            "conditioned_pair_coverage": _coverage(conditioned_accept, pair_valid),
            "raw_pair_coverage": _coverage(raw_accept, pair_valid),
            "coverage_delta_percentage_points": 100.0 * (
                _coverage(conditioned_accept, pair_valid)
                - _coverage(raw_accept, pair_valid)),
            "solver_target_max_position_deviation_mm": 1000.0 * max(
                item.maximum_position_deviation_m
                for item in target_differences.values()),
            "solver_target_max_orientation_deviation_deg": float(np.rad2deg(max(
                item.maximum_orientation_deviation_rad
                for item in target_differences.values()))),
            "smoothing_max_position_deviation_mm": 1000.0 * max(
                item.maximum_position_deviation_m
                for item in smoothing_differences.values()),
            "smoothing_max_orientation_deviation_deg": float(np.rad2deg(max(
                item.maximum_orientation_deviation_rad
                for item in smoothing_differences.values()))),
            "wrist_adaptation_max_position_mm": 1000.0 * max(
                item.maximum_position_deviation_m
                for item in adaptation_differences.values()),
            "wrist_adaptation_max_orientation_deg": float(np.rad2deg(max(
                item.maximum_orientation_deviation_rad
                for item in adaptation_differences.values()))),
            "stored_solver_target_matches_current_contract": not any(
                item.position_budget_exceeded or item.orientation_budget_exceeded
                for item in stored_target_differences.values()),
            "solver_target_exceeds_strict_position_tolerance": any(
                item.position_budget_exceeded
                for item in target_differences.values()),
            "solver_target_exceeds_strict_orientation_tolerance": any(
                item.orientation_budget_exceeded
                for item in target_differences.values()),
            "raw_max_position_error_mm": 1000.0 * max(
                _maximum(item.position_error_m) for item in side_errors.values()),
            "raw_max_orientation_error_deg": float(np.rad2deg(max(
                _maximum(item.orientation_error_rad)
                for item in side_errors.values()))),
            "source_interval_median_ms": 1000.0 * float(np.median(np.diff(source_time))),
            "source_tcp_linear_speed_p95_m_s": max(
                _percentile(item.linear_speed_m_s, 95.0) for item in rates.values()),
            "source_tcp_linear_speed_max_m_s": max(
                _maximum(item.linear_speed_m_s) for item in rates.values()),
            "source_tcp_angular_speed_p95_rad_s": max(
                _percentile(item.angular_speed_rad_s, 95.0) for item in rates.values()),
            "source_tcp_angular_speed_max_rad_s": max(
                _maximum(item.angular_speed_rad_s) for item in rates.values()),
            "maximum_joint_velocity_rad_s": maximum_velocity,
            "maximum_joint_acceleration_rad_s2": maximum_acceleration,
            "study_dynamics_pass": (
                maximum_velocity <= STUDY_VELOCITY_LIMIT_RAD_S + 1e-12
                and maximum_acceleration <= STUDY_ACCELERATION_LIMIT_RAD_S2 + 1e-12),
            "official_ceiling_dynamics_pass": (
                maximum_velocity <= OFFICIAL_VELOCITY_LIMIT_RAD_S + 1e-12
                and maximum_acceleration <= OFFICIAL_ACCELERATION_LIMIT_RAD_S2 + 1e-12),
            "collision_frames": int(np.count_nonzero(payload["collision"])),
            "edge_collision_frames": int(
                np.count_nonzero(payload["edge_collision"])),
            "topology_invalid_frames": int(
                np.count_nonzero(~np.asarray(payload["topology_valid"], dtype=bool))),
        }


def _aggregate(rows):
    total_frames = sum(row["valid_pair_frames"] for row in rows)
    weighted = lambda name: (
        sum(row[name] * row["valid_pair_frames"] for row in rows) / total_frames
        if total_frames else 0.0)
    by_mode = {}
    for mode in STUDY_MODES:
        selected = [row for row in rows if row["mode"] == mode]
        mode_frames = sum(row["valid_pair_frames"] for row in selected)
        by_mode[mode] = {
            "shards": len(selected),
            "valid_pair_frames": mode_frames,
            "conditioned_pair_coverage": (
                sum(row["conditioned_pair_coverage"] * row["valid_pair_frames"]
                    for row in selected) / mode_frames if mode_frames else 0.0),
            "raw_pair_coverage": (
                sum(row["raw_pair_coverage"] * row["valid_pair_frames"]
                    for row in selected) / mode_frames if mode_frames else 0.0),
            "study_dynamics_pass_shards": sum(
                bool(row["study_dynamics_pass"]) for row in selected),
            "official_ceiling_dynamics_pass_shards": sum(
                bool(row["official_ceiling_dynamics_pass"]) for row in selected),
        }
    return {
        "schema": "piperx-fixed-time-raw-source-evidence-audit-v1",
        "contract": {
            "timing": "fixed source timestamps; no retiming",
            "strict_position_tolerance_m": POSITION_TOLERANCE_M,
            "strict_orientation_tolerance_deg": 0.5,
            "study_joint_velocity_limit_rad_s": STUDY_VELOCITY_LIMIT_RAD_S,
            "study_joint_acceleration_limit_rad_s2": STUDY_ACCELERATION_LIMIT_RAD_S2,
            "official_configurable_joint_velocity_ceiling_rad_s": OFFICIAL_VELOCITY_LIMIT_RAD_S,
            "official_configurable_joint_acceleration_ceiling_rad_s2": OFFICIAL_ACCELERATION_LIMIT_RAD_S2,
        },
        "shards": len(rows),
        "valid_pair_frames": total_frames,
        "weighted_conditioned_pair_coverage": weighted("conditioned_pair_coverage"),
        "weighted_raw_pair_coverage": weighted("raw_pair_coverage"),
        "solver_target_position_budget_exceeded_shards": sum(
            row["solver_target_exceeds_strict_position_tolerance"] for row in rows),
        "solver_target_orientation_budget_exceeded_shards": sum(
            row["solver_target_exceeds_strict_orientation_tolerance"] for row in rows),
        "stored_solver_target_contract_match_shards": sum(
            row["stored_solver_target_matches_current_contract"] for row in rows),
        "study_dynamics_pass_shards": sum(row["study_dynamics_pass"] for row in rows),
        "official_ceiling_dynamics_pass_shards": sum(
            row["official_ceiling_dynamics_pass"] for row in rows),
        "nonzero_raw_coverage_and_study_dynamics_pass_shards": sum(
            row["raw_pair_coverage"] > 0.0 and row["study_dynamics_pass"]
            for row in rows),
        "nonzero_raw_coverage_and_official_ceiling_pass_shards": sum(
            row["raw_pair_coverage"] > 0.0
            and row["official_ceiling_dynamics_pass"] for row in rows),
        "collision_frames": sum(row["collision_frames"] for row in rows),
        "edge_collision_frames": sum(row["edge_collision_frames"] for row in rows),
        "topology_invalid_frames": sum(row["topology_invalid_frames"] for row in rows),
        "maximum_source_tcp_linear_speed_m_s": max(
            (row["source_tcp_linear_speed_max_m_s"] for row in rows), default=0.0),
        "maximum_source_tcp_angular_speed_rad_s": max(
            (row["source_tcp_angular_speed_max_rad_s"] for row in rows), default=0.0),
        "by_mode": by_mode,
        "primary_references": [
            {
                "topic": "Piper joint speed and acceleration settings",
                "url": "https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD",
            },
            {
                "topic": "fixed-path time parameterization",
                "doi": "10.1109/TRO.2018.2819195",
            },
            {
                "topic": "multi-objective continuous IK",
                "doi": "10.15607/RSS.2018.XIV.043",
            },
        ],
    }


def run(study_root=DEFAULT_STUDY, output=DEFAULT_OUTPUT):
    study_root = Path(study_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (study_root / "bundle_manifest.json").read_text(encoding="utf-8"))
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    spec_by_key = {spec.key: spec for spec in specs}
    cache = {}
    rows = []
    for manifest_row in manifest["shards"]:
        key = manifest_row["trajectory"]
        spec = spec_by_key[key]
        if key not in cache:
            cache[key] = _target_cache(spec)
        rows.append(_audit_shard(spec, manifest_row, study_root, cache[key]))
    rows.sort(key=lambda row: (row["trajectory"], STUDY_MODES.index(row["mode"])))
    csv_path = output / "raw_source_shard_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _aggregate(rows)
    summary["artifacts"] = {
        "shard_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
    }
    json_path = output / "raw_source_audit_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return csv_path, json_path, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    csv_path, json_path, summary = run(args.study_root, args.output)
    print(csv_path)
    print(json_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
