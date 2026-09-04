"""Upgrade verified Fixed-time evidence to portable, fingerprinted artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from factory_bimanual.multitask_fixed_time_study import (
    STUDY_MODES,
    discover_dual_hand_trajectories,
    validate_shard,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from scripts import build_piperx_multitask_fixed_time_bundle as bundle
from scripts import render_factory_dual_piperx_fixed_time as fixed_runner
from scripts import run_piperx_multitask_fixed_time_mount_study as search
from scripts.search_fold_box_piperx_paired_mount import _scene_mount_kwargs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/piperx_multitask_fixed_time_mount_study"


def _topology_issue_count(record) -> int:
    record = record or {}
    return (int(record.get("structural_crossing_frames", 0))
            + int(record.get("structural_edge_crossing_frames", 0)))


def _select_topology_safe_record(state):
    current = state.get("selected_result")
    if _topology_issue_count(current) == 0:
        return current
    safe = [
        row for row in state.get("records", [])
        if _topology_issue_count(row) == 0
    ]
    if not safe:
        raise ValueError("checkpoint has no topology-safe mount evidence")
    return min(safe, key=lambda row: search.rank_mount_result(
        search._as_rank_record(row)))


def _checkpoint_state(output_root, spec, mode, specs):
    path = bundle._checkpoint_path(output_root, spec, mode)
    state = json.loads(path.read_text(encoding="utf-8"))
    shard_dir = (Path(output_root) / "shards" / spec.family.date
                 / spec.family.task / spec.take / mode)
    summaries = list(shard_dir.glob("*.summary.json"))
    formal_mount = None
    if len(summaries) == 1:
        formal_mount = json.loads(
            summaries[0].read_text(encoding="utf-8")).get("mount")
    same_mount_safe = [
        row for row in state.get("records", [])
        if row.get("mount") == formal_mount and _topology_issue_count(row) == 0
    ]
    selected = (min(
        same_mount_safe, key=lambda row: search.rank_mount_result(
            search._as_rank_record(row)))
        if same_mount_safe else _select_topology_safe_record(state))
    if selected is not state.get("selected_result"):
        state["selected_result"] = search._as_rank_record(selected)
        state["selected_mount"] = selected["mount"]
        state["selection_stage"] = "topology_safe_evidence_migration_v1"
    recommended = search.load_recommended_config().mounts[spec.family.key]
    representative = search.family_representative_spec(
        specs, spec, recommended.source_take)
    target_contract = search._target_contract(spec)
    state.update({
        "source_path": bundle._repo_relative(spec.path),
        "search_schema": search.STUDY_SEARCH_CONFIG.schema,
        "job_fingerprint": search._job_fingerprint(
            spec, mode, search.STUDY_SEARCH_CONFIG, target_contract,
            dependency_source_sha256=(
                representative.source_sha256 if mode == "baseline" else None)),
        "migration": {
            "schema": "piperx-topology-conjunctive-cache-migration-v1",
            "rule": "reuse only selected evidence with zero topology violations",
        },
    })
    if mode == "baseline":
        state["baseline_schema"] = search.BASELINE_SCHEMA
    bundle._atomic_json(path, state)
    return state, formal_mount is not None and formal_mount != state.get("selected_mount")


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    for name in ("schema", "mode"):
        payload[name] = str(payload[name].item())
    payload["retiming_applied"] = bool(payload["retiming_applied"].item())
    return payload


def _save_npz(path, payload):
    serializable = dict(payload)
    temporary = Path(path).with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **serializable)
    temporary.replace(path)


def _migrate_shard(output_root, spec, mode, state):
    shard_dir = (Path(output_root) / "shards" / spec.family.date
                 / spec.family.task / spec.take / mode)
    summaries = list(shard_dir.glob("*.summary.json"))
    trajectories = list(shard_dir.glob("*.trajectory.npz"))
    scenes = list(shard_dir.glob("*.scene.xml"))
    if not (len(summaries) == len(trajectories) == len(scenes) == 1):
        raise ValueError(f"{spec.key}/{mode}: expected one formal artifact set")
    summary_path, trajectory, scene = summaries[0], trajectories[0], scenes[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = _load_npz(trajectory)
    task, _registration = search._load_registered_spec(spec)
    task, _mapped, _audit = search.prepare_family_follow_targets(spec, task)
    count = len(payload["source_time_s"])
    left_valid = np.asarray(task.left_valid[:count], dtype=bool)
    right_valid = np.asarray(task.right_valid[:count], dtype=bool)
    position = np.asarray(payload["position_error_m"], dtype=float)
    orientation = np.asarray(payload["orientation_error_rad"], dtype=float)
    payload["left_source_valid"] = left_valid
    payload["right_source_valid"] = right_valid
    payload["left_accept"] = (
        left_valid & (position[:, 0] <= .001 + 1e-12)
        & (orientation[:, 0] <= np.deg2rad(.5) + 1e-12))
    payload["right_accept"] = (
        right_valid & (position[:, 1] <= .001 + 1e-12)
        & (orientation[:, 1] <= np.deg2rad(.5) + 1e-12))
    payload["both_accept"] = payload["left_accept"] & payload["right_accept"]
    validate_shard(payload, spec, mode)
    _save_npz(trajectory, payload)

    mount = state["selected_mount"]
    separation = float(np.linalg.norm(
        np.asarray(mount["xy"]["left"], dtype=float)
        - np.asarray(mount["xy"]["right"], dtype=float)))
    build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], separation, scene,
        table_height_m=fixed_runner.TABLE_HEIGHT_M,
        mount_xy_m=mount["xy"], mount_yaw_deg=mount["yaw"],
        **_scene_mount_kwargs(task, mount))

    summary.update({
        "solver_protocol": bundle.FORMAL_SOLVER_PROTOCOL,
        "formal_fingerprint": bundle._formal_fingerprint(spec, mode, state),
        "search_job_fingerprint": state.get("job_fingerprint"),
        "trajectory": spec.key,
        "family": spec.family.key,
        "take": spec.take,
        "mode": mode,
        "timing_mode": "fixed_source_time",
        "retiming_applied": False,
        "source_sha256": spec.source_sha256,
        "mount": mount,
        "metrics": bundle.aggregate_shard(payload),
        "dynamics": bundle._dynamics_summary(payload),
        "artifacts": {
            "trajectory_npz": {
                "path": bundle._repo_relative(trajectory),
                "sha256": bundle._sha256(trajectory),
            },
            "scene_xml": {
                "path": bundle._repo_relative(scene),
                "sha256": bundle._sha256(scene),
            },
        },
    })
    bundle._atomic_json(summary_path, summary)


def migrate(output_root=DEFAULT_OUTPUT):
    output_root = Path(output_root).resolve()
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    changed_mounts = []
    states = {}
    for spec in specs:
        for mode in STUDY_MODES:
            state, changed = _checkpoint_state(output_root, spec, mode, specs)
            states[(spec.key, mode)] = state
            if changed:
                changed_mounts.append((spec.key, mode))
    for spec in specs:
        for mode in STUDY_MODES:
            if (spec.key, mode) in changed_mounts:
                continue
            _migrate_shard(output_root, spec, mode, states[(spec.key, mode)])
    return changed_mounts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    changed = migrate(args.output)
    print(json.dumps({"changed_mounts": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
