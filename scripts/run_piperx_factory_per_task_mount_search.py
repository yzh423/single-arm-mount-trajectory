"""Resumable independent three-mode PiperX mount search for factory tasks."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

from factory_bimanual.factory_task_catalog import (
    TaskRepresentative, build_factory_task_catalog, write_dataset_manifest,
)
from factory_bimanual.orientation_mount_search import (
    OrientationSearchConfig, evenly_spaced, generate_mounts,
)
from factory_bimanual.per_task_mount_search import (
    PerTaskSearchConfig, candidate_fingerprint,
    load_registered_representative, prefix_registered_task, select_safe_layout,
)
from scripts.search_fold_box_piperx_mount import (
    local_paired_refinements, rank_paired_mount_candidate,
)
from scripts.search_fold_box_piperx_paired_mount import (
    _sparse_safe, _valid_mount, evaluate_full_pair, evaluate_pair,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (ROOT / "reports" / "factory_bimanual"
          / "piperx_factory_per_task_mount_search")


@dataclass(frozen=True)
class ModeJob:
    task_name: str
    source_sha256: str
    mode: str


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def plan_mode_jobs(representatives, config: PerTaskSearchConfig):
    return tuple(ModeJob(item.task_name, item.sha256, mode)
                 for item in representatives for mode in config.modes)


def evaluate_stage_resumable(*, state, checkpoint, source_sha256, mode,
                             stage, mounts, config, settings, evaluator):
    existing = {row["candidate_fingerprint"]: row
                for row in state.setdefault("records", [])}
    rows = []
    for index, mount in enumerate(mounts):
        fingerprint = candidate_fingerprint(
            source_sha256, mode, mount, config,
            stage=stage, settings=settings)
        if fingerprint in existing:
            rows.append(existing[fingerprint]); continue
        serial = int(fingerprint[:8], 16)
        started = time.perf_counter()
        record = evaluator(mount, serial, settings)
        record.update({
            "stage": stage, "mode": mode,
            "candidate_fingerprint": fingerprint,
            "wall_time_s": time.perf_counter() - started,
        })
        state["records"].append(record)
        existing[fingerprint] = record
        rows.append(record)
        state["heartbeat"] = {"stage": stage, "completed": index + 1,
                              "total": len(mounts)}
        atomic_json(checkpoint, state)
        print(state.get("task_name"), mode, stage, index + 1, len(mounts),
              f"coverage={record.get('continuous_pair_coverage', record.get('coverage', 0)):.4f}",
              f"collision={int(record.get('pair_collision_frames', 0)) + int(record.get('pair_edge_collision_frames', 0))}",
              flush=True)
    return rows


def _normalized_full_record(record):
    normalized = dict(record)
    normalized.setdefault("synchronous_strict_coverage",
                          float(record["continuous_pair_coverage"]))
    normalized.setdefault("longest_failure_frames",
                          int(record.get("longest_hold_frames", 0)))
    normalized.setdefault("p95_pair_pose_error",
                          float(record["mean_pair_pose_error"]))
    normalized.setdefault("minimum_joint_limit_margin_rad", 0.0)
    return normalized


def run_mode(representative, task, mode, config, output=OUTPUT):
    checkpoint = Path(output) / "checkpoints" / representative.task_name / f"{mode}.json"
    state = (json.loads(checkpoint.read_text(encoding="utf-8"))
             if checkpoint.exists() else {
                 "task_name": representative.task_name,
                 "source_path": str(representative.csv_path),
                 "source_sha256": representative.sha256,
                 "mode": mode, "status": "running", "records": []})

    def sparse(mount, serial, settings):
        return evaluate_pair(task, mount, serial, **settings)

    coarse_mounts = generate_mounts(
        mode, task, OrientationSearchConfig(
            modes=(mode,), maximum_candidates=config.coarse_budget))
    coarse = evaluate_stage_resumable(
        state=state, checkpoint=checkpoint,
        source_sha256=representative.sha256, mode=mode, stage="coarse",
        mounts=coarse_mounts, config=config,
        settings={"uniform_count": 8, "global_seed_count": 1,
                  "max_iterations": 45, "maximum_candidates": 1,
                  "constrained_fallback_enabled": False}, evaluator=sparse)
    safe_coarse = sorted((row for row in coarse if _sparse_safe(row)),
                         key=rank_paired_mount_candidate)
    if not safe_coarse:
        state.update(status="no_safe_coarse", selected_layout=None)
        atomic_json(checkpoint, state); return state
    dense_mounts = [row["mount"] for row in safe_coarse[:config.dense_budget]]
    dense = evaluate_stage_resumable(
        state=state, checkpoint=checkpoint,
        source_sha256=representative.sha256, mode=mode, stage="dense",
        mounts=dense_mounts, config=config,
        settings={"uniform_count": 30, "global_seed_count": 6,
                  "max_iterations": 100, "maximum_candidates": 3,
                  "constrained_fallback_enabled": True}, evaluator=sparse)
    safe_dense = sorted((row for row in dense if _sparse_safe(row)),
                        key=rank_paired_mount_candidate)
    if not safe_dense:
        state.update(status="no_safe_dense", selected_layout=None)
        atomic_json(checkpoint, state); return state
    local_mounts = []
    for side in ("left", "right"):
        for mount in local_paired_refinements(
                safe_dense[0]["mount"], side=side,
                xy_step_m=.04, yaw_step_deg=10.):
            mount["mode"] = mode
            mount["base_z_m"] = {"left": mount["shared_base_z_m"],
                                 "right": mount["shared_base_z_m"]}
            if _valid_mount(mount):
                local_mounts.append(mount)
    local = evaluate_stage_resumable(
        state=state, checkpoint=checkpoint,
        source_sha256=representative.sha256, mode=mode, stage="local",
        mounts=evenly_spaced(local_mounts, config.local_budget), config=config,
        settings={"uniform_count": 36, "global_seed_count": 7,
                  "max_iterations": 110, "maximum_candidates": 4,
                  "constrained_fallback_enabled": True}, evaluator=sparse)
    finalists = sorted(
        (row for row in [*dense, *local] if _sparse_safe(row)),
        key=rank_paired_mount_candidate)[:config.finalist_budget]

    def full(mount, serial, _settings):
        return _normalized_full_record(evaluate_full_pair(task, mount, serial))

    full_rows = evaluate_stage_resumable(
        state=state, checkpoint=checkpoint,
        source_sha256=representative.sha256, mode=mode, stage="full",
        mounts=[row["mount"] for row in finalists], config=config,
        # Bump the full-audit fingerprint when the target representation or
        # audit gates change.  This prevents resumable runs from reusing old
        # zero-coverage records produced by an incompatible audit.
        settings={"audit_scope": "full_source_timeline-v2-smoothed-targets"},
        evaluator=full)
    try:
        selected = select_safe_layout(full_rows)
        state.update(status="complete", selected_layout=selected)
    except RuntimeError:
        state.update(status="no_safe_layout", selected_layout=None)
    atomic_json(checkpoint, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--short-prefix", type=int)
    parser.add_argument("--task")
    args = parser.parse_args(argv)
    catalog = build_factory_task_catalog(ROOT / "data" / "factory")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_dataset_manifest(catalog, OUTPUT / "dataset_manifest.json")
    config = PerTaskSearchConfig()
    representatives = tuple(
        item for item in catalog.representatives
        if args.task is None or item.task_name == args.task)
    jobs = plan_mode_jobs(representatives, config)
    if args.dry_run:
        atomic_json(OUTPUT / "batch_status.json", {
            "status": "planned", "task_count": len(representatives),
            "job_count": len(jobs), "jobs": [job.__dict__ for job in jobs]})
        return
    if not args.full and args.short_prefix is None:
        parser.error("use --full or --short-prefix ROWS")
    selected = {}
    for representative in representatives:
        task = load_registered_representative(representative)
        if args.short_prefix is not None:
            task = prefix_registered_task(task, args.short_prefix)
        task_result = {"source_path": str(representative.csv_path), "modes": {}}
        for mode in config.modes:
            try:
                state = run_mode(representative, task, mode, config)
                task_result["modes"][mode] = {
                    "status": state["status"],
                    "selected_layout": state.get("selected_layout")}
            except Exception as error:
                task_result["modes"][mode] = {
                    "status": "error", "error": repr(error)}
            atomic_json(OUTPUT / "selected_layouts.json", selected | {
                representative.task_name: task_result})
        safe = [value["selected_layout"]
                for value in task_result["modes"].values()
                if value["status"] == "complete"]
        task_result["selected_layout"] = (
            min(safe, key=lambda row: (
                -row["synchronous_strict_coverage"],
                row["longest_failure_frames"],
                row["mean_pair_pose_error"])) if safe else None)
        task_result["status"] = "complete" if safe else "no_safe_layout"
        selected[representative.task_name] = task_result
        atomic_json(OUTPUT / "selected_layouts.json", selected)
    atomic_json(OUTPUT / "batch_status.json", {
        "status": "complete", "task_count": len(representatives),
        "job_count": len(jobs)})


if __name__ == "__main__":
    main()
