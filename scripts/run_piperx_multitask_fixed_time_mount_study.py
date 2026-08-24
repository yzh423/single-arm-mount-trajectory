"""Run the resumable 27-trajectory PiperX fixed-time mount study."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from factory_bimanual.multitask_fixed_time_study import (
    STUDY_MODES,
    TrajectorySpec,
    discover_dual_hand_trajectories,
)
from factory_bimanual.orientation_mount_search import (
    OrientationSearchConfig,
    evenly_spaced,
    generate_mounts,
)
from factory_bimanual.per_task_mount_search import PerTaskSearchConfig
from factory_bimanual.piperx_recommended import (
    load_recommended_config,
    world_mount_for_family,
)
from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.source_data import load_factory_task
from scripts.search_fold_box_piperx_mount import (
    local_paired_refinements,
    rank_paired_mount_candidate,
)
from scripts.search_fold_box_piperx_paired_mount import (
    TABLE_HEIGHT_M,
    _sparse_safe,
    _valid_mount,
    evaluate_pair,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/piperx_multitask_fixed_time_mount_study"
BASELINE_SCHEMA = "piperx-physical-baseline-scene-v2"
BASELINE_ADAPTER_HEIGHT_M = .08
STUDY_POSITION_TOLERANCE_M = .001
STUDY_ORIENTATION_TOLERANCE_RAD = float(np.deg2rad(.5))
STUDY_SEARCH_CONFIG = PerTaskSearchConfig(
    coarse_budget=16,
    dense_budget=3,
    local_budget=6,
    finalist_budget=2,
    schema="piperx-multitask-family-shared-search-v5-current-funnel-only",
)


@dataclass(frozen=True)
class StudyJob:
    spec: TrajectorySpec
    mode: str

    @property
    def key(self) -> str:
        return f"{self.spec.key}/{self.mode}"


def plan_jobs(specs, modes=STUDY_MODES):
    modes = tuple(modes)
    unknown = set(modes) - set(STUDY_MODES)
    if unknown:
        raise ValueError(f"unsupported study modes: {sorted(unknown)}")
    if len(modes) != len(set(modes)):
        raise ValueError("study modes must be unique")
    jobs = tuple(StudyJob(spec, mode) for spec in specs for mode in modes)
    if len({job.key for job in jobs}) != len(jobs):
        raise ValueError("study job keys must be unique")
    return jobs


def family_representative_spec(specs, spec, representative_take):
    """Resolve the configured dual-hand representative within one task family."""
    matches = tuple(
        candidate for candidate in specs
        if (candidate.family.key == spec.family.key
            and candidate.take == representative_take)
    )
    if len(matches) != 1:
        raise ValueError(
            f"{spec.family.key}: configured representative take "
            f"{representative_take!r} was not found exactly once in the "
            "discovered dual-hand data")
    return matches[0]


def rank_mount_result(record):
    """Rank audited results without hiding unsafe or failed trajectories."""
    return (
        -float(record["both_accept_coverage"]),
        int(record.get("collision_frames", 0))
        + int(record.get("edge_collision_frames", 0)),
        int(record.get("topology_invalid_frames", 0)),
        int(record.get("longest_hold_frames", 0)),
        float(record.get("maximum_normalized_error", float("inf"))),
    )


def _json_spec(spec):
    value = asdict(spec)
    value["family"] = spec.family.key
    value["path"] = str(spec.path)
    return value


def atomic_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def study_status_path(output, mode):
    suffix = f"_{mode}" if mode else ""
    return Path(output) / f"study_status{suffix}.json"


def _load_registered_spec(spec: TrajectorySpec):
    source = load_factory_task(
        spec.path, spec.family.key, max_translation_jump_m=0.20,
        repair_invalid_pose_rows=True)
    points = np.vstack((source.left_position_m, source.right_position_m))
    translation = np.asarray((
        -points[:, 0].mean(), -points[:, 1].mean(),
        0.90 - points[:, 2].min()))
    registration = RigidTaskRegistration(np.eye(3), translation)
    return register_task(source, registration), registration


def _prefix_task(task, rows):
    if rows is None or rows >= len(task.time_s):
        return task
    from dataclasses import replace
    count = min(len(task.time_s), max(2, int(rows)))
    values = {}
    for name, value in task.__dict__.items():
        if isinstance(value, np.ndarray) and value.ndim and len(value) == len(task.time_s):
            values[name] = value[:count].copy()
    return replace(task, **values)


def _baseline_mount(spec, registration):
    mount = world_mount_for_family(
        load_recommended_config(), spec.family,
        registration.rotation_world_from_vr,
        registration.translation_world_m)
    return _normalize_baseline_mount_payload(mount.as_scene_mount())


def _normalize_baseline_mount_payload(payload):
    """Retain the configured orientation while enforcing physical support."""
    result = json.loads(json.dumps(payload))
    original_z = float(result["shared_base_z_m"])
    mode = result.get("mode", "upright_table")
    lower = TABLE_HEIGHT_M + BASELINE_ADAPTER_HEIGHT_M
    normalized_z = original_z
    if mode in {"upright_table", "horizontal_wall", "horizontal_forward"}:
        normalized_z = min(1.50, max(lower, original_z))
    result["shared_base_z_m"] = normalized_z
    result["base_z_m"] = {
        side: normalized_z for side in ("left", "right")}
    if not np.isclose(normalized_z, original_z, rtol=0.0, atol=1e-12):
        method = result.get("selection_method", "configured baseline")
        result["selection_method"] = (
            f"{method}; physical support clamp "
            f"{original_z:.6f} m -> {normalized_z:.6f} m")
    return result


def _candidate_fingerprint(spec, mode, stage, mount, settings):
    payload = {
        "schema": "piperx-multitask-family-shared-search-v2",
        "source_sha256": spec.source_sha256,
        "mode": mode, "stage": stage, "mount": mount,
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _evaluate_stage(*, state, checkpoint, spec, mode, stage, mounts,
                    settings, evaluator):
    existing = {row["candidate_fingerprint"]: row
                for row in state.setdefault("records", [])}
    rows = []
    for index, mount in enumerate(mounts):
        fingerprint = _candidate_fingerprint(
            spec, mode, stage, mount, settings)
        if fingerprint in existing:
            rows.append(existing[fingerprint])
            continue
        started = time.perf_counter()
        record = evaluator(mount, int(fingerprint[:8], 16), settings)
        record.update({
            "stage": stage,
            "mode": mode,
            "candidate_fingerprint": fingerprint,
            "wall_time_s": time.perf_counter() - started,
        })
        state["records"].append(record)
        existing[fingerprint] = record
        rows.append(record)
        state["heartbeat"] = {
            "stage": stage, "completed": index + 1, "total": len(mounts)}
        atomic_json(checkpoint, state)
        print(spec.key, mode, stage, index + 1, len(mounts),
              f"coverage={record.get('continuous_pair_coverage', 0.0):.4f}",
              f"collision={int(record.get('pair_collision_frames', 0)) + int(record.get('pair_edge_collision_frames', 0))}",
              flush=True)
    return rows


def _as_rank_record(record):
    coverage = float(record.get(
        "synchronous_strict_coverage",
        record.get("continuous_pair_coverage", 0.0)))
    return {
        **record,
        "both_accept_coverage": coverage,
        "collision_frames": int(record.get("pair_collision_frames", 0)),
        "edge_collision_frames": int(record.get("pair_edge_collision_frames", 0)),
        "topology_invalid_frames": int(record.get("structural_crossing_frames", 0))
        + int(record.get("structural_edge_crossing_frames", 0)),
        "longest_hold_frames": int(record.get("longest_hold_frames", 0)),
        "maximum_normalized_error": float(record.get(
            "p95_pair_pose_error", record.get("mean_pair_pose_error", 1e9))),
    }


def _best_observed_mount(records):
    if not records:
        return None, None
    safe_records = [record for record in records if _sparse_safe(record)]
    candidates = safe_records or records
    selected = min(candidates, key=lambda row: rank_mount_result(
        _as_rank_record(row)))
    return selected.get("mount"), _as_rank_record(selected)


def _best_current_funnel_mount(*stage_rows):
    """Select only from rows evaluated under the active search settings."""
    return _best_observed_mount([
        row for rows in stage_rows for row in rows])


def run_job(job: StudyJob, output=DEFAULT_OUTPUT, *, short_prefix=None,
            search_config=STUDY_SEARCH_CONFIG, study_specs=None):
    spec, mode = job.spec, job.mode
    checkpoint = (Path(output) / "checkpoints" / spec.family.date
                  / spec.family.task / spec.take / f"{mode}.json")
    state = (json.loads(checkpoint.read_text(encoding="utf-8"))
             if checkpoint.exists() else {
                 "schema": "piperx-multitask-fixed-time-search-state-v1",
                 "trajectory": spec.key, "source_path": str(spec.path),
                 "source_sha256": spec.source_sha256, "mode": mode,
                 "status": "running", "records": []})
    if state.get("source_sha256") != spec.source_sha256:
        raise ValueError(f"{job.key}: checkpoint source hash mismatch")
    terminal = state.get("status") in {"complete", "infeasible"}
    current_search = state.get("search_schema") == search_config.schema
    current_baseline = state.get("baseline_schema") == BASELINE_SCHEMA
    if (terminal and state.get("selected_mount")
            and ((mode == "baseline" and current_baseline)
                 or (mode != "baseline" and current_search))):
        return state

    if mode != "baseline":
        specs = (tuple(study_specs) if study_specs is not None else
                 discover_dual_hand_trajectories(ROOT / "data/factory"))
        recommended = load_recommended_config().mounts[spec.family.key]
        representative = family_representative_spec(
            specs, spec, recommended.source_take)
        if representative.key != spec.key:
            representative_state = run_job(
                StudyJob(representative, mode), output,
                short_prefix=short_prefix, search_config=search_config,
                study_specs=specs)
            state.update(
                status=representative_state["status"],
                selected_mount=representative_state.get("selected_mount"),
                selected_result=representative_state.get("selected_result"),
                selection_stage="family_shared_representative",
                representative_trajectory=representative.key,
                search_schema=search_config.schema,
            )
            if representative_state.get("failure_stage"):
                state["failure_stage"] = representative_state["failure_stage"]
            atomic_json(checkpoint, state)
            return state

    state["search_schema"] = search_config.schema
    task, registration = _load_registered_spec(spec)
    task = _prefix_task(task, short_prefix)

    if mode == "baseline":
        # Baseline is the configured comparison anchor, not a searched
        # candidate.  Its complete metrics are produced by the strict shard
        # solver, so running a second expensive screening solve here would
        # only duplicate work.
        state.update(
            status="complete",
            selected_mount=_baseline_mount(spec, registration),
            selected_result=None,
            selection_stage="configured_baseline",
            representative_trajectory=spec.key,
            baseline_schema=BASELINE_SCHEMA,
        )
        atomic_json(checkpoint, state)
        return state
    else:
        config = OrientationSearchConfig(
            modes=(mode,), maximum_candidates=search_config.coarse_budget)
        coarse = _evaluate_stage(
            state=state, checkpoint=checkpoint, spec=spec, mode=mode,
            stage="coarse", mounts=generate_mounts(mode, task, config),
            settings={"uniform_count": 8, "global_seed_count": 1,
                      "max_iterations": 45, "maximum_candidates": 1,
                      "position_tolerance_m": STUDY_POSITION_TOLERANCE_M,
                      "orientation_tolerance_rad": STUDY_ORIENTATION_TOLERANCE_RAD,
                      "constrained_fallback_enabled": False},
            evaluator=lambda mount, serial, settings: evaluate_pair(
                task, mount, serial, **settings))
        safe_coarse = sorted(
            (row for row in coarse if _sparse_safe(row)),
            key=rank_paired_mount_candidate)
        if not safe_coarse:
            selected_mount, selected_result = _best_observed_mount(coarse)
            state.update(status="infeasible", selected_mount=selected_mount,
                         selected_result=selected_result,
                         failure_stage="coarse")
            atomic_json(checkpoint, state)
            return state
        dense_mounts = [row["mount"] for row in safe_coarse[:search_config.dense_budget]]
        dense = _evaluate_stage(
            state=state, checkpoint=checkpoint, spec=spec, mode=mode,
            stage="dense", mounts=dense_mounts,
            settings={"uniform_count": 30, "global_seed_count": 6,
                      "max_iterations": 100, "maximum_candidates": 3,
                      "position_tolerance_m": STUDY_POSITION_TOLERANCE_M,
                      "orientation_tolerance_rad": STUDY_ORIENTATION_TOLERANCE_RAD,
                      "constrained_fallback_enabled": True},
            evaluator=lambda mount, serial, settings: evaluate_pair(
                task, mount, serial, **settings))
        safe_dense = sorted(
            (row for row in dense if _sparse_safe(row)),
            key=rank_paired_mount_candidate)
        if not safe_dense:
            selected_mount, selected_result = _best_current_funnel_mount(
                coarse, dense)
            state.update(status="infeasible", selected_mount=selected_mount,
                         selected_result=selected_result,
                         failure_stage="dense")
            atomic_json(checkpoint, state)
            return state
        local_mounts = []
        for side in ("left", "right"):
            for mount in local_paired_refinements(
                    safe_dense[0]["mount"], side=side,
                    xy_step_m=.04, yaw_step_deg=10.):
                mount["mode"] = mode
                mount["base_z_m"] = {
                    side_name: mount["shared_base_z_m"]
                    for side_name in ("left", "right")}
                if _valid_mount(mount):
                    local_mounts.append(mount)
        local = _evaluate_stage(
            state=state, checkpoint=checkpoint, spec=spec, mode=mode,
            stage="local", mounts=evenly_spaced(
                local_mounts, search_config.local_budget),
            settings={"uniform_count": 36, "global_seed_count": 7,
                      "max_iterations": 110, "maximum_candidates": 4,
                      "position_tolerance_m": STUDY_POSITION_TOLERANCE_M,
                      "orientation_tolerance_rad": STUDY_ORIENTATION_TOLERANCE_RAD,
                      "constrained_fallback_enabled": True},
            evaluator=lambda mount, serial, settings: evaluate_pair(
                task, mount, serial, **settings))
        finalists = sorted(
            (row for row in [*dense, *local] if _sparse_safe(row)),
            key=rank_paired_mount_candidate)[:search_config.finalist_budget]
        full = _evaluate_stage(
            state=state, checkpoint=checkpoint, spec=spec, mode=mode,
            stage="full", mounts=[row["mount"] for row in finalists],
            settings={"uniform_count": 80, "global_seed_count": 8,
                      "max_iterations": 110, "maximum_candidates": 4,
                      "position_tolerance_m": STUDY_POSITION_TOLERANCE_M,
                      "orientation_tolerance_rad": STUDY_ORIENTATION_TOLERANCE_RAD,
                      "constrained_fallback_enabled": True,
                      "scope": "family-representative-uniform-probe-v3"},
            evaluator=lambda mount, serial, settings: evaluate_pair(
                task, mount, serial,
                **{key: value for key, value in settings.items()
                   if key != "scope"}))

    if not full:
        selected_mount, selected_result = _best_current_funnel_mount(
            coarse, dense, local)
        state.update(status="infeasible", selected_mount=selected_mount,
                     selected_result=selected_result, failure_stage="full")
    else:
        selected = min(full, key=lambda row: rank_mount_result(_as_rank_record(row)))
        state.update(
            status="complete", selected_mount=selected["mount"],
            selected_result=_as_rank_record(selected))
    atomic_json(checkpoint, state)
    return state


def run_study(output=DEFAULT_OUTPUT, *, trajectory=None, mode=None,
              short_prefix=None):
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    jobs = plan_jobs(specs, STUDY_MODES if mode is None else (mode,))
    if trajectory:
        jobs = tuple(job for job in jobs if job.spec.key == trajectory)
    if not jobs:
        raise ValueError("no study jobs matched the requested filters")
    results = {}
    for job in jobs:
        try:
            state = run_job(
                job, output, short_prefix=short_prefix, study_specs=specs)
            results[job.key] = {
                "status": state["status"],
                "selected_mount": state.get("selected_mount"),
                "checkpoint": str((Path(output) / "checkpoints" / job.spec.family.date
                                   / job.spec.family.task / job.spec.take
                                   / f"{job.mode}.json")),
            }
        except Exception as error:
            results[job.key] = {"status": "error", "error": repr(error)}
        atomic_json(study_status_path(output, mode), {
            "schema": "piperx-multitask-fixed-time-study-status-v1",
            "planned_jobs": len(jobs), "results": results})
    return results


def write_dry_run(output=DEFAULT_OUTPUT):
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    jobs = plan_jobs(specs)
    payload = {
        "schema": "piperx-multitask-fixed-time-plan-v1",
        "trajectory_count": len(specs),
        "family_count": len({spec.family.key for spec in specs}),
        "mode_count": len(STUDY_MODES),
        "job_count": len(jobs),
        "modes": list(STUDY_MODES),
        "trajectories": [_json_spec(spec) for spec in specs],
        "jobs": [{"trajectory": job.spec.key, "mode": job.mode}
                 for job in jobs],
    }
    path = Path(output) / "study_plan.json"
    atomic_json(path, payload)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trajectory")
    parser.add_argument("--mode", choices=STUDY_MODES)
    parser.add_argument("--short-prefix", type=int)
    args = parser.parse_args(argv)
    if args.dry_run:
        print(write_dry_run(args.output))
        return
    run_study(args.output, trajectory=args.trajectory, mode=args.mode,
              short_prefix=args.short_prefix)


if __name__ == "__main__":
    main()
