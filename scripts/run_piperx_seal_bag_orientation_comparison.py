"""Run a resumable, equal-budget Piper X Seal Bag mount-orientation study."""
from __future__ import annotations

import json
from pathlib import Path
import time

from factory_bimanual.orientation_mount_search import (
    OrientationSearchConfig, evenly_spaced, generate_mounts, mount_fingerprint,
)
from scripts.search_fold_box_piperx_mount import (
    local_paired_refinements, rank_paired_mount_candidate,
)
from scripts.search_fold_box_piperx_paired_mount import (
    _sparse_safe, _valid_mount, evaluate_pair,
)
from scripts.search_seal_bag_piperx_paired_mount import _registered_seal_bag_task


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (ROOT / "reports/factory_bimanual"
          / "piperx_seal_bag_mount_orientation_comparison")
RESULTS = OUTPUT / "results"


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load(path: Path, fallback):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def _run_record(task, mode, stage, mount, serial, settings):
    started = time.perf_counter()
    record = evaluate_pair(task, mount, serial, **settings)
    record.update({
        "mode": mode, "stage": stage,
        "mount_fingerprint": mount_fingerprint(mount),
        "wall_time_s": time.perf_counter() - started,
    })
    return record


def run_mode(task, mode, config: OrientationSearchConfig):
    path = RESULTS / f"{mode}.json"
    state = _load(path, {"mode": mode, "records": [], "status": "running"})
    existing = {(row["stage"], row["mount_fingerprint"]): row
                for row in state["records"]}
    serial_base = config.modes.index(mode) * 10000

    def evaluate(stage, mounts, settings):
        rows = []
        for index, mount in enumerate(mounts):
            key = (stage, mount_fingerprint(mount))
            if key in existing:
                rows.append(existing[key]); continue
            record = _run_record(
                task, mode, stage, mount,
                serial_base + len(state["records"]), settings)
            state["records"].append(record); existing[key] = record; rows.append(record)
            _atomic_json(path, state)
            print(mode, stage, index + 1, len(mounts),
                  f"coverage={record['continuous_pair_coverage']:.4f}",
                  f"collision={record['pair_collision_frames'] + record['pair_edge_collision_frames']}",
                  flush=True)
        return rows

    coarse = evaluate("coarse", generate_mounts(mode, task, config), {
        "uniform_count": 8, "global_seed_count": 1,
        "max_iterations": 45, "maximum_candidates": 1,
        "constrained_fallback_enabled": False})
    seeds = sorted((row for row in coarse if _sparse_safe(row)),
                   key=rank_paired_mount_candidate)[:6]
    dense = evaluate("dense", [row["mount"] for row in seeds], {
        "uniform_count": 30, "global_seed_count": 6,
        "max_iterations": 100, "maximum_candidates": 3,
        "constrained_fallback_enabled": True})
    if not dense:
        raise RuntimeError(f"{mode}: no collision-free dense seeds")
    best_dense = min((row for row in dense if _sparse_safe(row)),
                     key=rank_paired_mount_candidate)
    local_mounts = []
    for side in ("left", "right"):
        for mount in local_paired_refinements(
                best_dense["mount"], side=side,
                xy_step_m=.04, yaw_step_deg=10.):
            mount["mode"] = mode
            mount["base_z_m"] = {"left": mount["shared_base_z_m"],
                                 "right": mount["shared_base_z_m"]}
            if _valid_mount(mount):
                local_mounts.append(mount)
    local = evaluate("local", evenly_spaced(local_mounts, 12), {
        "uniform_count": 36, "global_seed_count": 7,
        "max_iterations": 110, "maximum_candidates": 4,
        "constrained_fallback_enabled": True})
    finalists = sorted(
        (row for row in [*dense, *local] if _sparse_safe(row)),
        key=rank_paired_mount_candidate)[:4]
    if not finalists:
        raise RuntimeError(f"{mode}: no collision-free execution shortlist")
    state["execution_shortlist"] = [row["mount"] for row in finalists]
    state["selected_mount"] = finalists[0]["mount"]
    state["status"] = "shortlist_complete"
    _atomic_json(path, state)
    return state


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    task = _registered_seal_bag_task()
    config = OrientationSearchConfig()
    manifest = {"task": "seal_bag", "robot": "piperx",
                "source_rows": len(task.time_s),
                "config": {"modes": list(config.modes),
                           "shared_z_values_m": list(config.shared_z_values_m),
                           "maximum_candidates": config.maximum_candidates},
                "modes": {}}
    for mode in config.modes:
        manifest["modes"][mode] = run_mode(task, mode, config)
        _atomic_json(RESULTS / "manifest.json", manifest)
    manifest["status"] = "complete"
    _atomic_json(RESULTS / "manifest.json", manifest)


if __name__ == "__main__":
    main()
