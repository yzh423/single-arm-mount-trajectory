"""Formal 12-arm study on cap-left and open-box-2 (Big YAM excluded)."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUT = ROOT / "reports/single_arm/twelve_arm_two_single_tasks"
ROBOTS = ("doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5",
          "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "nero",
          "openarm", "piperx", "willow")
TASKS = ("cap-left", "open-box-2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-candidates", type=int, default=8192)
    parser.add_argument("--candidate-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(3, max(2, (os.cpu_count() or 4) // 4)))
    parser.add_argument("--search-policy", choices=("legacy", "best-first"), default="legacy")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run(args: list[str], stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        return subprocess.run(args, cwd=ROOT, stdout=out, stderr=err).returncode


def write_progress(rows: list[dict]) -> None:
    (OUT / "matrix.partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _task_fingerprint(robot: str, task: str, args: argparse.Namespace) -> str:
    import numpy as np
    from scripts.formal_run_acceleration import input_fingerprint
    from scripts.solve_strict_urdf_task_cache import target_poses_for
    from scripts.strict_urdf_model_audit import MODELS

    entry = MODELS[robot]
    files = [entry.path,
             ROOT / "reports/single_arm/dense_search_results.json",
             ROOT / "scripts/strict_mujoco_ik.py", ROOT / "scripts/strict_mujoco_model.py",
             ROOT / "scripts/rolling_multibranch_ik.py",
             ROOT / "scripts/strict_trajectory_sources.py", ROOT / "scripts/search_strict_urdf_mount.py",
             ROOT / "scripts/solve_strict_urdf_task_cache.py",
             ROOT / "design_optimization/hierarchical_mount_search.py",
             ROOT / "design_optimization/best_first_mount_search.py",
             ROOT / "design_optimization/incremental_mount_evaluator.py",
             ROOT / "design_optimization/episode_follow_metrics.py"]
    files.extend(path for path in entry.path.parent.rglob("*") if path.is_file())
    trajectory_hash = hashlib.sha256()
    for split in ("validation", "test"):
        xyz, quat, time_s, _ = target_poses_for("local", task, split=split)
        for array in (xyz, quat, time_s):
            trajectory_hash.update(np.ascontiguousarray(array).tobytes())
    return input_fingerprint(files, {
        "robot": robot, "task": task, "trajectory": trajectory_hash.hexdigest(),
        "urdf_candidates": 128, "screen_frames": 64, "medium_frames": 256,
        "search_policy": args.search_policy,
        "planner": "rolling_multibranch", "branch_candidates": 8,
        "horizon_frames": 12, "beam_width": 8,
        "velocity_limit_deg_s": 720.0, "frame_jump_cap_deg": 25.0,
    })


def _render_fingerprint(robot: str, task: str, solution_fingerprint: str) -> str:
    from scripts.formal_run_acceleration import render_fingerprint

    return render_fingerprint([
        ROOT / "scripts/render_strict_single_arm_task.py",
        ROOT / "scripts/high_quality_robot_scene.py",
    ], {
        "robot": robot,
        "task": task,
        "solution_fingerprint": solution_fingerprint,
        "width": 960,
        "height": 540,
        "fps": 30,
    })


def _stamp_json_fingerprint(path: Path, fingerprint: str) -> None:
    from scripts.formal_run_acceleration import atomic_write_json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["input_fingerprint"] = fingerprint
    atomic_write_json(path, payload)


def _coarse_fingerprint(args: argparse.Namespace) -> str:
    from scripts.formal_run_acceleration import input_fingerprint
    from scripts.strict_urdf_model_audit import MODELS

    files = [ROOT / "data/processed/local_pose_benchmark/pilot_samples.json",
             ROOT / "reports/single_arm/model_audit.json",
             ROOT / "reports/single_arm/collision_proxy_profiles.json",
             ROOT / "scripts/run_thirteen_arm_dense_search.py",
             ROOT / "scripts/strict_trajectory_sources.py"]
    for robot in ROBOTS:
        parent = MODELS[robot].path.parent
        files.extend(path for path in parent.rglob("*") if path.is_file())
    return input_fingerprint(files, {
        "robots": ROBOTS, "tasks": TASKS, "candidates": args.gpu_candidates,
        "candidate_batch": args.candidate_batch, "seeds": 10, "iterations": 60,
    })


def _downstream_stages(
    *, search_returncode: int, solution_current: bool,
    cache_exists: bool, solution_passed: bool, render_current: bool,
) -> tuple[bool, bool]:
    """Return (run_solve, run_render) for the authoritative-search pipeline."""
    del solution_passed
    # Exit 2 is a valid completed episode whose strict result failed the
    # all-frame criterion.  Its current cache must still be rendered as a
    # diagnostic artifact; other non-zero exits may indicate a broken search.
    solution_ready = cache_exists and (
        solution_current or search_returncode in (0, 2))
    return False, bool(solution_ready and not render_current)


def _audit_status_is_pass(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "pass"
    except (OSError, json.JSONDecodeError):
        return False


def _run_task(robot: str, task: str, args: argparse.Namespace) -> dict:
    from scripts.formal_run_acceleration import artifact_is_current, json_fingerprint_matches

    stem = OUT / "logs" / f"{robot}__{task}"
    cache = ROOT / f"videos/single_arm/strict_cache/local/{robot}/{task}.npz"
    audit = cache.with_suffix(".json")
    video = OUT / "videos" / robot / f"{task}.mp4"
    fingerprint = _task_fingerprint(robot, task, args)
    render_fingerprint = _render_fingerprint(robot, task, fingerprint)
    video_audit = video.with_suffix(".json")
    solution_current = (args.resume and artifact_is_current(cache, audit, fingerprint)
                        and _audit_status_is_pass(audit))
    render_current = (args.resume and video.is_file()
                      and json_fingerprint_matches(video_audit, render_fingerprint))
    child_env = os.environ.copy()
    child_env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    def task_run(command: list[str], stdout: Path, stderr: Path) -> int:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            return subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, env=child_env).returncode
    search = solve = render = 0
    if not solution_current:
        search = task_run([
            PYTHON, "scripts/search_strict_urdf_mount.py", "--domain", "local",
            "--robot", robot, "--task", task, "--search-policy", args.search_policy,
            "--incumbent-json", "reports/single_arm/dense_search_results.json",
            "--input-fingerprint", fingerprint,
        ], stem.with_suffix(".search.stdout.log"), stem.with_suffix(".search.stderr.log"))
    search_cache_current = artifact_is_current(cache, audit, fingerprint)
    run_solve, run_render = _downstream_stages(
        search_returncode=search, solution_current=solution_current,
        cache_exists=search_cache_current, solution_passed=_audit_status_is_pass(audit),
        render_current=render_current)
    if run_solve:
        raise AssertionError("search is the authoritative solver; redundant solve is disabled")
    if run_render:
        render = task_run([PYTHON, "scripts/render_strict_single_arm_task.py", "--domain", "local",
                           "--robot", robot, "--task", task, "--output", str(video)],
                          stem.with_suffix(".render.stdout.log"), stem.with_suffix(".render.stderr.log"))
        if render == 0 and video_audit.is_file():
            _stamp_json_fingerprint(video_audit, render_fingerprint)
    return {"robot": robot, "task": task, "gpu_candidates": args.gpu_candidates,
            "urdf_candidates": 128, "search_returncode": search, "solve_returncode": solve,
            "render_returncode": render, "resumed": solution_current and render_current,
            "resumed_solution": solution_current, "resumed_render": render_current,
            "input_fingerprint": fingerprint, "render_fingerprint": render_fingerprint,
            "cache": str(cache), "video": str(video)}


def main() -> None:
    args = build_parser().parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts.formal_run_acceleration import json_fingerprint_matches
    coarse_output = ROOT / "reports/single_arm/dense_search_results.json"
    coarse_fingerprint = _coarse_fingerprint(args)
    if args.resume and json_fingerprint_matches(coarse_output, coarse_fingerprint):
        coarse = 0
        (OUT / "gpu_coarse.stdout.log").write_text("resumed matching GPU coarse result\n", encoding="utf-8")
        (OUT / "gpu_coarse.stderr.log").write_text("", encoding="utf-8")
    else:
        coarse = run([
            PYTHON, "scripts/run_thirteen_arm_dense_search.py", "--candidates", str(args.gpu_candidates),
            "--candidate-batch", str(args.candidate_batch), "--seeds", "10", "--iterations", "60",
            "--input-fingerprint", coarse_fingerprint,
            "--robots", *ROBOTS, "--tasks", *TASKS,
        ], OUT / "gpu_coarse.stdout.log", OUT / "gpu_coarse.stderr.log")
    if coarse:
        raise SystemExit(f"GPU coarse search failed: {coarse}")

    rows: list[dict] = []
    jobs = [(robot, task) for robot in ROBOTS for task in TASKS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_task, robot, task, args): (robot, task) for robot, task in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row); rows.sort(key=lambda item: jobs.index((item["robot"], item["task"])))
            write_progress(rows); print(json.dumps(row), flush=True)
    (OUT / "matrix.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
