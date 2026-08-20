"""Formal 12-arm full-episode study on the Local stick-battery task."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUT = ROOT / "reports/single_arm/twelve_arm_stick_battery_formal"
ROBOTS = ("doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5",
          "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "nero",
          "openarm", "piperx", "willow")
TASK = "stick-battery"


def run(args: list[str], stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        return subprocess.run(args, cwd=ROOT, stdout=out, stderr=err).returncode


def write_progress(rows: list[dict[str, object]]) -> None:
    (OUT / "matrix.partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    coarse = run([
        PYTHON, "scripts/run_thirteen_arm_dense_search.py", "--candidates", "4096",
        "--candidate-batch", "32", "--seeds", "10", "--iterations", "60",
        "--robots", *ROBOTS, "--tasks", TASK,
    ], OUT / "gpu_coarse.stdout.log", OUT / "gpu_coarse.stderr.log")
    if coarse:
        raise SystemExit(f"GPU coarse search failed: {coarse}")

    rows: list[dict[str, object]] = []
    for robot in ROBOTS:
        stem = OUT / "logs" / f"{robot}__{TASK}"
        search = run([
            PYTHON, "scripts/search_strict_urdf_mount.py", "--domain", "local",
            "--robot", robot, "--task", TASK, "--candidates", "128",
            "--screen-frames", "64", "--medium-frames", "256",
            "--incumbent-json", "reports/single_arm/dense_search_results.json",
        ], stem.with_suffix(".search.stdout.log"), stem.with_suffix(".search.stderr.log"))
        cache = ROOT / f"videos/single_arm/strict_cache/local/{robot}/{TASK}.npz"
        solve = render = None
        video = OUT / "videos" / robot / f"{TASK}.mp4"
        if cache.is_file():
            solve = run([PYTHON, "scripts/solve_strict_urdf_task_cache.py", "--domain", "local",
                         "--robot", robot, "--task", TASK],
                        stem.with_suffix(".solve.stdout.log"), stem.with_suffix(".solve.stderr.log"))
            render = run([PYTHON, "scripts/render_strict_single_arm_task.py", "--domain", "local",
                          "--robot", robot, "--task", TASK, "--output", str(video)],
                         stem.with_suffix(".render.stdout.log"), stem.with_suffix(".render.stderr.log"))
        audit_path = cache.with_suffix(".json")
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
        row = {
            "robot": robot, "task": TASK, "gpu_candidates": 4096, "urdf_candidates": 128,
            "mount_priority": "whole-episode success first", "search_returncode": search,
            "solve_returncode": solve, "render_returncode": render,
            "episode_success": audit.get("episode_success") if audit else None,
            "frame_coverage": audit.get("frame_coverage") if audit else None,
            "failure_reasons": audit.get("failure_reasons") if audit else None,
            "cache": str(cache), "video": str(video),
        }
        rows.append(row); write_progress(rows); print(json.dumps(row), flush=True)
    (OUT / "matrix.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
