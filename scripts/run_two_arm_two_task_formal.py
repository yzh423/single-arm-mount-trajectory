"""Run the formal two-robot/two-task search, held-out solve and rendering."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
JOBS = (("xarm6", "cap-left"), ("xarm6", "open-box-2"),
        ("i2rt_yam", "cap-left"), ("i2rt_yam", "open-box-2"))


def run(command: list[str], stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        return subprocess.run(command, cwd=ROOT, stdout=out, stderr=err).returncode


def main() -> None:
    output = ROOT / "reports/single_arm/two_arm_two_task_formal"
    coarse_code = run([
        PYTHON, "scripts/run_thirteen_arm_dense_search.py", "--candidates", "4096",
        "--candidate-batch", "32", "--seeds", "10", "--iterations", "60",
        "--robots", "xarm6", "i2rt_yam", "--tasks", "cap-left", "open-box-2",
    ], output / "gpu_coarse.stdout.log", output / "gpu_coarse.stderr.log")
    if coarse_code != 0:
        raise SystemExit(f"GPU coarse search failed with return code {coarse_code}")
    rows = []
    for robot, task in JOBS:
        stem = output / f"{robot}__{task}"
        search_code = run([
            PYTHON, "scripts/search_strict_urdf_mount.py", "--domain", "local",
            "--robot", robot, "--task", task, "--candidates", "128",
            "--screen-frames", "64", "--medium-frames", "256",
            "--incumbent-json", "reports/single_arm/dense_search_results.json",
        ], stem.with_suffix(".search.stdout.log"), stem.with_suffix(".search.stderr.log"))
        cache = ROOT / f"videos/single_arm/strict_cache/local/{robot}/{task}.npz"
        solve_code = None
        render_code = None
        video = output / "videos" / robot / f"{task}.mp4"
        if cache.is_file():
            solve_code = run([
                PYTHON, "scripts/solve_strict_urdf_task_cache.py", "--domain", "local",
                "--robot", robot, "--task", task,
            ], stem.with_suffix(".solve.stdout.log"), stem.with_suffix(".solve.stderr.log"))
            render_code = run([
                PYTHON, "scripts/render_strict_single_arm_task.py", "--domain", "local",
                "--robot", robot, "--task", task, "--output", str(video),
            ], stem.with_suffix(".render.stdout.log"), stem.with_suffix(".render.stderr.log"))
        row = {"robot": robot, "task": task, "gpu_coarse_candidates": 4096,
               "urdf_global_candidates": 128, "search_returncode": search_code,
               "solve_returncode": solve_code, "render_returncode": render_code,
               "cache": str(cache), "video": str(video)}
        rows.append(row)
        (output / "matrix.partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    (output / "matrix.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
