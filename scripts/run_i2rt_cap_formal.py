"""Formal CUDA-coarse/MuJoCo-strict I2RT YAM cap-left experiment."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUT = ROOT / "reports/single_arm/i2rt_cap_formal"


def run(args: list[str], stem: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{stem}.stdout.log").open("w", encoding="utf-8") as stdout, \
         (OUT / f"{stem}.stderr.log").open("w", encoding="utf-8") as stderr:
        return subprocess.run(args, cwd=ROOT, stdout=stdout, stderr=stderr).returncode


def main() -> None:
    coarse = run([
        PYTHON, "scripts/run_thirteen_arm_dense_search.py", "--candidates", "4096",
        "--candidate-batch", "32", "--seeds", "10", "--iterations", "60",
        "--robots", "i2rt_yam", "--tasks", "cap-left",
    ], "gpu_coarse")
    if coarse:
        raise SystemExit(coarse)
    search = run([
        PYTHON, "scripts/search_strict_urdf_mount.py", "--domain", "local",
        "--robot", "i2rt_yam", "--task", "cap-left", "--candidates", "128",
        "--screen-frames", "64", "--medium-frames", "256",
        "--incumbent-json", "reports/single_arm/dense_search_results.json",
    ], "urdf_search")
    cache = ROOT / "videos/single_arm/strict_cache/local/i2rt_yam/cap-left.npz"
    solve = render = None
    video = OUT / "i2rt_yam__cap-left.mp4"
    if cache.is_file():
        solve = run([PYTHON, "scripts/solve_strict_urdf_task_cache.py", "--domain", "local",
                     "--robot", "i2rt_yam", "--task", "cap-left"], "test_solve")
        render = run([PYTHON, "scripts/render_strict_single_arm_task.py", "--domain", "local",
                      "--robot", "i2rt_yam", "--task", "cap-left", "--output", str(video)], "render")
    result = {"gpu_coarse_returncode": coarse, "urdf_search_returncode": search,
              "test_solve_returncode": solve, "render_returncode": render,
              "cache": str(cache), "video": str(video)}
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
