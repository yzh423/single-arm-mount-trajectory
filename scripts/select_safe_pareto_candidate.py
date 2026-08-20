"""Select a Pareto candidate on validation only, prioritizing collision safety."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finalists", type=Path)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        help="isolated cache directory for per-candidate validation results")
    args = parser.parse_args(); payload = json.loads(args.finalists.read_text())
    robot = payload["config"]["topology"]
    count = len(payload["pareto_candidates"])
    finalists_hash = hashlib.sha256(args.finalists.read_bytes()).hexdigest()
    cache_root = args.output_dir or (
        ROOT / "reports/learning/collision_formal" / f"{robot}_candidate_validation")
    # Finalists files are intentionally overwritten by resumed searches. A
    # content-addressed subdirectory prevents a later Pareto front from ever
    # reusing validation JSON produced for an earlier front at the same path.
    directory = cache_root / finalists_hash[:16]
    directory.mkdir(parents=True, exist_ok=True); rows = []
    for rank in range(count):
        output = directory / f"candidate_{rank}.json"
        if not output.exists():
            subprocess.run([
                sys.executable, str(ROOT / "scripts/validate_pareto_candidate_cuda.py"),
                str(args.finalists), "--split", "validation", "--frames", str(args.frames),
                "--candidate", str(rank), "--output", str(output)], cwd=ROOT, check=True)
        rows.append(json.loads(output.read_text()))
    # Validation-only lexicographic policy: eliminate observed collisions,
    # prefer candidates meeting the configured 15 mm screening margin, then
    # maximize task success.  A +1 mm near-miss must not beat a +20 mm solution
    # solely through a tiny reach advantage.
    desired_clearance_mm = 1000 * payload["config"]["constraints"].get(
        "collision_margin_m", .015)
    selected = min(rows, key=lambda row: (
        row["metrics"]["proxy_collision_fraction"],
        max(0.0, desired_clearance_mm - row["metrics"]["proxy_minimum_clearance_mm"]),
        -row["metrics"]["success_rate"],
        -row["metrics"]["sigma_mean"]))
    result = {"robot": robot, "selection_split": "validation",
              "selection_frames": args.frames,
              "finalists_path": str(args.finalists.resolve()),
              "finalists_sha256": finalists_hash,
              "validation_cache_dir": str(directory.resolve()),
              "desired_clearance_mm": desired_clearance_mm,
              "selected_candidate_rank": selected["candidate_rank"],
              "selected_validation_metrics": selected["metrics"],
              "all_candidates": rows}
    output = args.output or ROOT / "reports/learning/collision_formal" / f"{robot}_safe_selection.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "all_candidates"}, indent=2))


if __name__ == "__main__":
    main()
