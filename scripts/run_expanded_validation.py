"""Validation-only selection and held-out test for expanded Pareto fronts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ("ur5", "kinova")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--validation-frames", type=int, default=1024)
    parser.add_argument("--test-frames", type=int, default=2048)
    args = parser.parse_args()
    if args.wait_for:
        sentinel = args.wait_for if args.wait_for.is_absolute() else ROOT / args.wait_for
        while not sentinel.exists():
            print(f"[waiting for GPU] {sentinel}", flush=True)
            time.sleep(15)
    report = ROOT / "reports/learning/collision_formal_expanded"
    report.mkdir(parents=True, exist_ok=True)
    for robot in ROBOTS:
        finalists = ROOT / "runs/collision_formal_expanded" / robot / "pareto_finalists.json"
        selection = report / f"{robot}_safe_selection.json"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/select_safe_pareto_candidate.py"),
            str(finalists), "--frames", str(args.validation_frames),
            "--output-dir", str(report / f"{robot}_candidate_validation"),
            "--output", str(selection)], cwd=ROOT, check=True)
        rank = int(json.loads(selection.read_text())["selected_candidate_rank"])
        output = report / f"{robot}_test.json"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/validate_pareto_candidate_cuda.py"),
            str(finalists), "--split", "test", "--frames", str(args.test_frames),
            "--candidate", str(rank), "--output", str(output),
        ], cwd=ROOT, check=True)
    print("[complete]", report, flush=True)


if __name__ == "__main__":
    main()
