from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINALISTS = {
    "doosan": ROOT / "runs/pose_roll_formal/doosan/pareto_finalists.json",
    "xarm6": ROOT / "runs/pose_roll_formal/xarm6/pareto_finalists.json",
    "ur5": ROOT / "runs/pose_roll_formal/ur5/pareto_finalists.json",
    "kinova": ROOT / "runs/pose_roll_formal/kinova_guided/pareto_finalists.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated held-out frame resampling audit")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports/learning/formal_resampling")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for robot, finalists in FINALISTS.items():
        rows = []
        for repeat in range(args.repeats):
            seed = 105479 + 1009 * repeat
            output = args.output_dir / f"{robot}_test_seed_{seed}.json"
            command = [sys.executable, str(ROOT / "scripts/validate_pareto_candidate_cuda.py"),
                       str(finalists), "--split", "test", "--frames", str(args.frames),
                       "--sample-seed", str(seed), "--output", str(output)]
            print(f"[{robot}] resample {repeat + 1}/{args.repeats}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            rows.append(json.loads(output.read_text(encoding="utf-8"))["metrics"])
        summary = {}
        for key in ("success_rate", "position_rmse_mm", "orientation_rmse_deg", "sigma_mean"):
            values = np.asarray([row[key] for row in rows])
            summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                            "minimum": float(values.min()), "maximum": float(values.max())}
        all_rows[robot] = {"repeats": args.repeats, "frames_per_repeat": args.frames,
                           "summary": summary}
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
