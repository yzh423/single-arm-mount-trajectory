from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.realtime import (ServoObservation, govern_cartesian_target,
                                          schedule_bounded_solve)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/learning/hourly/1200_realtime_scaffold_latency.json")
    args = parser.parse_args(); current = torch.eye(4); requested = torch.eye(4)
    requested[:3, 3] = torch.tensor((.5, -.3, .2))
    observation = ServoObservation(.01, .08, .08, .004)
    latency = np.empty(args.samples)
    for index in range(args.samples):
        started = time.perf_counter_ns()
        schedule = schedule_bounded_solve(observation)
        current = govern_cartesian_target(current, requested, .02, schedule)
        latency[index] = (time.perf_counter_ns() - started) / 1000
    result = {"samples": args.samples, "device": "cpu", "units": "microseconds",
              "p50": float(np.percentile(latency, 50)),
              "p95": float(np.percentile(latency, 95)),
              "p99": float(np.percentile(latency, 99)), "maximum": float(latency.max()),
              "note": "scheduler + SE(3) target governor only; excludes local MPC solve"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
