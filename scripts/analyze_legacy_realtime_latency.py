from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FILES = {
    "doosan_hybrid": "doosan_hybrid_branch_50hz_sim_mpc_pvt_perf.csv",
    "doosan_local": "doosan_spacing084_50hz_sim_mpc_pvt_perf.csv",
    "xarm6_local": "xarm6_spacing056_50hz_sim_mpc_pvt_perf.csv",
    "ur5_local": "ur5_spacing050_50hz_sim_mpc_pvt_perf.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/learning/hourly/1200_legacy_50hz_latency.json"))
    args = parser.parse_args(); result = {}
    for name, filename in FILES.items():
        path = args.logs / filename; data = pd.read_csv(path)
        control = pd.to_numeric(data["control_ms"], errors="coerce").dropna().to_numpy()
        overrun = pd.to_numeric(data["overrun_ms"], errors="coerce").fillna(0).to_numpy()
        result[name] = {"source": str(path), "samples": len(control),
                        "control_ms": {f"p{p}": float(np.percentile(control, p))
                                       for p in (50, 95, 99, 100)},
                        "deadline_miss_fraction_20ms": float((overrun > 0).mean())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
