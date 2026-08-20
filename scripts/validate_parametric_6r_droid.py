"""Dense hold-out-style validation of saved baseline and optimized geometries."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_parametric_topologies import templates
from scripts.optimize_parametric_6r_droid import evaluate, sampled_tasks, seed_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", choices=("doosan", "xarm6", "ur5", "kinova"))
    parser.add_argument("--samples-per-task", type=int, default=21)
    parser.add_argument("--branches", type=int, default=8)
    args = parser.parse_args()

    refined = ROOT / "offline_results" / f"geometry_700_750_{args.topology}_refined.json"
    source = refined if refined.exists() else ROOT / "offline_results" / f"geometry_700_750_{args.topology}_droid.json"
    search = json.loads(source.read_text(encoding="utf-8"))
    base = templates()[args.topology]
    tasks = sampled_tasks(args.samples_per_task)
    output = {
        "topology": args.topology,
        "source": str(source.relative_to(ROOT)),
        "samples_per_task": args.samples_per_task,
        "branches": args.branches,
        "variants": {},
    }
    tic = time.perf_counter()
    for label in ("baseline", "best"):
        saved = search[label]
        model = base.design(np.asarray(saved["scales"]), float(saved["reach_m"]))
        seeds = seed_bank(model, args.branches, np.random.default_rng(260804 + 991))
        score, metrics = evaluate(model, tasks, seeds)
        output["variants"][label] = {
            "score": score,
            "reach_m": saved["reach_m"],
            "scales": saved["scales"],
            "flange_deltas_m": model.deltas.tolist(),
            "metrics": metrics,
        }
        print(args.topology, label, score, metrics["path_success_fraction"], flush=True)
    output["compute_s"] = time.perf_counter() - tic
    target = ROOT / "offline_results" / f"geometry_700_750_{args.topology}_dense_validation.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
