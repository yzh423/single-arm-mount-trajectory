"""Re-rank geometry-search finalists with a much denser fixed IK branch set."""
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
    parser.add_argument("--samples-per-task", type=int, default=9)
    parser.add_argument("--branches", type=int, default=12)
    args = parser.parse_args()
    source = ROOT / "offline_results" / f"geometry_700_750_{args.topology}_droid.json"
    search = json.loads(source.read_text(encoding="utf-8"))
    candidates = [search["baseline"]] + search["top10"]
    # Remove the duplicated baseline if it was itself in the top ten.
    candidates = list({x["candidate"]: x for x in candidates}.values())
    base = templates()[args.topology]
    tasks = sampled_tasks(args.samples_per_task)
    rows = []
    tic = time.perf_counter()
    for index, saved in enumerate(candidates):
        model = base.design(np.asarray(saved["scales"]), float(saved["reach_m"]))
        seeds = seed_bank(model, args.branches, np.random.default_rng(260804 + 991))
        score, metrics = evaluate(model, tasks, seeds)
        rows.append({
            "candidate": saved["candidate"],
            "score": score,
            "reach_m": saved["reach_m"],
            "scales": saved["scales"],
            "flange_deltas_m": model.deltas.tolist(),
            "metrics": metrics,
        })
        print(args.topology, index + 1, "/", len(candidates), saved["candidate"], score, flush=True)
    rows.sort(key=lambda x: x["score"])
    output = {
        "topology": args.topology,
        "source": str(source.relative_to(ROOT)),
        "samples_per_task": args.samples_per_task,
        "branches": args.branches,
        "compute_s": time.perf_counter() - tic,
        "baseline": next(x for x in rows if x["candidate"] == 0),
        "best": rows[0],
        "ranking": rows,
    }
    target = ROOT / "offline_results" / f"geometry_700_750_{args.topology}_refined.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
