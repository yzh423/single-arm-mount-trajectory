"""Dense local geometry refinement around each topology's current optimum."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("topology", choices=("doosan", "xarm6", "ur5", "kinova"))
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--samples-per-task", type=int, default=15)
    ap.add_argument("--branches", type=int, default=16)
    ap.add_argument("--scale-sigma", type=float, default=.08)
    ap.add_argument("--seed", type=int, default=260812)
    ap.add_argument("--geometry-source", default="refined", help="input geometry JSON suffix")
    ap.add_argument("--output-suffix", default="local_dense", help="output geometry JSON suffix")
    a = ap.parse_args()

    source = ROOT / "offline_results" / f"geometry_700_750_{a.topology}_{a.geometry_source}.json"
    saved = json.loads(source.read_text(encoding="utf-8"))["best"]
    base = templates()[a.topology]
    best_scales = np.asarray(saved["scales"], dtype=float)
    active = np.linalg.norm(base.deltas, axis=1) > 1e-5
    rng = np.random.default_rng(a.seed)
    candidates = [(best_scales.copy(), float(saved["reach_m"]))]
    for _ in range(a.candidates - 1):
        scales = best_scales.copy()
        scales[active] *= np.exp(rng.normal(0., a.scale_sigma, active.sum()))
        candidates.append((scales, float(np.clip(rng.normal(saved["reach_m"], .012), .700, .750))))

    tasks = sampled_tasks(a.samples_per_task)
    results = []
    tic = time.perf_counter()
    for index, (scales, reach) in enumerate(candidates):
        model = base.design(scales, reach)
        seeds = seed_bank(model, a.branches, np.random.default_rng(260804 + 991))
        score, metrics = evaluate(model, tasks, seeds)
        lengths = np.linalg.norm(model.deltas[active], axis=1)
        score += 2.0 * float(np.std(np.log(np.maximum(lengths, 1e-6))))
        results.append({
            "candidate": index,
            "score": float(score),
            "reach_m": reach,
            "scales": scales.tolist(),
            "joint_and_tcp_deltas_m": model.deltas.tolist(),
            "metrics": metrics,
        })
        print(a.topology, index + 1, "/", len(candidates), score, flush=True)
    baseline = results[0]
    results.sort(key=lambda x: x["score"])
    out = {
        "topology": a.topology,
        "method": "dense local log-normal geometry refinement",
        "source": str(source.relative_to(ROOT)),
        "candidates": len(candidates),
        "samples_per_task": a.samples_per_task,
        "branches": a.branches,
        "scale_sigma": a.scale_sigma,
        "compute_s": time.perf_counter() - tic,
        "baseline": baseline,
        "best": results[0],
        "ranking": results,
    }
    dst = ROOT / "offline_results" / f"geometry_700_750_{a.topology}_{a.output_suffix}.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()
