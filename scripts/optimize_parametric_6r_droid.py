"""Optimize four topology-preserving 6R geometries on DROID SE(3) paths.

This is deliberately a kinematic benchmark.  Every task may choose its best
initial IK branch (equivalent to an oracle XYZ base placement), while the
orientation and the complete relative Cartesian path are kept unchanged.
Dynamics, speed limits and collision costs belong to the later MPC validation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_parametric_topologies import templates

TRACE = ROOT / "data/DROID/selected_20min/droid_franka_hard_20min.npz"
MANIFEST = ROOT / "data/DROID/selected_20min/droid_franka_hard_20min_manifest.json"


def quat_rotation(qwxyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(qwxyz[[1, 2, 3, 0]]).as_matrix()


def sampled_tasks(samples_per_task: int) -> list[dict]:
    trace = np.load(TRACE)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = []
    for segment in manifest["segments"]:
        ids = np.unique(np.linspace(segment["start_frame"], segment["end_frame"], samples_per_task).astype(int))
        p = trace["tcp_xyz"][ids]
        r = np.stack([quat_rotation(q) for q in trace["tcp_quat_wxyz"][ids]])
        tasks.append({
            "category": segment["category"],
            "instruction": segment["instruction"],
            "duration_s": segment["duration_s"],
            "dp": p - p[0],
            "dR": r @ r[0].T,
        })
    return tasks


def seed_bank(model, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    lo = np.maximum(model.qmin, -np.pi)
    hi = np.minimum(model.qmax, np.pi)
    pool = rng.uniform(lo, hi, size=(max(64, count * 12), 6))
    ranked = []
    for q in pool:
        tcp = model.fk(q)[:3, 3]
        sigma = np.linalg.svd(model.jacobian(q), compute_uv=False)[-1]
        # Reject folded-up and extreme-boundary starts; retain diverse useful branches.
        radial = np.linalg.norm(tcp)
        if 0.22 < radial < 0.72:
            ranked.append((sigma + 0.08 * radial, q))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in ranked[:count]]


def follow_task(model, task: dict, seed: np.ndarray) -> dict:
    q = seed.copy()
    start = model.fk(q)
    pos_err, rot_err, sigma, travel = [], [], [], 0.0
    for dp, dR in zip(task["dp"], task["dR"]):
        target = np.eye(4)
        target[:3, 3] = start[:3, 3] + dp
        target[:3, :3] = dR @ start[:3, :3]
        qn, error, smin = model.solve(target, q, iters=42, damping=0.018)
        travel += float(np.linalg.norm(qn - q))
        q = qn
        pos_err.append(np.linalg.norm(error[:3]))
        rot_err.append(np.linalg.norm(error[3:]))
        sigma.append(smin)
    pe, re = np.asarray(pos_err), np.asarray(rot_err)
    return {
        "success_fraction": float(np.mean((pe < .015) & (re < np.radians(5)))),
        "position_rmse_mm": float(1000 * np.sqrt(np.mean(pe * pe))),
        "orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(re * re)))),
        "sigma_min": float(np.min(sigma)),
        "joint_travel_rad": travel,
    }


def evaluate(model, tasks: list[dict], seeds: list[np.ndarray]) -> tuple[float, dict]:
    rows = []
    for task in tasks:
        attempts = [follow_task(model, task, q) for q in seeds]
        best = min(attempts, key=lambda x: (
            -x["success_fraction"], x["position_rmse_mm"] + 2 * x["orientation_rmse_deg"],
            -x["sigma_min"], x["joint_travel_rad"],
        ))
        rows.append(best)
    success = np.mean([x["success_fraction"] for x in rows])
    pos = np.mean([x["position_rmse_mm"] for x in rows])
    rot = np.mean([x["orientation_rmse_deg"] for x in rows])
    sigma = np.percentile([x["sigma_min"] for x in rows], 10)
    travel = np.mean([x["joint_travel_rad"] for x in rows])
    score = 500 * (1 - success) + pos + 2 * rot + 30 * max(0, .025 - sigma) + .08 * travel
    return float(score), {
        "path_success_fraction": float(success),
        "mean_position_rmse_mm": float(pos),
        "mean_orientation_rmse_deg": float(rot),
        "sigma_min_p10": float(sigma),
        "mean_joint_travel_rad": float(travel),
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", choices=("doosan", "xarm6", "ur5", "kinova"))
    parser.add_argument("--candidates", type=int, default=160)
    parser.add_argument("--samples-per-task", type=int, default=9)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260804)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    base = templates()[args.topology]
    tasks = sampled_tasks(args.samples_per_task)
    active = np.linalg.norm(base.deltas, axis=1) > 1e-5
    candidates = [(np.ones(6), .725)]
    for _ in range(args.candidates - 1):
        scales = np.ones(6)
        scales[active] = np.exp(rng.uniform(np.log(.55), np.log(1.60), active.sum()))
        candidates.append((scales, rng.uniform(.700, .750)))

    tic = time.perf_counter()
    results = []
    for index, (scales, reach) in enumerate(candidates):
        model = base.design(scales, reach)
        # Every geometry sees exactly the same random joint pool.  Ranking may
        # change with its Jacobian, but no candidate wins by RNG luck.
        seeds = seed_bank(model, args.branches, np.random.default_rng(args.seed + 991))
        score, metrics = evaluate(model, tasks, seeds)
        # Discourage pathological one-link designs without imposing a preferred topology.
        nonzero = np.linalg.norm(model.deltas[active], axis=1)
        regularity = float(np.std(np.log(np.maximum(nonzero, 1e-6))))
        score += 2.0 * regularity
        results.append({
            "candidate": index,
            "score": score,
            "reach_m": reach,
            "scales": scales.tolist(),
            "joint_and_tcp_deltas_m": model.deltas.tolist(),
            "metrics": metrics,
        })
        if index % 10 == 0:
            print(args.topology, index, f"best={min(x['score'] for x in results):.3f}", flush=True)
    baseline = next(x for x in results if x["candidate"] == 0)
    results.sort(key=lambda x: x["score"])
    output = {
        "topology": args.topology,
        "method": "topology-preserving random geometry search; per-task best initial branch",
        "trace": str(TRACE.relative_to(ROOT)),
        "tasks": len(tasks),
        "samples_per_task": args.samples_per_task,
        "candidates": args.candidates,
        "compute_s": time.perf_counter() - tic,
        "baseline": baseline,
        "best": results[0],
        "top10": results[:10],
    }
    out = ROOT / "offline_results" / f"geometry_700_750_{args.topology}_droid.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k not in ("best", "top10")}, indent=2))
    print(json.dumps(results[0], indent=2))
    print(out)


if __name__ == "__main__":
    main()
