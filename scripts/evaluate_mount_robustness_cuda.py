"""Audit a fixed finalist under deterministic installation perturbations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from design_optimization.egodex import EgoDexPoseDataset
from design_optimization.kinematics import assemble_from_deltas
from design_optimization.objectives import evaluate_population
from design_optimization.robustness import (symmetric_sobol_perturbations,
                                            upper_tail_cvar)
from design_optimization.taskspace import canonical_egodex_targets, mirrored_mount_transforms
from design_optimization.topology import load_templates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finalists", type=Path)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--perturbations", type=int, default=33)
    parser.add_argument("--spacing-mm", type=float, default=20.0)
    parser.add_argument("--y-mm", type=float, default=10.0)
    parser.add_argument("--z-mm", type=float, default=10.0)
    parser.add_argument("--roll-deg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.finalists.read_text())
    config, robot = payload["config"], payload["config"]["topology"]
    candidate = sorted(payload["pareto_candidates"],
                       key=lambda row: row["objectives"][0])[args.candidate]
    dataset = EgoDexPoseDataset(config["data"]["path"])
    episodes = getattr(dataset.split(), args.split)
    indices = dataset.sample_indices(episodes, args.frames, args.seed, .8)
    arrays = [torch.as_tensor(np.asarray(dataset.data[key][indices]).copy(),
                              device=args.device) for key in
              ("left_relative_xyz", "left_relative_quat_wxyz",
               "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*arrays)
    dtype = task.left.dtype
    template = load_templates(ROOT / "reports/parametric_topology_audit.json",
                              device=args.device, dtype=dtype)[robot]
    perturbation = symmetric_sobol_perturbations(
        args.perturbations, 4, seed=args.seed, device=args.device, dtype=dtype)
    scale = torch.tensor((args.spacing_mm / 1000, args.y_mm / 1000,
                          args.z_mm / 1000, np.deg2rad(args.roll_deg)),
                         device=args.device, dtype=dtype)
    delta = perturbation * scale
    spacing = torch.as_tensor(candidate["base_spacing_m"], device=args.device,
                              dtype=dtype) + delta[:, 0]
    base_y = torch.as_tensor(candidate["base_y_m"], device=args.device,
                             dtype=dtype) + delta[:, 1]
    base_z = torch.as_tensor(candidate["base_z_m"], device=args.device,
                             dtype=dtype) + delta[:, 2]
    roll = torch.clamp(torch.deg2rad(torch.tensor(candidate["mount_roll_deg"],
                                                 device=args.device, dtype=dtype)) + delta[:, 3],
                       0.0, torch.pi)
    left, right = mirrored_mount_transforms(spacing, base_y, base_z, roll)
    deltas = torch.tensor(candidate["strict_deltas_m"], device=args.device,
                          dtype=dtype)
    design = assemble_from_deltas(template, deltas[None].expand(len(delta), -1, -1))
    metrics = evaluate_population(
        design, task, left_base_world=left, right_base_world=right,
        seed_count=config["ik"]["seeds"], ik_iterations=config["ik"]["iterations"],
        compute_singular_values=True,
        collision_aware_branch_selection=config["constraints"].get(
            "collision_aware_branch_selection", True))
    fields = {
        "failure_rate": 1 - metrics.success_rate,
        "position_rmse_mm": 1000 * metrics.position_rmse_m,
        "orientation_rmse_deg": torch.rad2deg(metrics.orientation_rmse_rad),
        "inverse_sigma_mean": 1 / metrics.sigma_mean.clamp_min(1e-6),
        "collision_fraction": metrics.collision_frame_fraction,
        "clearance_margin_violation_mm": 1000 * metrics.collision_margin_violation_m,
    }
    summary = {}
    for name, value in fields.items():
        summary[name] = {"nominal": float(value[0]), "worst": float(value.max()),
                         "cvar_worst_20pct": float(upper_tail_cvar(value, .2))}
    result = {
        "robot": robot, "split": args.split, "frames": len(indices),
        "candidate_rank": args.candidate, "seed": args.seed,
        "perturbation_bounds": {"spacing_mm": args.spacing_mm, "y_mm": args.y_mm,
                                "z_mm": args.z_mm, "roll_deg": args.roll_deg},
        "perturbations": len(delta), "summary": summary,
        "samples": [{"normalized_delta": perturbation[index].cpu().tolist(),
                     **{name: float(value[index]) for name, value in fields.items()}}
                    for index in range(len(delta))],
    }
    output = args.output or (ROOT / "reports/learning/mount_robustness" /
                             f"{robot}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
