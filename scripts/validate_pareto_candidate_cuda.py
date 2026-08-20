from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.egodex import EgoDexPoseDataset
from design_optimization.kinematics import assemble_from_deltas
from design_optimization.objectives import evaluate_population
from design_optimization.taskspace import canonical_egodex_targets, mirrored_mount_transforms
from design_optimization.topology import load_templates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finalists", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); payload = json.loads(args.finalists.read_text())
    config = payload["config"]; robot = config["topology"]
    candidates = sorted(payload["pareto_candidates"], key=lambda row: row["objectives"][0])
    candidate = candidates[args.candidate]; device = torch.device(args.device)
    dataset = EgoDexPoseDataset(config["data"]["path"]); episodes = getattr(dataset.split(), args.split)
    sample_seed = (args.sample_seed if args.sample_seed is not None
                   else config["data"]["seed"] + 104729)
    indices = dataset.sample_indices(episodes, args.frames, sample_seed, .8)
    arrays = [torch.as_tensor(np.asarray(dataset.data[key][indices]).copy(), device=device)
              for key in ("left_relative_xyz", "left_relative_quat_wxyz",
                          "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*arrays)
    template = load_templates(ROOT / "reports" / "parametric_topology_audit.json",
                              device=device)[robot]
    design = assemble_from_deltas(template, torch.tensor(candidate["strict_deltas_m"], device=device))
    p = torch.tensor(candidate["parameters"], device=device)
    spacing = torch.tensor([candidate["base_spacing_m"]], device=device)
    by = torch.tensor([candidate["base_y_m"]], device=device)
    bz = torch.tensor([candidate["base_z_m"]], device=device)
    roll = torch.deg2rad(torch.tensor([candidate["mount_roll_deg"]], device=device))
    left, right = mirrored_mount_transforms(spacing, by, bz, roll)
    metrics = evaluate_population(design, task, left_base_world=left, right_base_world=right,
                                  seed_count=config["ik"]["seeds"],
                                  ik_iterations=config["ik"]["iterations"],
                                  compute_singular_values=True,
                                  collision_aware_branch_selection=config["constraints"].get(
                                      "collision_aware_branch_selection", False),
                                  outer_elbow_branch_weight=config["constraints"].get(
                                      "outer_elbow_branch_weight", 0.0))
    result = {"robot": robot, "split": args.split, "frames": len(indices),
              "sample_seed": sample_seed,
              "source_finalists": str(args.finalists), "candidate_rank": args.candidate,
              "mount_roll_deg": candidate["mount_roll_deg"],
              "base_spacing_m": candidate["base_spacing_m"],
              "metrics": {"success_rate": float(metrics.success_rate[0]),
                          "position_rmse_mm": float(1000 * metrics.position_rmse_m[0]),
                          "orientation_rmse_deg": float(torch.rad2deg(metrics.orientation_rmse_rad[0])),
                          "sigma_mean": float(metrics.sigma_mean[0]),
                          "proxy_collision_fraction": float(metrics.collision_frame_fraction[0]),
                          "proxy_minimum_clearance_mm": float(1000 * metrics.minimum_collision_clearance_m[0])}}
    output = args.output or args.finalists.parent / f"{args.split}_candidate_validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
