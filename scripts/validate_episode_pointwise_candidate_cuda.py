from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.egodex import EgoDexPoseDataset
from design_optimization.kinematics import assemble_from_deltas
from design_optimization.objectives import evaluate_population
from design_optimization.taskspace import canonical_egodex_targets, mirrored_mount_transforms
from design_optimization.topology import load_templates
from validate_continuous_egodex_cuda import richest_contiguous_window


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent-frame upper bound on one episode")
    parser.add_argument("finalists", type=Path)
    parser.add_argument("--episode-id", type=int, default=563)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports/learning/episode_pointwise_upper_bound")
    args = parser.parse_args(); payload = json.loads(args.finalists.read_text())
    config = payload["config"]; robot = config["topology"]
    candidate = sorted(payload["pareto_candidates"], key=lambda row: row["objectives"][0])[0]
    dataset = EgoDexPoseDataset(config["data"]["path"]); episode = dataset.episode(args.episode_id)
    indices = richest_contiguous_window(episode, args.max_frames)
    arrays = [torch.as_tensor(episode[key][indices], device=args.device) for key in
              ("left_relative_xyz", "left_relative_quat_wxyz",
               "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*arrays)
    template = load_templates(ROOT / "reports/parametric_topology_audit.json",
                              device=args.device)[robot]
    design = assemble_from_deltas(template, torch.tensor(candidate["strict_deltas_m"],
                                                          device=args.device))
    left, right = mirrored_mount_transforms(
        torch.tensor([candidate["base_spacing_m"]], device=args.device),
        torch.tensor([candidate["base_y_m"]], device=args.device),
        torch.tensor([candidate["base_z_m"]], device=args.device),
        torch.deg2rad(torch.tensor([candidate["mount_roll_deg"]], device=args.device)))
    metrics = evaluate_population(design, task, left_base_world=left, right_base_world=right,
                                  seed_count=args.seeds, ik_iterations=args.iterations,
                                  compute_singular_values=True,
                                  collision_aware_branch_selection=False)
    result = {"robot": robot, "episode_id": args.episode_id,
              "task": str(dataset.tasks[args.episode_id]), "frames": len(indices),
              "seeds": args.seeds, "iterations": args.iterations,
              "metrics": {"success_rate": float(metrics.success_rate[0]),
                          "position_rmse_mm": float(1000 * metrics.position_rmse_m[0]),
                          "orientation_rmse_deg": float(torch.rad2deg(metrics.orientation_rmse_rad[0])),
                          "sigma_mean": float(metrics.sigma_mean[0])}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{robot}_episode_{args.episode_id}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
