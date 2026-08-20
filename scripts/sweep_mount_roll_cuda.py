"""Profile 0--180 degree anthropomorphic mount roll with XYZ optimized per angle."""
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
from design_optimization.experiment import save_checkpoint_atomic, seed_everything
from design_optimization.kinematics import assemble_from_deltas
from design_optimization.mount_sweep import mount_quality_score, sobol_mount_candidates
from design_optimization.objectives import evaluate_population
from design_optimization.pareto import rank_population
from design_optimization.taskspace import canonical_egodex_targets, mirrored_mount_transforms
from design_optimization.topology import load_templates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finalists", type=Path)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--mount-candidates", type=int, default=64)
    parser.add_argument("--roll-step-deg", type=float, default=10.)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=70)
    parser.add_argument("--seed", type=int, default=180750)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device, dtype = torch.device(args.device), torch.float32
    seed_everything(args.seed)
    payload = json.loads(args.finalists.read_text(encoding="utf-8"))
    robot = payload["config"]["topology"]
    rows = sorted(payload["pareto_candidates"], key=lambda row: row["objectives"][0])
    candidate = rows[args.candidate]
    output = args.output or ROOT / "reports" / "learning" / "mount_roll_profile" / f"{robot}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".pt")

    dataset = EgoDexPoseDataset(payload["config"]["data"]["path"])
    episodes = getattr(dataset.split(), args.split)
    indices = dataset.sample_indices(episodes, args.frames, args.seed, .8)
    arrays = [torch.as_tensor(np.asarray(dataset.data[key][indices]).copy(), device=device)
              for key in ("left_relative_xyz", "left_relative_quat_wxyz",
                          "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*arrays)
    template = load_templates(ROOT / "reports" / "parametric_topology_audit.json",
                              device=device, dtype=dtype)[robot]
    delta = torch.tensor(candidate["strict_deltas_m"], device=device, dtype=dtype)
    count = args.mount_candidates
    design = assemble_from_deltas(template, delta.repeat(count, 1, 1))
    mounts = sobol_mount_candidates(count, args.seed, device=device, dtype=dtype)
    rolls = torch.arange(0., 180. + .5 * args.roll_step_deg, args.roll_step_deg)
    completed = []
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        completed = state["rows"]
    start = len(completed)
    for roll_deg in rolls[start:]:
        roll = torch.deg2rad(roll_deg.to(device)).expand(count)
        left, right = mirrored_mount_transforms(mounts[:, 0], mounts[:, 1], mounts[:, 2], roll)
        metrics = evaluate_population(
            design, task, left_base_world=left, right_base_world=right,
            seed_count=args.seeds, ik_iterations=args.iterations,
            compute_singular_values=True, collision_aware_branch_selection=True)
        score = mount_quality_score(metrics.success_rate, metrics.position_rmse_m,
                                    metrics.orientation_rmse_rad,
                                    metrics.collision_frame_fraction,
                                    metrics.collision_margin_violation_m)
        winner = int(score.argmin())
        objectives = torch.stack((1 - metrics.success_rate, metrics.position_rmse_m,
                                  metrics.orientation_rmse_rad,
                                  metrics.collision_frame_fraction,
                                  metrics.collision_margin_violation_m), dim=-1)
        # Penetration is a constraint; positive safety-margin shortfall remains
        # a Pareto objective. Keep the full first front so scalar weights do not
        # silently define the scientific conclusion.
        violation = torch.relu(-metrics.minimum_collision_clearance_m)
        front = rank_population(objectives, violation).fronts[0]
        pareto_mounts = []
        for index in front.tolist():
            pareto_mounts.append({
                "candidate_index": index,
                "spacing_m": float(mounts[index, 0]),
                "base_y_m": float(mounts[index, 1]),
                "base_z_m": float(mounts[index, 2]),
                "objectives": objectives[index].detach().cpu().tolist(),
                "minimum_clearance_mm": float(
                    1000 * metrics.minimum_collision_clearance_m[index]),
            })
        completed.append({
            "roll_deg": float(roll_deg), "score": float(score[winner]),
            "spacing_m": float(mounts[winner, 0]), "base_y_m": float(mounts[winner, 1]),
            "base_z_m": float(mounts[winner, 2]),
            "success_rate": float(metrics.success_rate[winner]),
            "position_rmse_mm": float(1000 * metrics.position_rmse_m[winner]),
            "orientation_rmse_deg": float(torch.rad2deg(metrics.orientation_rmse_rad[winner])),
            "sigma_mean": float(metrics.sigma_mean[winner]),
            "collision_fraction": float(metrics.collision_frame_fraction[winner]),
            "minimum_clearance_mm": float(1000 * metrics.minimum_collision_clearance_m[winner]),
            "pareto_mounts": pareto_mounts,
        })
        save_checkpoint_atomic(checkpoint, {"rows": completed, "indices": indices,
                                            "mounts": mounts.cpu(), "seed": args.seed})
        print(f"[{robot}] roll={float(roll_deg):.1f} deg "
              f"success={100*completed[-1]['success_rate']:.2f}% "
              f"pos={completed[-1]['position_rmse_mm']:.2f} mm", flush=True)
    result = {"robot": robot, "source_finalists": str(args.finalists.resolve()),
              "candidate_rank": args.candidate, "split": args.split,
              "sample_seed": args.seed, "sample_indices": indices.tolist(),
              "frames": len(indices), "mount_candidates": count,
              "roll_step_deg": args.roll_step_deg, "ik_seeds": args.seeds,
              "ik_iterations": args.iterations,
              "pareto_objectives": ["failure_rate", "position_rmse_m",
                                     "orientation_rmse_rad", "collision_fraction",
                                     "collision_margin_violation_m"],
              "selection_score": "12*failure + 80*pos + 3*ori + 20*collision + 120*margin",
              "rows": completed}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
