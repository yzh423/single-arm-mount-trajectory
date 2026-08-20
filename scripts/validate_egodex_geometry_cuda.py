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
from design_optimization.kinematics import assemble_from_deltas, build_designs, deterministic_joint_samples
from design_optimization.objectives import evaluate_population
from design_optimization.taskspace import canonical_egodex_targets
from design_optimization.topology import load_templates


DEFAULT_DATA = Path(r"C:\Users\KelvinLM\Documents\Codex\Doosan_Dual_Quest3\data\EgoDex\pose_only_test\egodex_pose_only_test.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out EgoDex validation of four optimized 6R geometries")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--ik-iterations", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "reports" / "learning" / f"heldout_{args.split}.json"
    dataset = EgoDexPoseDataset(args.data)
    episodes = getattr(dataset.split(), args.split)
    indices = dataset.sample_indices(episodes, args.frames, args.seed)
    tensors = [torch.as_tensor(np.asarray(dataset.data[key][indices]).copy(), device=args.device)
               for key in ("left_relative_xyz", "left_relative_quat_wxyz",
                           "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*tensors)
    templates = load_templates(ROOT / "reports" / "parametric_topology_audit.json", device=args.device)
    rows = {}
    for name, template in templates.items():
        learned_path = ROOT / "reports" / "learning" / f"{name}_egodex_optimized.json"
        learned = json.loads(learned_path.read_text(encoding="utf-8"))
        optimized = assemble_from_deltas(template, torch.tensor(learned["optimized_deltas_m"],
                                                                device=args.device))
        baseline = build_designs(template, torch.zeros((1, 6), device=args.device),
                                 deterministic_joint_samples(template, 32768))
        base = learned["best"]
        spacing, by, bz = base["base_spacing_m"], base["base_y_m"], base["base_z_m"]
        left, right = (-spacing / 2, by, bz), (spacing / 2, by, bz)
        rows[name] = {}
        for label, design in (("baseline750", baseline), ("optimized750", optimized)):
            metrics = evaluate_population(design, task, left_base_xyz=left, right_base_xyz=right,
                                          seed_count=args.seeds, ik_iterations=args.ik_iterations,
                                          compute_singular_values=True)
            row = {"success_rate": float(metrics.success_rate[0]),
                   "position_rmse_mm": float(metrics.position_rmse_m[0] * 1000),
                   "orientation_rmse_deg": float(torch.rad2deg(metrics.orientation_rmse_rad[0])),
                   "sigma_mean": float(metrics.sigma_mean[0]),
                   "topology_error_mm": float(metrics.topology_error_m[0] * 1000),
                   "motor_clearance_violation_mm": float(metrics.motor_clearance_violation_m[0] * 1000)}
            row["minimum_collision_clearance_mm"] = float(metrics.minimum_collision_clearance_m[0] * 1000)
            row["collision_frame_fraction"] = float(metrics.collision_frame_fraction[0])
            rows[name][label] = row
            print(f"{name:7s} {label:12s} success={100*row['success_rate']:.1f}% "
                  f"pos={row['position_rmse_mm']:.1f}mm rot={row['orientation_rmse_deg']:.1f}deg")
    payload = {"data": str(args.data), "split": args.split, "frames": args.frames,
               "seed": args.seed, "ik_seeds": args.seeds, "ik_iterations": args.ik_iterations,
               "robots": rows}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
