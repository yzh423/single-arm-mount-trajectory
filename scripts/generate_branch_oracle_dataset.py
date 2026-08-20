"""Generate a compact, pose-only multi-branch IK oracle corpus on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.collision import self_capsule_clearance, table_capsule_clearance
from design_optimization.egodex import EgoDexPoseDataset
from design_optimization.ik import deterministic_seeds, solve_multistart
from design_optimization.kinematics import assemble_from_deltas
from design_optimization.taskspace import canonical_egodex_targets, mirrored_mount_transforms
from design_optimization.topology import load_templates


ROBOTS = ("doosan", "xarm6", "ur5", "kinova")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", choices=ROBOTS, default=list(ROBOTS))
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports/learning/branch_oracle")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); templates = load_templates(
        ROOT / "reports/parametric_topology_audit.json", device=device)
    for robot in args.robots:
        formal_path = ROOT / "runs/collision_formal" / robot / "pareto_finalists.json"
        selection = json.loads((ROOT / "reports/learning/collision_formal" /
                                f"{robot}_safe_selection.json").read_text())
        rank = int(selection["selected_candidate_rank"])
        formal = json.loads(formal_path.read_text()); candidates = sorted(
            formal["pareto_candidates"], key=lambda row: row["objectives"][0])
        candidate = candidates[rank]; template = templates[robot]
        design = assemble_from_deltas(template, torch.tensor(candidate["strict_deltas_m"], device=device))
        left_base, right_base = mirrored_mount_transforms(
            torch.tensor([candidate["base_spacing_m"]], device=device),
            torch.tensor([candidate["base_y_m"]], device=device),
            torch.tensor([candidate["base_z_m"]], device=device),
            torch.deg2rad(torch.tensor([candidate["mount_roll_deg"]], device=device)))
        dataset = EgoDexPoseDataset(formal["config"]["data"]["path"]); split = dataset.split()
        indices = dataset.sample_indices(split.train, args.frames, 20260806, .8)
        arrays = [torch.as_tensor(np.asarray(dataset.data[key][indices]).copy(), device=device)
                  for key in ("left_relative_xyz", "left_relative_quat_wxyz",
                              "right_relative_xyz", "right_relative_quat_wxyz")]
        task = canonical_egodex_targets(*arrays); payload = {"indices": indices}
        for side, targets, base, seed in (("left", task.left, left_base[0], 20260806),
                                          ("right", task.right, right_base[0], 20260807)):
            local = torch.linalg.inv(base)[None] @ targets
            saved_q, saved_safe, saved_pos, saved_rot = [], [], [], []
            for start in range(0, len(indices), args.chunk):
                stop = min(start + args.chunk, len(indices)); target = local[start:stop]
                result = solve_multistart(
                    design, target, deterministic_seeds(template, args.seeds, seed),
                    iterations=args.iterations)
                q = result.q[0]; B, K = q.shape[:2]
                self_clearance = self_capsule_clearance(design, q.reshape(B * K, 6))[0].reshape(B, K)
                table_clearance = table_capsule_clearance(
                    design, q.reshape(B * K, 6), base)[0].reshape(B, K)
                safe = (self_clearance >= 0) & (table_clearance >= 0)
                cost = 400 * result.position_error_m[0] + 8 * result.orientation_error_rad[0]
                ranked_cost = torch.where(safe, cost, torch.full_like(cost, torch.inf))
                chosen = torch.topk(ranked_cost, min(args.modes, K), largest=False).indices
                saved_q.append(q.gather(1, chosen[..., None].expand(B, chosen.shape[1], 6)).cpu())
                saved_safe.append(safe.gather(1, chosen).cpu())
                saved_pos.append(result.position_error_m[0].gather(1, chosen).cpu())
                saved_rot.append(result.orientation_error_rad[0].gather(1, chosen).cpu())
                print(f"[{robot} {side}] {stop}/{len(indices)}", flush=True)
            payload[f"{side}_target_base"] = local.cpu().numpy()
            payload[f"{side}_q"] = torch.cat(saved_q).numpy()
            payload[f"{side}_safe"] = torch.cat(saved_safe).numpy()
            payload[f"{side}_position_error_m"] = torch.cat(saved_pos).numpy()
            payload[f"{side}_orientation_error_rad"] = torch.cat(saved_rot).numpy()
        output = args.output_dir / f"{robot}_train_oracle.npz"
        np.savez_compressed(output, **payload)
        meta = {"robot": robot, "frames": len(indices), "modes": args.modes,
                "candidate_rank": rank, "source_finalists": str(formal_path),
                "selection_split": "train", "dual_collision_certification": False,
                "note": "per-arm self/table certified; dual-arm pairing remains online"}
        output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(output, flush=True)


if __name__ == "__main__":
    main()
