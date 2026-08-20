"""Measure per-frame collision-free bimanual IK availability before temporal search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.collision import (dual_capsule_clearance, self_capsule_clearance,
                                           table_capsule_clearance)
from design_optimization.egodex import EgoDexPoseDataset
from design_optimization.ik import deterministic_seeds, solve_multistart
from design_optimization.kinematics import assemble_from_deltas
from design_optimization.taskspace import canonical_egodex_targets, mirrored_mount_transforms
from design_optimization.topology import load_templates
from scripts.validate_continuous_egodex_cuda import richest_contiguous_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finalists", type=Path)
    parser.add_argument("--episode-id", type=int, default=563)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); payload = json.loads(args.finalists.read_text())
    candidates = sorted(payload["pareto_candidates"], key=lambda row: row["objectives"][0])
    candidate = candidates[args.candidate]
    robot = payload["config"]["topology"]; device = torch.device(args.device)
    template = load_templates(ROOT / "reports/parametric_topology_audit.json", device=device)[robot]
    design = assemble_from_deltas(template, torch.tensor(candidate["strict_deltas_m"], device=device))
    left_base, right_base = mirrored_mount_transforms(
        torch.tensor([candidate["base_spacing_m"]], device=device),
        torch.tensor([candidate["base_y_m"]], device=device),
        torch.tensor([candidate["base_z_m"]], device=device),
        torch.deg2rad(torch.tensor([candidate["mount_roll_deg"]], device=device)))
    dataset = EgoDexPoseDataset(payload["config"]["data"]["path"])
    episode = dataset.episode(args.episode_id)
    indices = richest_contiguous_window(episode, args.frames)
    arrays = [torch.as_tensor(episode[key][indices], device=device) for key in
              ("left_relative_xyz", "left_relative_quat_wxyz",
               "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*arrays)
    results = []
    for target, base, seed in ((task.left, left_base, 20260806),
                               (task.right, right_base, 20260807)):
        local = torch.linalg.inv(base[0])[None] @ target
        results.append(solve_multistart(
            design, local, deterministic_seeds(template, args.seeds, seed),
            iterations=args.iterations))
    left, right = results; T, K = left.q.shape[1:3]
    lq, rq = left.q[0], right.q[0]
    left_safe = ((self_capsule_clearance(design, lq.reshape(T * K, 6))[0].reshape(T, K) >= 0) &
                 (table_capsule_clearance(design, lq.reshape(T * K, 6), left_base[0])[0]
                  .reshape(T, K) >= 0))
    right_safe = ((self_capsule_clearance(design, rq.reshape(T * K, 6))[0].reshape(T, K) >= 0) &
                  (table_capsule_clearance(design, rq.reshape(T * K, 6), right_base[0])[0]
                   .reshape(T, K) >= 0))
    any_safe, any_safe_success = [], []
    for frame in range(T):
        dual = dual_capsule_clearance(
            design, lq[frame][None, :, None, :], design,
            rq[frame][None, None, :, :], left_base, right_base)[0] >= 0
        safe = left_safe[frame, :, None] & right_safe[frame, None, :] & dual
        successful = left.success[0, frame, :, None] & right.success[0, frame, None, :]
        any_safe.append(safe.any()); any_safe_success.append((safe & successful).any())
    any_safe = torch.stack(any_safe); any_safe_success = torch.stack(any_safe_success)
    result = {
        "robot": robot, "candidate_rank": args.candidate,
        "episode_id": args.episode_id, "frames": T,
        "window_start_frame": int(indices[0]), "window_end_frame": int(indices[-1]),
        "any_collision_free_pair_fraction": float(any_safe.float().mean()),
        "any_collision_free_success_pair_fraction": float(any_safe_success.float().mean()),
        "first_no_collision_free_pair": (int(torch.nonzero(~any_safe)[0])
                                          if (~any_safe).any() else None),
        "first_no_collision_free_success_pair": (int(torch.nonzero(~any_safe_success)[0])
                                                  if (~any_safe_success).any() else None),
        "no_collision_free_pair_frames": torch.nonzero(~any_safe).squeeze(-1).cpu().tolist(),
        "no_collision_free_success_pair_frames": torch.nonzero(
            ~any_safe_success).squeeze(-1).cpu().tolist(),
    }
    output = args.output or ROOT / "reports/learning/collision_formal" / f"{robot}_episode_candidate_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
