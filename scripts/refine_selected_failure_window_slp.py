"""Repair the worst local window of a saved continuous path with open SLP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from design_optimization.collision import (dual_capsule_clearance,
                                           self_capsule_clearance,
                                           table_capsule_clearance)
from design_optimization.ik import pose_error
from design_optimization.kinematics import assemble_from_deltas, fk_tcp
from design_optimization.slp_refinement import refine_joint_window_slp
from design_optimization.taskspace import mirrored_mount_transforms
from design_optimization.topology import load_templates


def _metrics(design, local_targets, q, left_base, right_base, dt):
    errors = [pose_error(fk_tcp(design, q[side])[0], local_targets[side])
              for side in ("left", "right")]
    position = torch.cat([torch.linalg.vector_norm(error[:, :3], dim=-1)
                          for error in errors])
    orientation = torch.cat([torch.linalg.vector_norm(error[:, 3:], dim=-1)
                             for error in errors])
    self_clearance = torch.minimum(
        self_capsule_clearance(design, q["left"])[0],
        self_capsule_clearance(design, q["right"])[0])
    dual = dual_capsule_clearance(design, q["left"], design, q["right"],
                                  left_base, right_base)[0]
    table = torch.minimum(
        table_capsule_clearance(design, q["left"], left_base)[0],
        table_capsule_clearance(design, q["right"], right_base)[0])
    clearance = torch.minimum(torch.minimum(self_clearance, dual), table)
    dq = torch.cat((torch.diff(q["left"], dim=0), torch.diff(q["right"], dim=0)))
    return {
        "position_rmse_mm": float(torch.sqrt(position.square().mean()) * 1000),
        "position_maximum_mm": float(position.max() * 1000),
        "orientation_rmse_deg": float(torch.rad2deg(torch.sqrt(orientation.square().mean()))),
        "orientation_maximum_deg": float(torch.rad2deg(orientation.max())),
        "success_rate": float(((position <= .0025) &
                               (orientation <= torch.deg2rad(torch.tensor(
                                   1.5, dtype=q["left"].dtype)))).float().mean()),
        "minimum_clearance_mm": float(clearance.min() * 1000),
        "collision_frame_fraction": float((clearance < 0).float().mean()),
        "maximum_joint_step_rad": float(dq.abs().max()),
        "maximum_joint_speed_rad_s": float(dq.abs().max() / dt),
    }, self_clearance, dual, table, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", choices=("doosan", "xarm6"))
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--radius", type=int, default=18)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--force-worst-window", action="store_true",
                        help="refine one worst window even when all frames pass")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/learning/mujoco_30s_slp"))
    args = parser.parse_args()
    finalists_path = ROOT / "runs/collision_formal" / args.robot / "pareto_finalists.json"
    finalists = json.loads(finalists_path.read_text())
    candidate = sorted(finalists["pareto_candidates"],
                       key=lambda row: row["objectives"][0])[args.candidate]
    result = args.input or (ROOT / "reports/learning/mujoco_30s_selected" /
                            f"{args.robot}_test_episode_2243.npz")
    result = result if result.is_absolute() else ROOT / result
    with np.load(result) as archive:
        trace = {key: np.asarray(archive[key]) for key in archive.files}
    dtype, device = torch.float64, "cpu"
    template = load_templates(ROOT / "reports/parametric_topology_audit.json",
                              device=device, dtype=dtype)[args.robot]
    design = assemble_from_deltas(
        template, torch.tensor(candidate["strict_deltas_m"], dtype=dtype))
    left_base, right_base = mirrored_mount_transforms(
        torch.tensor([candidate["base_spacing_m"]], dtype=dtype),
        torch.tensor([candidate["base_y_m"]], dtype=dtype),
        torch.tensor([candidate["base_z_m"]], dtype=dtype),
        torch.deg2rad(torch.tensor([candidate["mount_roll_deg"]], dtype=dtype)))
    bases = {"left": left_base[0], "right": right_base[0]}
    world_targets = {side: torch.tensor(trace[f"{side}_target"], dtype=dtype)
                     for side in ("left", "right")}
    local_targets = {side: torch.linalg.inv(bases[side])[None] @ world_targets[side]
                     for side in ("left", "right")}
    q = {side: torch.tensor(trace[f"{side}_q"], dtype=dtype)
         for side in ("left", "right")}
    dt = float(np.median(np.diff(trace["time_s"])))
    before, _, _, _, _ = _metrics(
        design, local_targets, q, bases["left"], bases["right"], dt)
    rescues = []
    for rescue_index in range(args.max_windows):
        _, _, _, _, errors = _metrics(
            design, local_targets, q, bases["left"], bases["right"], dt)
        scores, failures = [], []
        for error in errors:
            position = torch.linalg.vector_norm(error[:, :3], dim=-1)
            orientation = torch.linalg.vector_norm(error[:, 3:], dim=-1)
            scores.append(400 * position + 8 * orientation)
            failures.append((position > .0025) | (orientation > torch.deg2rad(
                torch.tensor(1.5, dtype=dtype))))
        all_failures = torch.cat(failures)
        if not bool(all_failures.any()):
            if not (args.force_worst_window and rescue_index == 0):
                break
            all_failures = torch.ones_like(all_failures)
        masked_score = torch.where(all_failures, torch.cat(scores),
                                   torch.full_like(torch.cat(scores), -torch.inf))
        flat_bad = int(masked_score.argmax())
        side_index, frame = divmod(flat_bad, len(q["left"]))
        side = ("left", "right")[side_index]
        other = "right" if side == "left" else "left"
        start = max(0, frame - args.radius)
        stop = min(len(q[side]), frame + args.radius + 1)
        other_window = q[other][start:stop]

        def feasible(proposal):
            self_ok = self_capsule_clearance(design, proposal)[0] >= 0
            table_ok = table_capsule_clearance(design, proposal, bases[side])[0] >= 0
            if side == "left":
                dual = dual_capsule_clearance(
                    design, proposal, design, other_window,
                    bases[side], bases[other])[0]
            else:
                dual = dual_capsule_clearance(
                    design, other_window, design, proposal,
                    bases[other], bases[side])[0]
            return bool((self_ok & table_ok & (dual >= 0)).all())

        refinement = refine_joint_window_slp(
            design, local_targets[side][start:stop], q[side][start:stop],
            iterations=args.iterations, maximum_joint_step_rad=.5,
            fix_endpoints=True, feasibility_fn=feasible)
        rescues.append({
            "side": side, "worst_frame": frame, "window": [start, stop],
            "accepted_iterations": refinement.accepted_iterations,
            "attempted_iterations": refinement.attempted_iterations,
            "initial_merit": refinement.initial_merit,
            "final_merit": refinement.final_merit,
            "final_trust_region_rad": refinement.final_trust_region_rad,
            "history": refinement.history,
        })
        if refinement.accepted_iterations == 0:
            break
        q[side][start:stop] = refinement.q
    after, self_clearance, dual, table, _ = _metrics(
        design, local_targets, q, bases["left"], bases["right"], dt)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{args.robot}_test_episode_2243_slp"
    np.savez_compressed(
        stem.with_suffix(".npz"), time_s=trace["time_s"],
        left_q=q["left"].float().numpy(), right_q=q["right"].float().numpy(),
        left_target=trace["left_target"], right_target=trace["right_target"],
        self_clearance_m=self_clearance.float().numpy(),
        dual_clearance_m=dual.float().numpy(), table_clearance_m=table.float().numpy())
    report = {
        "robot": args.robot, "candidate_rank": args.candidate,
        "source": str(result), "before": before, "after": after,
        "rescued_windows": len(rescues), "rescues": rescues,
    }
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rescues"},
                     indent=2))


if __name__ == "__main__":
    main()
