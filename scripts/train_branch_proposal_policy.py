"""Train a multi-modal asynchronous IK proposal policy from certified paths.

This is an imitation-learning speedup, not a safety authority.  Every proposal
must still pass exact IK, limits and collision certification before publication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.branch_policy import (MultiBranchIKPolicy, branch_imitation_loss,
                                                wrapped_joint_error)
from design_optimization.experiment import save_checkpoint_atomic, seed_everything
from design_optimization.taskspace import mirrored_mount_transforms


ROBOTS = ("doosan", "xarm6", "ur5", "kinova")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT / "reports/learning/collision_continuous_selected")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/learning/branch_policy/branch_policy.pt")
    args = parser.parse_args(); seed_everything(20260806); device = torch.device(args.device)
    current, targets, topology, desired, groups = [], [], [], [], []
    for topology_id, robot in enumerate(ROBOTS):
        trace = np.load(args.source / f"{robot}_test_episode_563.npz")
        meta = json.loads((args.source / f"{robot}_test_episode_563.json").read_text())
        formal_path = Path(meta["source_finalists"])
        if not formal_path.is_absolute(): formal_path = ROOT / formal_path
        formal = json.loads(formal_path.read_text()); candidates = sorted(
            formal["pareto_candidates"], key=lambda row: row["objectives"][0])
        candidate = candidates[int(meta["candidate_rank"])]
        left_base, right_base = mirrored_mount_transforms(
            torch.tensor([candidate["base_spacing_m"]]), torch.tensor([candidate["base_y_m"]]),
            torch.tensor([candidate["base_z_m"]]),
            torch.deg2rad(torch.tensor([candidate["mount_roll_deg"]])))
        for side_id, (q_key, target_key, base) in enumerate((
                ("left_q", "left_target", left_base[0]),
                ("right_q", "right_target", right_base[0]))):
            q = torch.as_tensor(trace[q_key]); target = torch.as_tensor(trace[target_key])
            local = torch.linalg.inv(base)[None] @ target
            previous = torch.cat((q[:1], q[:-1]), dim=0)
            current.append(previous); targets.append(local); desired.append(q)
            one_hot = torch.zeros((len(q), len(ROBOTS))); one_hot[:, topology_id] = 1
            topology.append(one_hot); groups.extend([(topology_id, side_id, i) for i in range(len(q))])
    current = torch.cat(current).to(device); targets = torch.cat(targets).to(device)
    topology = torch.cat(topology).to(device); desired = torch.cat(desired).to(device)
    # Deterministic temporal hold-out: every fifth frame from each arm/topology.
    test = torch.tensor([frame % 5 == 0 for _, _, frame in groups], device=device)
    train = ~test; model = MultiBranchIKPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(args.epochs):
        output = model(current[train], targets[train], topology[train])
        loss = branch_imitation_loss(output, desired[train])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(current[test], targets[test], topology[test])
        error = wrapped_joint_error(output.candidates_q, desired[test, None])
        per_mode_rms = torch.sqrt(error.square().mean(dim=-1))
        best = per_mode_rms.amin(dim=-1)
        report = {
            "samples": len(current), "train_samples": int(train.sum()),
            "temporal_holdout_samples": int(test.sum()), "modes": model.modes,
            "best_mode_joint_rmse_deg_mean": float(torch.rad2deg(best.mean())),
            "best_mode_joint_rmse_deg_p95": float(torch.rad2deg(torch.quantile(best, .95))),
            "within_10_deg_fraction": float((best < torch.deg2rad(torch.tensor(10., device=device))).float().mean()),
            "role": "uncertified asynchronous proposal; exact solver and collision checker remain mandatory",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint_atomic(args.output, {"state_dict": model.state_dict(), "report": report,
                                         "topologies": ROBOTS, "seed": 20260806})
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
