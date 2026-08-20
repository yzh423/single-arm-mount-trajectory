from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.kinematics import assemble_from_deltas, joint_world_positions
from design_optimization.taskspace import mirrored_mount_transforms
from design_optimization.topology import load_templates


ROBOTS = ("doosan", "xarm6", "ur5", "kinova")


def finalists(robot: str) -> Path:
    guided = ROOT / f"runs/pose_roll_formal/{robot}_guided/pareto_finalists.json"
    return guided if guided.exists() else ROOT / f"runs/pose_roll_formal/{robot}/pareto_finalists.json"


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,...j->...i", transform[:3, :3], points) + transform[:3, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT / "reports/learning/continuous_retimed")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/learning/hourly/1200_continuous_intrinsic_playback.npz")
    args = parser.parse_args(); templates = load_templates(
        ROOT / "reports/parametric_topology_audit.json")
    payload = {}
    for robot in ROBOTS:
        trace = np.load(args.source / f"{robot}_test_episode_563.npz")
        meta = json.loads((args.source / f"{robot}_test_episode_563.json").read_text())
        source_finalists = Path(meta.get("source_finalists", finalists(robot)))
        if not source_finalists.is_absolute(): source_finalists = ROOT / source_finalists
        formal = json.loads(source_finalists.read_text())
        candidates = sorted(formal["pareto_candidates"], key=lambda row: row["objectives"][0])
        candidate = candidates[int(meta.get("candidate_rank", 0))]
        design = assemble_from_deltas(templates[robot], torch.tensor(candidate["strict_deltas_m"]))
        spacing = torch.tensor([candidate["base_spacing_m"]]); by = torch.tensor([candidate["base_y_m"]])
        bz = torch.tensor([candidate["base_z_m"]]); roll = torch.deg2rad(
            torch.tensor([candidate["mount_roll_deg"]]))
        left_base, right_base = mirrored_mount_transforms(spacing, by, bz, roll)
        left = joint_world_positions(design, torch.tensor(trace["left_q"]))[0]
        right = joint_world_positions(design, torch.tensor(trace["right_q"]))[0]
        payload[f"{robot}_left"] = transform_points(left, left_base[0]).numpy()
        payload[f"{robot}_right"] = transform_points(right, right_base[0]).numpy()
        payload[f"{robot}_left_target"] = trace["left_target"][:, :3, 3]
        payload[f"{robot}_right_target"] = trace["right_target"][:, :3, 3]
        payload[f"{robot}_clearance"] = np.minimum.reduce(
            [trace["self_clearance_m"], trace["dual_clearance_m"], trace["table_clearance_m"]])
        payload[f"{robot}_metrics"] = np.asarray([
            meta["metrics"]["position_rmse_mm"], meta["metrics"]["orientation_rmse_deg"],
            meta["metrics"]["success_rate"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload); print(args.output)


if __name__ == "__main__":
    main()
