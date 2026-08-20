from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from design_optimization.egodex import EgoDexPoseDataset
from design_optimization.kinematics import build_designs, deterministic_joint_samples, fk_tcp
from design_optimization.taskspace import canonical_egodex_targets, mirrored_mount_transforms
from design_optimization.topology import load_templates


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,nj->ni", transform[:3, :3], points) + transform[:3, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "learning" / "hourly" /
                        "0800_mount_roll_coverage.npz")
    parser.add_argument("--samples", type=int, default=2500)
    args = parser.parse_args(); dtype = torch.float32
    templates = load_templates(ROOT / "reports" / "parametric_topology_audit.json", dtype=dtype)
    dataset = EgoDexPoseDataset(ROOT.parent / "Doosan_Dual_Quest3" / "data" / "EgoDex" /
                                "pose_only_test" / "egodex_pose_only_test.npz")
    split = dataset.split(); indices = dataset.sample_indices(split.test, 500, 807, .8)
    arrays = [torch.as_tensor(np.asarray(dataset.data[key][indices]).copy(), dtype=dtype)
              for key in ("left_relative_xyz", "left_relative_quat_wxyz",
                          "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*arrays); rolls = (0, 45, 90, 135, 180)
    payload = {"rolls_deg": np.asarray(rolls),
               "target_left": task.left[:, :3, 3].numpy(),
               "target_right": task.right[:, :3, 3].numpy()}
    for row, name in enumerate(("doosan", "xarm6", "ur5", "kinova")):
        template = templates[name]; calibration = deterministic_joint_samples(template, 8192)
        design = build_designs(template, torch.zeros((1, 6), dtype=dtype), calibration)
        q = deterministic_joint_samples(template, args.samples, seed=1800 + row)
        cloud = fk_tcp(design, q)[0, :, :3, 3]
        for degrees in rolls:
            roll = torch.deg2rad(torch.tensor([float(degrees)]))
            left, right = mirrored_mount_transforms(torch.tensor([.65]), torch.tensor([0.]),
                                                     torch.tensor([.12]), roll)
            payload[f"{name}_{degrees}_left"] = transform_points(left[0], cloud).numpy()
            payload[f"{name}_{degrees}_right"] = transform_points(right[0], cloud).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.output, **payload)
    print(args.output)


if __name__ == "__main__": main()

