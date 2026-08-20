from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.egodex import EgoDexPoseDataset
from design_optimization.ik import rotation_log
from design_optimization.taskspace import canonical_egodex_targets


def summarize(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(values, torch.tensor((.5, .9, .95, .99, 1.), device=values.device))
    return dict(zip(("p50", "p90", "p95", "p99", "maximum"), map(float, quantiles)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=563)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/learning/hourly/1100_egodex_input_dynamics.json")
    args = parser.parse_args()
    dataset = EgoDexPoseDataset(args.data); episode = dataset.episode(args.episode_id)
    arrays = [torch.as_tensor(episode[key]) for key in
              ("left_relative_xyz", "left_relative_quat_wxyz",
               "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*arrays)
    dt = torch.diff(torch.as_tensor(episode["time_s"])).clamp_min(1e-4)
    result = {"episode_id": args.episode_id, "task": str(dataset.tasks[args.episode_id]),
              "frames": len(episode["time_s"]), "source": str(args.data), "arms": {}}
    for name, transforms in (("left", task.left), ("right", task.right)):
        linear = torch.linalg.vector_norm(torch.diff(transforms[:, :3, 3], dim=0), dim=-1) / dt
        relative_rotation = transforms[1:, :3, :3] @ transforms[:-1, :3, :3].transpose(-1, -2)
        angular = torch.linalg.vector_norm(rotation_log(relative_rotation), dim=-1) / dt
        result["arms"][name] = {"linear_m_s": summarize(linear),
                                "angular_rad_s": summarize(angular)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
