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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finalists", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); formal = json.loads(args.finalists.read_text())
    robot = formal["config"]["topology"]
    candidate = sorted(formal["pareto_candidates"], key=lambda row: row["objectives"][0])[0]
    template = load_templates(ROOT / "reports/parametric_topology_audit.json")[robot]
    design = assemble_from_deltas(template, torch.tensor(candidate["strict_deltas_m"]))
    left_base, right_base = mirrored_mount_transforms(
        torch.tensor([candidate["base_spacing_m"]]), torch.tensor([candidate["base_y_m"]]),
        torch.tensor([candidate["base_z_m"]]),
        torch.deg2rad(torch.tensor([candidate["mount_roll_deg"]])))
    trace = np.load(args.trace); result = {"robot": robot, "trace": str(args.trace), "arms": {}}
    for side, base, key in (("left", left_base[0], "left_q"),
                            ("right", right_base[0], "right_q")):
        elbow = joint_world_positions(design, torch.tensor(trace[key]))[0][:, 2]
        elbow = (base[:3, :3] @ elbow[..., None]).squeeze(-1) + base[:3, 3]
        inward = elbow[:, 0] - base[0, 3] if side == "left" else base[0, 3] - elbow[:, 0]
        result["arms"][side] = {"inward_frame_fraction": float((inward > 0).float().mean()),
                                "mean_signed_inward_mm": float(inward.mean() * 1000),
                                "maximum_inward_mm": float(inward.max() * 1000)}
    output = args.output or args.trace.with_name(args.trace.stem + "_outer_elbow.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
