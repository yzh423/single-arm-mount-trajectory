from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.kinematics import assemble_from_deltas, joint_world_positions
from design_optimization.topology import load_templates


def main() -> None:
    source = ROOT / "reports" / "learning" / "continuous"
    output = ROOT / "reports" / "learning" / "hourly" / "0800_continuous_playback.npz"
    templates = load_templates(ROOT / "reports" / "parametric_topology_audit.json")
    payload = {}
    for name in ("doosan", "xarm6", "ur5", "kinova"):
        trace = np.load(source / f"{name}_test_episode_563.npz")
        meta = json.loads((source / f"{name}_test_episode_563.json").read_text())
        learned = json.loads((ROOT / "reports" / "learning" /
                              f"{name}_egodex_optimized.json").read_text())
        design = assemble_from_deltas(templates[name], torch.tensor(learned["optimized_deltas_m"]))
        left = joint_world_positions(design, torch.tensor(trace["left_q"]))[0].numpy()
        right = joint_world_positions(design, torch.tensor(trace["right_q"]))[0].numpy()
        left += np.asarray(meta["base_left_xyz"])[None, None]
        right += np.asarray(meta["base_right_xyz"])[None, None]
        payload[f"{name}_left"] = left; payload[f"{name}_right"] = right
        payload[f"{name}_left_target"] = trace["left_target"][:, :3, 3]
        payload[f"{name}_right_target"] = trace["right_target"][:, :3, 3]
        payload[f"{name}_clearance"] = np.minimum.reduce(
            [trace["self_clearance_m"], trace["dual_clearance_m"], trace["table_clearance_m"]])
        payload[f"{name}_metrics"] = np.asarray([
            meta["metrics"]["position_rmse_mm"], meta["metrics"]["orientation_rmse_deg"],
            meta["metrics"]["success_rate"]])
    output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(output, **payload)
    print(output)


if __name__ == "__main__": main()

