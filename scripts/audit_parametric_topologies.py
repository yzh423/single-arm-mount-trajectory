"""Build the native-geometry topology audit used by legacy optimizers.

This compatibility audit preserves each robot's native dimensions.  The
thirteen-arm benchmark uses the third-party native URDF registry as
its geometry authority; the historical ``kinova`` key is retained for older
optimization modules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from design_optimization.robot_registry import load_robot_registry
from design_optimization.urdf_chain import load_native_chain


LEGACY_TO_REGISTRY = {
    "doosan": "doosan",
    "xarm6": "xarm6",
    "ur5": "ur5",
    "kinova": "kinova_gen3_lite",
}


def _line_distance(p1: np.ndarray, a1: np.ndarray,
                   p2: np.ndarray, a2: np.ndarray) -> float:
    cross = np.cross(a1, a2)
    norm = np.linalg.norm(cross)
    if norm > 1e-8:
        return float(abs(np.dot(p2 - p1, cross)) / norm)
    return float(np.linalg.norm(np.cross(p2 - p1, a1)))


def build_audit(registry_path: Path) -> dict[str, dict]:
    registry = load_robot_registry(registry_path)
    payload: dict[str, dict] = {}
    for output_name, robot_id in LEGACY_TO_REGISTRY.items():
        chain = load_native_chain(registry[robot_id])
        pair_distances = {
            f"{i + 1}{j + 1}": 1000.0 * _line_distance(
                chain.points[i], chain.axes[i], chain.points[j], chain.axes[j]
            )
            for i in range(chain.dof)
            for j in range(i + 1, chain.dof)
        }
        payload[output_name] = {
            "axes": chain.axes.tolist(),
            "points_m": chain.points.tolist(),
            "flange_deltas_m": chain.deltas.tolist(),
            "flange_home_rotation": chain.home[:3, :3].tolist(),
            "q_min_rad": chain.q_min.tolist(),
            "q_max_rad": chain.q_max.tolist(),
            "pair_distances_mm": pair_distances,
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "configs/robot_registry_13.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/parametric_topology_audit.json"
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = build_audit(args.registry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
