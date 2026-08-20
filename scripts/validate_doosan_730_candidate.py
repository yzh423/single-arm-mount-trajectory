"""Build and validate the selected 730 mm M0609-topology candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.four_robot_sim_app import SCENE
from scripts.optimize_doosan_730_geometry import candidate_scene, evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "COFFAIL/benchmark/coffee_dual_active_set2_70s.csv",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--z-scale", type=float, default=0.55)
    parser.add_argument(
        "--scene-output",
        type=Path,
        default=ROOT / "models/doosan_730_candidate_scene.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tuning/doosan_730_full_validation.json",
    )
    args = parser.parse_args()
    candidate_scene(
        SCENE,
        args.scene_output,
        upper=0.355,
        forearm=0.295,
        wrist=0.080,
        spacing=0.700,
        base_height=0.467,
    )
    duration = args.duration if args.duration > 0.0 else 1e9
    metrics = evaluate(
        args.scene_output,
        args.dataset.resolve(),
        duration,
        rate=50.0,
        z_scale=args.z_scale,
    )
    document = {
        "geometry_m": {
            "upper": 0.355,
            "forearm": 0.295,
            "wrist": 0.080,
            "reach": 0.730,
            "base_spacing": 0.700,
            "base_height": 0.467,
        },
        "dataset": str(args.dataset.resolve()),
        "z_scale": args.z_scale,
        "metrics": metrics,
        "scope": "kinematic joint-origin model; visual/collision meshes are not rescaled",
    }
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
