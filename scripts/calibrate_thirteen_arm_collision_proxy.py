"""Calibrate coarse capsule proxies from the exact official strict registry."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from design_optimization.kinematics import assemble_from_deltas, deterministic_joint_samples
from design_optimization.collision import self_segment_pair_distances
from scripts.run_thirteen_arm_dense_search import load_official_search_templates


def calibrate(*, sample_count: int = 4096, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    templates = load_official_search_templates(device=device)
    profiles = {}
    for name, template in templates.items():
        design = assemble_from_deltas(template, template.deltas)
        lengths = torch.linalg.vector_norm(template.deltas, dim=1)
        radii = torch.clamp(0.10 * lengths, min=0.012, max=0.035)
        q = deterministic_joint_samples(template, sample_count, seed=20260806)
        distances, pairs = self_segment_pair_distances(design, q, allowed_pairs=[])
        thresholds = torch.tensor(
            [radii[first] + radii[second] for first, second in pairs], device=device
        )
        overlap = (distances[0] < thresholds).float().mean(0)
        allowed = [list(pair) for pair, fraction in zip(pairs, overlap)
                   if float(fraction) >= 0.98]
        profiles[name] = {
            "method": "official_chain_length_scaled_capsules_with_fixed_overlap_exclusion",
            "source_model": str(template.source_model),
            "tcp_authority": template.tcp_authority,
            "radii_m": radii.tolist(),
            "allowed_pairs": allowed,
            "sample_count": len(q),
            "fixed_overlap_threshold": 0.98,
            "observed_overlap_fraction": {
                f"{first}-{second}": float(value)
                for (first, second), value in zip(pairs, overlap)
            },
        }
        print(name, "radii", profiles[name]["radii_m"], "excluded", allowed, flush=True)
    return {
        "status": "calibrated_proxy",
        "geometry_policy": "strict_official_model_registry",
        "robots": profiles,
    }


def main() -> None:
    payload = calibrate()
    output = ROOT / "reports/single_arm/collision_proxy_profiles.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
