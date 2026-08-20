from pathlib import Path
import argparse
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from design_optimization.kinematics import build_designs, deterministic_joint_samples, fk_flange, fk_tcp
from design_optimization.topology import load_templates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    templates = load_templates(ROOT / "reports/parametric_topology_audit.json", device=device)
    for name, template in templates.items():
        calibration = deterministic_joint_samples(template, args.samples)
        raw = torch.zeros((args.designs, 6), device=device)
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        start = time.perf_counter()
        designs = build_designs(template, raw, calibration)
        flange = fk_flange(designs, calibration)
        tcp = fk_tcp(designs, calibration[:, :])
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        reach = torch.linalg.vector_norm(flange[..., :3, 3], dim=-1).amax(dim=1)
        print(f"{name:7s} designs={args.designs} poses={args.samples} "
              f"flange_reach=[{reach.min():.6f},{reach.max():.6f}]m "
              f"tcp_max={torch.linalg.vector_norm(tcp[..., :3,3],dim=-1).amax():.6f}m "
              f"time={elapsed:.3f}s peak={torch.cuda.max_memory_allocated()/2**20:.1f}MiB")


if __name__ == "__main__":
    main()
