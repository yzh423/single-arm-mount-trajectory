from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROBOTS = ("doosan", "xarm6", "ur5", "kinova")


def load(directory: Path) -> dict[str, dict]:
    return {robot: json.loads((directory / f"{robot}_test_episode_563.json").read_text())
            for robot in ROBOTS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrinsic", type=Path,
                        default=Path("reports/learning/continuous_retimed"))
    parser.add_argument("--proxy", type=Path,
                        default=Path("reports/learning/continuous_proxy_penalized"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/learning/hourly/1200_continuous_proxy_ablation.png"))
    args = parser.parse_args(); intrinsic, proxy = load(args.intrinsic), load(args.proxy)
    x = np.arange(len(ROBOTS)); width = .36
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    fields = (("success_rate", "Success (%)", 100),
              ("position_rmse_mm", "Position RMSE (mm)", 1),
              ("retime_slowdown", "Retiming slowdown", 1))
    for axis, (field, title, scale) in zip(axes, fields):
        a = [scale * intrinsic[r]["metrics"][field] for r in ROBOTS]
        b = [scale * proxy[r]["metrics"][field] for r in ROBOTS]
        bars_a = axis.bar(x - width / 2, a, width, label="intrinsic continuous IK")
        bars_b = axis.bar(x + width / 2, b, width, label="generic capsule weighted", alpha=.75)
        axis.set_xticks(x, ROBOTS); axis.set_title(title); axis.grid(axis="y", alpha=.25)
        axis.bar_label(bars_a, fmt="%.2f", fontsize=8, rotation=90, padding=2)
        axis.bar_label(bars_b, fmt="%.2f", fontsize=8, rotation=90, padding=2)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Continuous EgoDex ablation: topology-biased generic collision proxy", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.output, dpi=220)
    print(args.output)


if __name__ == "__main__":
    main()
