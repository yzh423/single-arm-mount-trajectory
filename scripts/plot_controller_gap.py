from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROBOTS = ("doosan", "xarm6", "ur5", "kinova")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise", type=Path,
                        default=Path("reports/learning/episode_pointwise_upper_bound"))
    parser.add_argument("--continuous", type=Path,
                        default=Path("reports/learning/continuous_retimed"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/learning/hourly/1200_controller_gap.png"))
    args = parser.parse_args(); pointwise, continuous = [], []
    for robot in ROBOTS:
        p = json.loads((args.pointwise / f"{robot}_episode_563.json").read_text())
        c = json.loads((args.continuous / f"{robot}_test_episode_563.json").read_text())
        pointwise.append(100 * p["metrics"]["success_rate"])
        continuous.append(100 * c["metrics"]["success_rate"])
    x = np.arange(len(ROBOTS)); width = .36
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    a = axis.bar(x - width / 2, pointwise, width, label="independent-frame oracle")
    b = axis.bar(x + width / 2, continuous, width, label="finite continuous solver")
    axis.set_xticks(x, ROBOTS); axis.set_ylim(0, 105); axis.set_ylabel("success (%)")
    axis.set_title("Same EgoDex episode: morphology upper bound vs controller realization")
    axis.grid(axis="y", alpha=.25); axis.legend(); axis.bar_label(a, fmt="%.1f")
    axis.bar_label(b, fmt="%.1f")
    args.output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.output, dpi=220)
    print(args.output)


if __name__ == "__main__":
    main()
