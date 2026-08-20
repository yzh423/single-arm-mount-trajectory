from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "reports" / "learning" / "collision_calibration"
    output = ROOT / "reports" / "learning" / "hourly" / "0900_collision_proxy_audit.png"
    names = ("doosan", "xarm6", "ur5", "kinova")
    rows = [json.loads((source / f"{name}_capsule_calibration.json").read_text()) for name in names]
    precision = [row["test_metrics"]["precision"] for row in rows]
    recall = [row["test_metrics"]["recall"] for row in rows]
    fpr = [row["test_metrics"]["false_positive_rate"] for row in rows]
    margin = [1000 * row["validation_selected_safety_margin_m"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(names)); width = .25
    axes[0].bar(x - width, precision, width, label="precision")
    axes[0].bar(x, recall, width, label="recall")
    axes[0].bar(x + width, fpr, width, label="false-positive rate")
    axes[0].axhline(.95, color="black", ls="--", lw=1, label="target recall")
    axes[0].set_xticks(x, names); axes[0].set_ylim(0, 1.05); axes[0].grid(axis="y", alpha=.2)
    axes[0].set_title("Held-out MuJoCo mesh collision labels"); axes[0].legend()
    bars = axes[1].bar(names, margin, color=("#277da1", "#f9844a", "#f94144", "#90be6d"))
    axes[1].set_ylabel("validation-selected safety margin (mm)")
    axes[1].set_title("Margin required by one-centerline capsule proxy")
    axes[1].grid(axis="y", alpha=.2)
    for bar, value in zip(bars, margin):
        axes[1].text(bar.get_x() + bar.get_width()/2, value + 1, f"{value:.1f}", ha="center")
    figure.suptitle("Collision proxy fidelity audit: single capsules are screening-only")
    output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=220); print(output)


if __name__ == "__main__": main()

