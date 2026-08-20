from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "reports" / "learning" / "collision_calibration"
    names = ("doosan", "xarm6", "ur5", "kinova")
    capsule = [json.loads((source/f"{n}_capsule_calibration.json").read_text())["test_metrics"]
               for n in names]
    neural = [json.loads((source/f"{n}_collision_classifier.json").read_text())["test_metrics"]
              for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True); x=np.arange(4); w=.36
    axes[0].bar(x-w/2, [r["recall"] for r in capsule], w, label="single centerline")
    axes[0].bar(x+w/2, [r["recall"] for r in neural], w, label="neural C-space")
    axes[0].set_title("held-out collision recall"); axes[0].set_xticks(x,names); axes[0].set_ylim(0,1.05)
    axes[1].bar(x-w/2, [r["false_positive_rate"] for r in capsule], w)
    axes[1].bar(x+w/2, [r["false_positive_rate"] for r in neural], w)
    axes[1].set_title("held-out false-positive rate"); axes[1].set_xticks(x,names); axes[1].set_ylim(0,1.05)
    for axis in axes: axis.grid(axis="y",alpha=.2)
    axes[0].legend(); fig.suptitle("Vendor MuJoCo mesh labels: fixed-morphology collision proxies")
    out=ROOT/'reports'/'learning'/'hourly'/'1000_collision_classifier_comparison.png'
    fig.savefig(out,dpi=220);print(out)


if __name__=='__main__':main()
