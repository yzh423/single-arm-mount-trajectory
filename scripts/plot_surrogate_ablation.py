from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    random = rows(ROOT / "runs" / "surrogate_seed" / "doosan" / "history.jsonl")
    guided = rows(ROOT / "runs" / "surrogate_guided" / "doosan" / "history.jsonl")
    report_dir = ROOT / "reports" / "learning" / "hourly"
    heldout = {}
    for split in ("validation", "test"):
        for kind in ("random", "guided"):
            heldout[f"{kind}_{split}"] = json.loads(
                (report_dir / f"0900_{kind}_doosan_{split}.json").read_text())["metrics"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    generation = [row["generation"] + 1 for row in random]
    axes[0].plot(generation, [100*r["best_success_rate"] for r in random], "o-", label="random")
    axes[0].plot(generation, [100*r["best_success_rate"] for r in guided], "o-", label="surrogate UCB")
    axes[0].set_ylabel("training success (%)"); axes[0].set_xlabel("generation"); axes[0].legend()
    axes[1].plot(generation, [r["best_position_rmse_mm"] for r in random], "o-")
    axes[1].plot(generation, [r["best_position_rmse_mm"] for r in guided], "o-")
    axes[1].set_ylabel("training position RMSE (mm)"); axes[1].set_xlabel("generation")
    labels = ("val random", "val guided", "test random", "test guided")
    keys = ("random_validation", "guided_validation", "random_test", "guided_test")
    success = [100*heldout[key]["success_rate"] for key in keys]
    bars = axes[2].bar(labels, success, color=("#999999", "#2a9d8f", "#666666", "#1d6f65"))
    axes[2].set_ylabel("held-out success (%)"); axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_ylim(85, 100)
    for bar, value in zip(bars, success): axes[2].text(bar.get_x()+bar.get_width()/2, value+.2,
                                                       f"{value:.1f}", ha="center")
    for axis in axes: axis.grid(axis="y", alpha=.2)
    figure.suptitle("Doosan surrogate-guided NSGA-II ablation (equal evaluation budget)")
    output = report_dir / "0900_surrogate_ablation.png"; figure.savefig(output, dpi=220)
    summary = {"training": {"random": random, "guided": guided}, "heldout": heldout}
    (report_dir / "0900_surrogate_ablation.json").write_text(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__": main()

