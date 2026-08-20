from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/learning/mount_roll_profile/comparison.png")
    args = parser.parse_args()
    inputs = args.inputs or sorted((ROOT / "reports/learning/mount_roll_profile").glob("*.json"))
    if not inputs:
        raise SystemExit("no mount-roll profile JSON files found")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fields = (("success_rate", "Success (%)", 100.),
              ("position_rmse_mm", "Position RMSE (mm)", 1.),
              ("orientation_rmse_deg", "Orientation RMSE (deg)", 1.),
              ("minimum_clearance_mm", "Minimum clearance (mm)", 1.))
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8")); rows = payload["rows"]
        roll = np.asarray([row["roll_deg"] for row in rows])
        for axis, (field, label, scale) in zip(axes.flat, fields):
            axis.plot(roll, scale * np.asarray([row[field] for row in rows]), marker="o",
                      ms=3, label=payload["robot"])
            axis.set_ylabel(label); axis.grid(alpha=.25)
    for axis in axes[-1]: axis.set_xlabel("Mirrored shoulder roll (deg)")
    axes[0, 0].legend(ncol=2)
    figure.suptitle("Held-out EgoDex: XYZ/spacing optimized independently at each mount roll")
    figure.tight_layout(); args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220); print(args.output)


if __name__ == "__main__":
    main()
