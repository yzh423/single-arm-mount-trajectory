"""Generate compact figures for the final configuration/MPC report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "tuning"


def main() -> int:
    local = np.load(TUNING / "doosan_atomic_mpc_set2_z055_full.npz")
    planned = np.load(TUNING / "doosan_planned_mpc_final_set2_z055_full.npz")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = ("#177ddc", "#d46b08")
    for arm_index, (side, color) in enumerate(zip(("left", "right"), colors)):
        axes[0].plot(
            local["time"],
            1000.0 * local["position_error"][:, arm_index],
            color=color,
            alpha=0.72,
            label=f"local MPC {side}",
        )
        axes[0].plot(
            planned["time"],
            1000.0 * planned["position_error"][:, arm_index],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"planned MPC {side}",
        )
    axes[0].set_ylabel("TCP position error (mm)")
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=2)
    axes[0].set_title("M0609: local safety stop versus global-plan-guided MPC")

    axes[1].plot(
        local["time"],
        100.0 * local["safety_rollback"][:, 0],
        color="#cf1322",
        label="local MPC atomic rollback",
    )
    axes[1].plot(
        planned["time"],
        100.0 * planned["safety_rollback"][:, 0],
        color="#389e0d",
        label="planned MPC atomic rollback",
    )
    axes[1].set_ylabel("Rollback state (%)")
    axes[1].set_xlabel("Dataset time (s)")
    axes[1].set_ylim(-3, 103)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    output = TUNING / "final_mpc_tracking_comparison.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)

    labels = ["Desktop", "45° old", "45° opt.", "90° old", "90° opt."]
    xarm = [0.55, 0.54, np.nan, 0.56, np.nan]
    ur5 = [0.33, 119.13, 0.375, 0.36, np.nan]
    doosan = [0.33, 0.33, np.nan, 12.61, 0.338]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width, xarm, width, label="xArm6 left")
    ax.bar(x, ur5, width, label="UR5 left")
    ax.bar(x + width, doosan, width, label="M0609 left")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Left TCP RMSE (mm, symlog)")
    ax.set_xticks(x, labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.set_title(
        "Mounting is a first-order variable; optimized bars are 30 s prefixes"
    )
    fig.tight_layout()
    mount_output = TUNING / "final_mount_comparison.png"
    fig.savefig(mount_output, dpi=170)
    plt.close(fig)

    manifest = {
        "tracking_figure": str(output),
        "mount_figure": str(mount_output),
        "tracking_sources": [
            "doosan_atomic_mpc_set2_z055_full.npz",
            "doosan_planned_mpc_final_set2_z055_full.npz",
        ],
    }
    (TUNING / "final_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output)
    print(mount_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
