"""Render the Fold Box demonstration with the refined dual-PiperX mount."""
from pathlib import Path

from scripts.render_factory_dual_xarm6_fold_box import run_fold_box


ROOT = Path(__file__).resolve().parents[1]
OUT = (ROOT / "reports/factory_bimanual/fold_box_dual_piperx/"
       "fold_box_full_se3_acceleration_smoothed_front_720p.mp4")

SELECTED_MOUNT = {
    "xy": {
        "left": [-0.2312103678324848, -0.0130977167988798],
        "right": [0.0412103678324847, -0.3569022832011202],
    },
    "yaw": {"left": 15.0, "right": 45.0},
    "shared_base_z_m": .81,
    "selection_method": (
        "official PiPER-X URDF and model ee_frame TCP; Seal Bag paired strict "
        "collision-free synchronous 6D IK with hard state and swept-edge "
        "collision rejection; 60-frame final mount verification"),
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    return run_fold_box("piperx", SELECTED_MOUNT, OUT, 3.0)


if __name__ == "__main__":
    main()
