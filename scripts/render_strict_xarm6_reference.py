"""Render the saved xArm6 reference solution through its matching URDF pipeline."""
from pathlib import Path
import sys

REFERENCE_ROOT = Path(r"E:\YZH123123\data\robot_reachability")
sys.path.insert(0, str(REFERENCE_ROOT))

import reachability_pipeline as pipeline


if __name__ == "__main__":
    pipeline.ROBOT_SPECS = {"xarm6": pipeline.ROBOT_SPECS["xarm6"]}
    pipeline.run(
        ["7-27__open-box-2"],
        ["xarm6"],
        frames_per_trial=0,
        skip_video=False,
        render_only=True,
        reuse_mount=False,
    )
