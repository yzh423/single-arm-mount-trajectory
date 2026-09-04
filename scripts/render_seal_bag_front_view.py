"""Render the restored seal-bag trajectory from the canonical front view."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from factory_bimanual.artifacts import FrameDiagnostics
from factory_bimanual.video import VideoRenderConfig, render_mujoco_mp4


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/factory_bimanual/seal_bag_dual_xarm6"
DEFAULT_SOURCE_STEM = "seal_bag_full_se3_follow_ik_v5_fixed_time_front_720p"


def report_directory_for_robot(robot_name):
    if robot_name not in ("xarm6", "i2rt_yam", "piperx"):
        raise ValueError(f"unsupported Seal_Bag robot: {robot_name}")
    return ROOT / "reports" / "factory_bimanual" / f"seal_bag_dual_{robot_name}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("xarm6", "i2rt_yam", "piperx"), default="xarm6")
    parser.add_argument("--source-stem", default=DEFAULT_SOURCE_STEM)
    parser.add_argument("--output-stem", default="seal_bag_safe_reference_front_view_720p")
    args = parser.parse_args(argv)
    if Path(args.source_stem).name != args.source_stem or Path(args.output_stem).name != args.output_stem:
        parser.error("stems must be safe filename components")
    source_stem = args.source_stem
    report = report_directory_for_robot(args.robot)
    output = report / f"{args.output_stem}.mp4"
    arrays = np.load(report / f"{source_stem}.trajectory.npz", allow_pickle=False)
    time_s = arrays["time_s"]
    success = arrays["synchronous_success"].astype(bool)
    try:
        reasons = arrays["failure_reason"].astype(str)
    except ValueError:
        reasons = np.where(success, "ok", "legacy_failure")
    collision = arrays["collision"].astype(bool)
    diagnostics = [FrameDiagnostics(
        index, float(timestamp), str(reasons[index]), str(reasons[index]),
        bool(collision[index]), False, "restored_v3_collision_aware",
    ) for index, timestamp in enumerate(time_s)]
    result = render_mujoco_mp4(
        report / f"{source_stem}.scene.xml", output, time_s,
        qpos=arrays["qpos"], diagnostics=diagnostics,
        left_targets=arrays["left_target_position"],
        right_targets=arrays["right_target_position"],
        follow_success=success,
        failure_reasons=reasons,
        config=VideoRenderConfig(
            width=1280, height=720, fps=60, trajectory_radius_m=.008,
            camera_azimuth_deg=180, camera_elevation_deg=-18,
            camera_distance_scale=1.55,
        ),
    )
    metadata = {
        "source_stem": source_stem,
        "robot": args.robot,
        "view": "front", "camera_azimuth_deg": 180,
        "camera_elevation_deg": -18, "camera_distance_scale": 1.55,
        "decode": result.check.__dict__,
    }
    output.with_suffix(".render.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
