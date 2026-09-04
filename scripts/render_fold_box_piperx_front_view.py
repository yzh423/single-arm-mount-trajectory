"""Re-render the solved official PiPER-X Fold Box path from true front."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from factory_bimanual.artifacts import FrameDiagnostics
from factory_bimanual.video import VideoRenderConfig, render_mujoco_mp4


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/factory_bimanual/fold_box_dual_piperx"
SOURCE_STEM = "fold_box_full_se3_acceleration_smoothed_front_720p"
OUTPUT = REPORT / "fold_box_official_piperx_true_front_720p.mp4"
FRONT_ELEVATION_DEG = -12.0


def front_azimuth_from_mount(left_xy, right_xy):
    """Camera direction normal to the two-base baseline, in MuJoCo degrees."""
    baseline = np.asarray(right_xy, float) - np.asarray(left_xy, float)
    if baseline.shape != (2,) or np.linalg.norm(baseline) < 1e-9:
        raise ValueError("mount points must define a nonzero baseline")
    return float((np.degrees(np.arctan2(baseline[1], baseline[0])) - 90.0) % 360.0)


def main():
    summary = json.loads((REPORT / f"{SOURCE_STEM}.summary.json").read_text(
        encoding="utf-8"))
    mount = summary["mount"]["xy"]
    front_azimuth_deg = front_azimuth_from_mount(mount["left"], mount["right"])
    arrays = np.load(REPORT / f"{SOURCE_STEM}.trajectory.npz", allow_pickle=False)
    time_s = arrays["time_s"]
    success = arrays["synchronous_success"].astype(bool)
    try:
        reasons = arrays["failure_reason"].astype(str)
    except ValueError:
        reasons = np.where(success, "ok", "legacy_failure")
    collision = arrays["collision"].astype(bool)
    diagnostics = [FrameDiagnostics(
        index, float(timestamp), str(reasons[index]), str(reasons[index]),
        bool(collision[index]), False, "official_piperx_fold_box",
    ) for index, timestamp in enumerate(time_s)]
    result = render_mujoco_mp4(
        REPORT / f"{SOURCE_STEM}.scene.xml", OUTPUT, time_s,
        qpos=arrays["qpos"], diagnostics=diagnostics,
        left_targets=arrays["left_target_position"],
        right_targets=arrays["right_target_position"],
        follow_success=success,
        failure_reasons=reasons,
        config=VideoRenderConfig(
            width=1280, height=720, fps=60, trajectory_radius_m=.008,
            interpolate_states=True,
            camera_azimuth_deg=front_azimuth_deg,
            camera_elevation_deg=FRONT_ELEVATION_DEG,
            camera_distance_scale=1.48,
            title="Dual official PiPER-X trajectory follow | front view",
        ),
    )
    metadata = {
        "source_stem": SOURCE_STEM,
        "view": "true_front",
        "camera_azimuth_deg": front_azimuth_deg,
        "camera_elevation_deg": FRONT_ELEVATION_DEG,
        "camera_distance_scale": 1.48,
        "decode": result.check.__dict__,
    }
    OUTPUT.with_suffix(".render.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
