"""Build a ~5 minute, image-free DROID tabletop motion benchmark.

The source Cartesian state is [x, y, z, roll, pitch, yaw] in the Franka
base frame. Episodes retain those base-frame poses and are connected with
smooth SE(3) bridges. This prevents cumulative drift and makes the whole
sequence portable via one rigid transform into another robot base frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "DROID" / "candidates_5min"
OUTPUT_DIR = ROOT / "data" / "DROID" / "selected_5min"
FPS = 15.0
TRANSITION_SECONDS = 1.2

SELECTED = [
    "ILIAD+50aee79f+2023-08-17-21h-47m-30s",  # fold/flip/unfold towel
    "PennPAL+acda9df3+2023-07-14-16h-53m-06s",  # repeated pouring
    "RAIL+t3d58310+2023-07-28-17h-03m-26s",  # long table wiping
    "ILIAD+5e938e3b+2023-07-20-10h-54m-28s",  # microwave composite
    "PennPAL+acda9df3+2023-07-19-17h-10m-04s",  # stirring
    "REAL+4f8ca688+2023-06-08-15h-00m-30s",  # drawer + two objects
]


def _resample(time_s: np.ndarray, pose_rpy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_s = time_s - time_s[0]
    duration = float(time_s[-1])
    dst_t = np.arange(0.0, duration, 1.0 / FPS)
    if duration - dst_t[-1] > 0.25 / FPS:
        dst_t = np.append(dst_t, duration)
    pos = np.column_stack(
        [np.interp(dst_t, time_s, pose_rpy[:, axis]) for axis in range(3)]
    )
    rotations = Rotation.from_euler("xyz", pose_rpy[:, 3:6])
    quat_xyzw = Slerp(time_s, rotations)(dst_t).as_quat()
    return dst_t, np.column_stack((pos, quat_xyzw[:, [3, 0, 1, 2]]))


def _smooth_edges(pose: np.ndarray, seconds: float = 0.8) -> np.ndarray:
    """Ease displacement at episode edges without changing its geometric path."""
    result = pose.copy()
    count = min(int(round(seconds * FPS)), max(1, len(result) // 4))
    if count < 2:
        return result
    # Re-time the first and last short windows with cubic smoothstep.
    u = np.linspace(0.0, 1.0, count)
    smooth = u * u * (3.0 - 2.0 * u)
    for start, stop, weights in ((0, count, smooth), (len(result) - count, len(result), smooth)):
        p0, p1 = result[start, :3], result[stop - 1, :3]
        result[start:stop, :3] = p0 + weights[:, None] * (p1 - p0)
        rots = Rotation.from_quat(result[start:stop, 3:7][:, [1, 2, 3, 0]])
        key = Rotation.from_quat(np.vstack((rots.as_quat()[0], rots.as_quat()[-1])))
        q = Slerp([0.0, 1.0], key)(weights).as_quat()
        result[start:stop, 3:7] = q[:, [3, 0, 1, 2]]
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pose: list[np.ndarray] = []
    all_gripper: list[np.ndarray] = []
    segments: list[dict] = []
    frame_cursor = 0

    for segment_index, uuid in enumerate(SELECTED):
        source_dir = SOURCE_ROOT / uuid
        annotation = json.loads((source_dir / "annotation.json").read_text(encoding="utf-8"))
        with h5py.File(source_dir / "trajectory.h5", "r") as h5:
            raw_time = np.asarray(h5["observation/timestamp/control/step_start"], dtype=float)
            raw_time = (raw_time - raw_time[0]) / 1000.0
            raw_pose = np.asarray(h5["observation/robot_state/cartesian_position"], dtype=float)
            raw_gripper = np.asarray(h5["observation/robot_state/gripper_position"], dtype=float)

        sample_t, pose = _resample(raw_time, raw_pose)
        gripper = np.interp(sample_t, raw_time, raw_gripper)
        pose = _smooth_edges(pose)

        if all_pose:
            hold_count = int(round(TRANSITION_SECONDS * FPS))
            u = np.linspace(0.0, 1.0, hold_count + 2)[1:-1]
            smooth = u * u * (3.0 - 2.0 * u)
            previous = all_pose[-1][-1]
            bridge_pos = previous[:3] + smooth[:, None] * (pose[0, :3] - previous[:3])
            bridge_key = Rotation.from_quat(
                np.vstack((previous[3:7][[1, 2, 3, 0]], pose[0, 3:7][[1, 2, 3, 0]]))
            )
            bridge_quat = Slerp([0.0, 1.0], bridge_key)(smooth).as_quat()[:, [3, 0, 1, 2]]
            all_pose.append(np.column_stack((bridge_pos, bridge_quat)))
            all_gripper.append(
                all_gripper[-1][-1] + smooth * (gripper[0] - all_gripper[-1][-1])
            )
            frame_cursor += hold_count
        all_pose.append(pose)
        all_gripper.append(gripper)
        segments.append(
            {
                "uuid": uuid,
                "instruction": annotation["language_instruction1"],
                "start_frame": frame_cursor,
                "end_frame": frame_cursor + len(pose) - 1,
                "duration_s": round((len(pose) - 1) / FPS, 3),
            }
        )
        frame_cursor += len(pose)

    pose = np.concatenate(all_pose)
    gripper = np.concatenate(all_gripper)
    time_s = np.arange(len(pose), dtype=float) / FPS
    np.savez_compressed(
        OUTPUT_DIR / "droid_vr_tabletop_5min.npz",
        time_s=time_s,
        tcp_xyz=pose[:, :3],
        tcp_quat_wxyz=pose[:, 3:7],
        gripper=gripper,
    )
    manifest = {
        "format": "DROID base-frame portable SE(3) trajectory",
        "fps": FPS,
        "duration_s": round(float(time_s[-1]), 3),
        "samples": len(time_s),
        "pose_fields": ["tcp_xyz", "tcp_quat_wxyz"],
        "translation_semantics": "absolute DROID robot-base frame",
        "rotation_semantics": "absolute DROID robot-base frame",
        "transition_seconds": TRANSITION_SECONDS,
        "segments": segments,
    }
    (OUTPUT_DIR / "droid_vr_tabletop_5min_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(time_s)} samples, {time_s[-1]:.2f} s")
    print(OUTPUT_DIR / "droid_vr_tabletop_5min.npz")


if __name__ == "__main__":
    main()
