"""Derive mount-orientation comparison metrics from immutable row arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _summary(values, scale=1.0):
    values = np.asarray(values, dtype=float) * scale
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def _longest_failure(followed, time_s):
    followed = np.asarray(followed, dtype=bool); time_s = np.asarray(time_s, float)
    dt = np.diff(time_s)
    row_duration = np.r_[dt, np.median(dt) if len(dt) else 0.]
    best_frames = current_frames = 0; best_s = current_s = 0.
    for ok, duration in zip(followed, row_duration):
        if ok:
            current_frames = 0; current_s = 0.
        else:
            current_frames += 1; current_s += float(duration)
            if current_frames > best_frames:
                best_frames, best_s = current_frames, current_s
    return int(best_frames), float(best_s)


def derive_orientation_metrics(npz_path: Path):
    with np.load(npz_path, allow_pickle=False) as values:
        required = {
            "time_s", "followed", "q", "joint_lower", "joint_upper",
            "position_error_left_m", "position_error_right_m",
            "orientation_error_left_rad", "orientation_error_right_rad",
            "singularity_left", "singularity_right", "state_collision",
            "edge_collision",
        }
        missing = required - set(values.files)
        if missing:
            raise ValueError(f"missing metric arrays: {sorted(missing)}")
        time_s = values["time_s"]; followed = values["followed"]
        q = values["q"]
        if not (len(time_s) == len(followed) == len(q)):
            raise ValueError("metric arrays are not source-row aligned")
        failure_frames, failure_s = _longest_failure(followed, time_s)
        margin = np.minimum(q - values["joint_lower"],
                            values["joint_upper"] - q)
        dt = np.diff(time_s)
        velocity = np.diff(q, axis=0) / dt[:, None]
        position = np.concatenate((values["position_error_left_m"],
                                   values["position_error_right_m"]))
        orientation = np.concatenate((values["orientation_error_left_rad"],
                                      values["orientation_error_right_rad"]))
        singularity = np.concatenate((values["singularity_left"],
                                      values["singularity_right"]))
        return {
            "source_rows": int(len(time_s)),
            "coverage": float(np.mean(followed)),
            "longest_failure_frames": failure_frames,
            "longest_failure_s": failure_s,
            "position_error_mm": _summary(position, 1000.),
            "orientation_error_deg": _summary(orientation, 180. / np.pi),
            "minimum_joint_margin_rad": float(np.min(margin)),
            "p10_joint_margin_rad": float(np.percentile(margin, 10)),
            "minimum_singularity_margin": float(np.min(singularity)),
            "joint_velocity_rad_s": _summary(np.abs(velocity)),
            "joint_total_variation_rad": float(np.sum(np.abs(np.diff(q, axis=0)))),
            "state_collision_frames": int(np.count_nonzero(values["state_collision"])),
            "edge_collision_frames": int(np.count_nonzero(values["edge_collision"])),
        }

