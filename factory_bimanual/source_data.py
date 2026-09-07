"""Lossless loader for complete factory handheld bimanual CSV recordings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactoryBimanualTask:
    name: str
    source_path: Path
    source_row_index: np.ndarray
    time_s: np.ndarray
    coordinate_frame: str
    left_position_m: np.ndarray
    left_quaternion_wxyz: np.ndarray
    right_position_m: np.ndarray
    right_quaternion_wxyz: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    left_gripper_angle_rad: np.ndarray | None
    right_gripper_angle_rad: np.ndarray | None
    timing_source: str = field(default="host_poll", kw_only=True)
    source_poll_row_count: int | None = field(default=None, kw_only=True)


def _columns(frame: pd.DataFrame, side: str, kind: str) -> np.ndarray:
    suffixes = ("x", "y", "z") if kind == "pos" else ("w", "x", "y", "z")
    return frame[[f"{side}_tcp_{kind}_{suffix}" for suffix in suffixes]].to_numpy(float)


def _normalized(quaternion: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("TCP quaternion contains a zero norm")
    return quaternion / norms


def load_factory_task(path: Path, task_name: str, *,
                      max_translation_jump_m: float = 0.05,
                      repair_invalid_pose_rows: bool = False,
                      timing_mode: str = "host_poll",
                      max_receive_skew_s: float = 1e-3) -> FactoryBimanualTask:
    path = Path(path).resolve()
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("factory CSV is empty")
    source_poll_row_count = len(frame)
    source_row_index = np.arange(source_poll_row_count, dtype=int)
    timing_source = "host_poll"
    if timing_mode == "controller_updates":
        required = [
            "left_frame_counter", "right_frame_counter",
            "left_receive_monotonic_s", "right_receive_monotonic_s",
        ]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(
                "controller-update timing requires columns: " + ", ".join(missing))
        left_counter = frame["left_frame_counter"].to_numpy()
        right_counter = frame["right_frame_counter"].to_numpy()
        left_changed = np.r_[True, left_counter[1:] != left_counter[:-1]]
        right_changed = np.r_[True, right_counter[1:] != right_counter[:-1]]
        if not np.array_equal(left_changed, right_changed):
            raise ValueError("asynchronous left/right controller frames")
        receive = frame[[
            "left_receive_monotonic_s", "right_receive_monotonic_s",
        ]].to_numpy(float)
        if not np.all(np.isfinite(receive)):
            raise ValueError("controller receive time must be finite")
        if not np.isfinite(max_receive_skew_s) or max_receive_skew_s < 0:
            raise ValueError("max_receive_skew_s must be finite and nonnegative")
        receive_skew = np.abs(receive[:, 0] - receive[:, 1])
        if np.any(receive_skew[left_changed] > max_receive_skew_s):
            raise ValueError("left/right controller receive-time skew exceeds limit")
        source_row_index = source_row_index[left_changed]
        event_time = receive[left_changed].mean(axis=1)
        event_time -= event_time[0]
        frame = frame.loc[left_changed].reset_index(drop=True)
        time_s = event_time
        timing_source = "paired_controller_receive"
    elif timing_mode == "host_poll":
        time_s = frame["t"].to_numpy(float)
    else:
        raise ValueError(f"unknown timing_mode: {timing_mode}")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        raise ValueError("source time must be finite and strictly increasing")
    coordinate_frames = tuple(frame["coordinate_frame"].drop_duplicates())
    if coordinate_frames != ("vr_world",):
        raise ValueError(f"expected one vr_world frame, got {coordinate_frames}")
    left_valid = frame["left_tcp_valid"].to_numpy(bool)
    right_valid = frame["right_tcp_valid"].to_numpy(bool)

    def repaired(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).copy()
        nonfinite_rows = ~np.isfinite(values).all(axis=1)
        if not np.any(nonfinite_rows):
            return values
        if not repair_invalid_pose_rows or np.any(nonfinite_rows & valid):
            raise ValueError("TCP pose contains non-finite values")
        rows = np.arange(len(values))
        for column in range(values.shape[1]):
            finite = np.isfinite(values[:, column])
            if not np.any(finite):
                raise ValueError("TCP pose channel has no finite values")
            values[~finite, column] = np.interp(
                rows[~finite], rows[finite], values[finite, column])
        return values

    left_position = repaired(_columns(frame, "left", "pos"), left_valid)
    right_position = repaired(_columns(frame, "right", "pos"), right_valid)
    left_quaternion = _normalized(repaired(
        _columns(frame, "left", "quat"), left_valid))
    right_quaternion = _normalized(repaired(
        _columns(frame, "right", "quat"), right_valid))
    arrays = (left_position, right_position, left_quaternion, right_quaternion)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("TCP pose contains non-finite values")
    maximum_jump = max(
        float(np.linalg.norm(np.diff(left_position, axis=0), axis=1).max()),
        float(np.linalg.norm(np.diff(right_position, axis=0), axis=1).max()),
    )
    if not np.isfinite(max_translation_jump_m) or max_translation_jump_m <= 0:
        raise ValueError("max_translation_jump_m must be finite and positive")
    if maximum_jump > max_translation_jump_m:
        raise ValueError(
            f"TCP translation jump exceeds {100*max_translation_jump_m:g} cm: "
            f"{maximum_jump:.6f} m")
    def optional_gripper(side: str) -> np.ndarray | None:
        column = f"{side}_gripper_angle_rad"
        return frame[column].to_numpy(float) if column in frame else None
    return FactoryBimanualTask(
        name=str(task_name), source_path=path,
        source_row_index=source_row_index, time_s=time_s,
        coordinate_frame="vr_world", left_position_m=left_position,
        left_quaternion_wxyz=left_quaternion,
        right_position_m=right_position,
        right_quaternion_wxyz=right_quaternion,
        left_valid=left_valid,
        right_valid=right_valid,
        left_gripper_angle_rad=optional_gripper("left"),
        right_gripper_angle_rad=optional_gripper("right"),
        timing_source=timing_source,
        source_poll_row_count=source_poll_row_count,
    )
