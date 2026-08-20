"""Lossless loader for complete factory handheld bimanual CSV recordings."""
from __future__ import annotations

from dataclasses import dataclass
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
                      repair_invalid_pose_rows: bool = False) -> FactoryBimanualTask:
    path = Path(path).resolve()
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("factory CSV is empty")
    time_s = frame["t"].to_numpy(float)
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
        source_row_index=np.arange(len(frame), dtype=int), time_s=time_s,
        coordinate_frame="vr_world", left_position_m=left_position,
        left_quaternion_wxyz=left_quaternion,
        right_position_m=right_position,
        right_quaternion_wxyz=right_quaternion,
        left_valid=left_valid,
        right_valid=right_valid,
        left_gripper_angle_rad=optional_gripper("left"),
        right_gripper_angle_rad=optional_gripper("right"),
    )
