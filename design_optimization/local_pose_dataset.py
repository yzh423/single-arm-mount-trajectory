"""Canonical single-arm pose episodes derived from local teleoperation CSVs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from typing import Literal, Mapping

import numpy as np
import pandas as pd


Hand = Literal["left", "right"]
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class CleanedPoseFrame:
    time_s: np.ndarray
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    source_rows: np.ndarray


@dataclass(frozen=True)
class CleaningRecord:
    disposition: str
    source_rows: int
    retained_rows: int
    reason_counts: Mapping[str, int]


def infer_hands(batch: str, task: str) -> tuple[Hand, ...]:
    """Return the target hands encoded by the approved batch conventions."""
    if batch in {"7-27", "7-28"}:
        return ("right",)
    if batch == "8-05":
        return ("left",) if "left" in task.lower() else ("left", "right")
    if batch == "8-06":
        return ("left", "right")
    raise ValueError(f"unsupported source batch: {batch}")


def content_split(content_sha256: str) -> Split:
    """Map one content group deterministically to the existing 70/15/15 split."""
    if len(content_sha256) != 64:
        raise ValueError("content_sha256 must contain 64 hexadecimal characters")
    try:
        bucket = int(content_sha256[:16], 16) / float(16**16)
    except ValueError as exc:
        raise ValueError("content_sha256 must be hexadecimal") from exc
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "validation"
    return "test"


def assign_stratified_splits(episodes: list[dict]) -> list[dict]:
    """Assign task-stratified splits, grouping both hands of one source episode."""
    def group_key(episode: dict) -> str:
        return str(episode.get("source_sha256", episode["content_sha256"]))
    by_task: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        if episode.get("trajectory_edge_trim_eligible", True):
            by_task[str(episode["task"])].add(group_key(episode))
    assignment: dict[str, Split] = {}
    for task in sorted(by_task):
        hashes = sorted(by_task[task])
        count = len(hashes)
        if count < 3:
            for digest in hashes:
                assignment.setdefault(digest, content_split(digest))
            continue
        validation_count = max(1, int(np.floor(0.15 * count)))
        test_count = max(1, int(np.floor(0.15 * count)))
        train_count = count - validation_count - test_count
        if train_count < 1:
            train_count = 1
            test_count = count - train_count - validation_count
        for index, digest in enumerate(hashes):
            split: Split = (
                "train" if index < train_count
                else "validation" if index < train_count + validation_count
                else "test"
            )
            assignment.setdefault(digest, split)
    return [dict(row, split=(assignment[group_key(row)] if row.get("trajectory_edge_trim_eligible", True) else "train")) for row in episodes]


def _bounded_internal_runs(invalid: np.ndarray, maximum: int) -> tuple[np.ndarray, ...]:
    indices = np.flatnonzero(invalid)
    if not len(indices):
        return ()
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return tuple(
        run
        for run in runs
        if run[0] > 0 and run[-1] < len(invalid) - 1 and len(run) <= maximum
    )


def _interpolate_rows(values: np.ndarray, invalid: np.ndarray, maximum: int) -> np.ndarray:
    result = values.copy()
    for run in _bounded_internal_runs(invalid, maximum):
        left, right = int(run[0] - 1), int(run[-1] + 1)
        if invalid[left] or invalid[right]:
            continue
        alpha = (run - left) / float(right - left)
        result[run] = ((1.0 - alpha[:, None]) * result[left]
                       + alpha[:, None] * result[right])
        invalid[run] = False
    return result


def clean_pose_frame(
    frame: pd.DataFrame,
    *,
    hand: Hand,
    maximum_internal_repair_samples: int = 3,
    exclude_invalid_fraction: float = 0.20,
) -> tuple[CleanedPoseFrame | None, CleaningRecord]:
    """Clean one hand's TCP pose while retaining source-row provenance."""
    source_count = len(frame)
    time = frame["t"].to_numpy(dtype=float)
    if source_count < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        reversals = int(np.sum(np.diff(time) <= 0.0)) if source_count >= 2 else 0
        return None, CleaningRecord(
            "excluded_structural", source_count, 0,
            {"time_reversal": reversals},
        )

    prefix = f"{hand}_tcp_"
    valid = frame[f"{prefix}valid"].to_numpy(dtype=bool)
    position = frame[[f"{prefix}pos_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
    quaternion = frame[[f"{prefix}quat_{axis}" for axis in "wxyz"]].to_numpy(dtype=float)
    norms = np.linalg.norm(np.where(np.isfinite(quaternion), quaternion, 0.0), axis=1)
    invalid = (~valid | ~np.isfinite(position).all(axis=1)
               | ~np.isfinite(quaternion).all(axis=1) | (norms <= 1e-9))
    initial_invalid = invalid.copy()
    position_invalid = invalid.copy()
    quaternion_invalid = invalid.copy()
    position = _interpolate_rows(
        position, position_invalid, maximum_internal_repair_samples
    )
    quaternion = _interpolate_rows(
        quaternion, quaternion_invalid, maximum_internal_repair_samples
    )
    invalid = position_invalid | quaternion_invalid
    retained = ~invalid
    invalid_fraction = float(np.mean(~retained))
    if retained.sum() < 2 or invalid_fraction > exclude_invalid_fraction:
        return None, CleaningRecord(
            "excluded_unrepairable_numeric", source_count, int(retained.sum()),
            {"invalid_rows": int(initial_invalid.sum()),
             "unrepaired_rows": int((~retained).sum())},
        )

    time = time[retained]
    position = position[retained]
    quaternion = quaternion[retained]
    source_rows = np.flatnonzero(retained)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    for index in range(1, len(quaternion)):
        if float(quaternion[index - 1] @ quaternion[index]) < 0.0:
            quaternion[index] *= -1.0
    return CleanedPoseFrame(time, position, quaternion, source_rows), CleaningRecord(
        "eligible", source_count, len(time),
        {"repaired_rows": int(np.sum(initial_invalid & retained)),
         "unrepaired_rows": int((~retained).sum())},
    )


def _array_content_sha256(cleaned: CleanedPoseFrame) -> str:
    digest = hashlib.sha256()
    for values in (cleaned.time_s, cleaned.position_m, cleaned.quaternion_wxyz):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_local_pose_benchmark(
    source_root: str | Path,
    output_root: str | Path,
    *,
    maximum_internal_repair_samples: int = 3,
    exclude_invalid_fraction: float = 0.20,
) -> dict:
    """Clean every supported CSV and publish one deterministic manifest."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    episode_root = output_root / "episodes"
    episode_root.mkdir(parents=True, exist_ok=True)
    source_records: list[dict] = []
    episodes: list[dict] = []
    sources = sorted(
        path for batch in ("7-27", "7-28", "8-05", "8-06")
        for path in (source_root / batch).rglob("*.csv")
    )
    for source_path in sources:
        relative = source_path.relative_to(source_root)
        batch, task = relative.parts[0], relative.parts[1]
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        frame = pd.read_csv(source_path)
        hand_records: list[dict] = []
        for hand in infer_hands(batch, task):
            cleaned, record = clean_pose_frame(
                frame,
                hand=hand,
                maximum_internal_repair_samples=maximum_internal_repair_samples,
                exclude_invalid_fraction=exclude_invalid_fraction,
            )
            hand_record = {
                "hand": hand,
                "disposition": record.disposition,
                "source_rows": record.source_rows,
                "retained_rows": record.retained_rows,
                "reason_counts": dict(record.reason_counts),
            }
            hand_records.append(hand_record)
            if cleaned is None:
                continue
            content_sha256 = _array_content_sha256(cleaned)
            episode_id = (
                f"{batch}__{task}__{source_path.stem}__{hand}__"
                f"{content_sha256[:12]}"
            )
            artifact = Path("episodes") / f"{episode_id}.npz"
            np.savez_compressed(
                output_root / artifact,
                time_s=cleaned.time_s,
                position_m=cleaned.position_m,
                quaternion_wxyz=cleaned.quaternion_wxyz,
                source_rows=cleaned.source_rows,
            )
            edge_mask = ((cleaned.time_s >= cleaned.time_s[0] + 0.15)
                         & (cleaned.time_s <= cleaned.time_s[-1] - 0.15))
            episodes.append(
                {
                    "episode_id": episode_id,
                    "batch": batch,
                    "task": task,
                    "hand": hand,
                    "source_path": relative.as_posix(),
                    "source_sha256": source_sha256,
                    "content_sha256": content_sha256,
                    "frames": len(cleaned.time_s),
                    "trajectory_edge_trim_eligible": bool(
                        int(np.sum(edge_mask)) >= 2
                        and trajectory_has_motion(cleaned.position_m[edge_mask], cleaned.quaternion_wxyz[edge_mask])
                    ),
                    "artifact": artifact.as_posix(),
                }
            )
        source_records.append(
            {
                "source_path": relative.as_posix(),
                "source_sha256": source_sha256,
                "disposition": (
                    "eligible" if any(row["disposition"] == "eligible" for row in hand_records)
                    else "excluded"
                ),
                "hands": hand_records,
            }
        )
    episodes = assign_stratified_splits(episodes)
    manifest = {
        "schema_version": 1,
        "split_policy": {
            "method": "task_stratified_source_episode_hash_groups",
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "source_root": str(source_root.resolve()),
        "source_file_count": len(sources),
        "source_records": source_records,
        "episodes": episodes,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest

def trajectory_has_motion(position_m: np.ndarray, quaternion_wxyz: np.ndarray,
                          *, translation_threshold_m: float = 1e-3,
                          rotation_threshold_rad: float = np.deg2rad(1.0)) -> bool:
    """Reject recordings that contain no meaningful Cartesian motion."""
    position = np.asarray(position_m, dtype=float)
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if len(position) < 2 or position.shape != (len(position), 3) or quaternion.shape != (len(position), 4):
        return False
    translation_path = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
    dots = np.abs(np.sum(quaternion[:-1] * quaternion[1:], axis=1))
    rotation_path = float(np.sum(2.0 * np.arccos(np.clip(dots, 0.0, 1.0))))
    return translation_path > translation_threshold_m or rotation_path > rotation_threshold_rad
