"""Contracts for the repository-wide PiperX fixed-time mount study."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from .task_family import TaskFamily, family_from_path


STUDY_MODES = (
    "baseline",
    "upright_table",
    "horizontal_wall",
    "inverted",
)
SHARD_SCHEMA = "piperx-multitask-fixed-time-shard-v1"

_DUAL_HAND_COLUMNS = {
    "t",
    "left_tcp_pos_x", "left_tcp_pos_y", "left_tcp_pos_z",
    "left_tcp_quat_w", "left_tcp_quat_x", "left_tcp_quat_y",
    "left_tcp_quat_z",
    "right_tcp_pos_x", "right_tcp_pos_y", "right_tcp_pos_z",
    "right_tcp_quat_w", "right_tcp_quat_x", "right_tcp_quat_y",
    "right_tcp_quat_z",
}


@dataclass(frozen=True, order=True)
class TrajectorySpec:
    family: TaskFamily
    take: str
    path: Path
    source_sha256: str
    row_count: int

    @property
    def key(self) -> str:
        return f"{self.family.key}/{self.take}"


@dataclass(frozen=True)
class StudyConfig:
    modes: tuple[str, ...] = STUDY_MODES
    position_tolerance_m: float = 0.001
    orientation_tolerance_deg: float = 0.5
    anchor_restarts: int = 40

    def __post_init__(self):
        if self.modes != STUDY_MODES:
            raise ValueError("study modes must use the published fixed ordering")
        if self.position_tolerance_m != 0.001:
            raise ValueError("position tolerance must remain 1 mm")
        if self.orientation_tolerance_deg != 0.5:
            raise ValueError("orientation tolerance must remain 0.5 degrees")
        if self.anchor_restarts < 1:
            raise ValueError("anchor restarts must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = _DUAL_HAND_COLUMNS - columns
        if missing:
            raise ValueError(f"{path}: missing dual-hand columns: {sorted(missing)}")
        previous = None
        rows = 0
        for row in reader:
            timestamp = float(row["t"])
            if previous is not None and timestamp <= previous:
                raise ValueError(f"{path}: source timestamps are not strictly increasing")
            previous = timestamp
            rows += 1
    if rows < 2:
        raise ValueError(f"{path}: dual-hand trajectory is too short")
    return rows


def discover_dual_hand_trajectories(factory_root: Path) -> tuple[TrajectorySpec, ...]:
    """Return every accepted repository dual-hand source trajectory."""
    factory_root = Path(factory_root).resolve()
    specs = []
    for path in sorted(factory_root.rglob("handheld_*.csv")):
        if "_rejected_short" in path.parts:
            continue
        row_count = _inspect_csv(path)
        stem = path.stem
        take = stem.rsplit("_", 1)[-1]
        if len(take) != 6 or not take.isdigit():
            raise ValueError(f"{path}: source take must be a six-digit suffix")
        specs.append(TrajectorySpec(
            family=family_from_path(path, factory_root),
            take=take,
            path=path.resolve(),
            source_sha256=_sha256(path),
            row_count=row_count,
        ))
    keys = [item.key for item in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("dual-hand trajectory keys must be unique")
    return tuple(specs)


def _array(payload: Mapping, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in payload:
        raise ValueError(f"fixed-time shard is missing {name}")
    value = np.asarray(payload[name])
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value


def _source_timestamps(path: Path, count: int) -> np.ndarray:
    """Read the authoritative prefix of the recorded source timeline."""
    values = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if len(values) == count:
                break
            values.append(float(row["t"]))
    if len(values) != count:
        raise ValueError("raw CSV contains fewer timestamps than the shard")
    timestamps = np.asarray(values, dtype=float)
    return timestamps - timestamps[0]


def _finite_array(payload: Mapping, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = _array(payload, name, shape).astype(float)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def validate_shard(payload: Mapping, spec: TrajectorySpec, mode: str):
    """Validate fixed-time identity and mandatory per-frame evidence."""
    if payload.get("schema") != SHARD_SCHEMA:
        raise ValueError("unexpected fixed-time shard schema")
    if mode not in STUDY_MODES or payload.get("mode") != mode:
        raise ValueError("fixed-time shard mount mode mismatch")
    if payload.get("retiming_applied") is not False:
        raise ValueError("fixed-time shard must not apply retiming")
    count = spec.row_count
    source_time = _finite_array(payload, "source_time_s", (count,))
    fixed_time = _finite_array(payload, "fixed_time_s", (count,))
    if not np.array_equal(fixed_time, source_time):
        raise ValueError("fixed-time schedule must equal source timestamps")
    if not np.all(np.diff(source_time) > 0):
        raise ValueError("source timestamps must be strictly increasing")
    if not np.array_equal(source_time, _source_timestamps(spec.path, count)):
        raise ValueError("source timestamps must equal the raw CSV timeline")
    for name in ("left_accept", "right_accept", "both_accept", "collision",
                 "edge_collision", "topology_valid", "left_source_valid",
                 "right_source_valid"):
        _array(payload, name, (count,))
    left = np.asarray(payload["left_accept"], bool)
    right = np.asarray(payload["right_accept"], bool)
    both = np.asarray(payload["both_accept"], bool)
    left_source_valid = np.asarray(payload["left_source_valid"], bool)
    right_source_valid = np.asarray(payload["right_source_valid"], bool)
    if not np.array_equal(both, left & right):
        raise ValueError("both_accept must equal left_accept and right_accept")
    position = _finite_array(payload, "position_error_m", (count, 2))
    orientation = _finite_array(payload, "orientation_error_rad", (count, 2))
    if np.any(left & ~left_source_valid) or np.any(right & ~right_source_valid):
        raise ValueError("accepted frames must use source-valid TCP poses")
    expected_left = (left_source_valid
                     & (position[:, 0] <= 0.001 + 1e-12)
                     & (orientation[:, 0] <= np.deg2rad(0.5) + 1e-12))
    expected_right = (right_source_valid
                      & (position[:, 1] <= 0.001 + 1e-12)
                      & (orientation[:, 1] <= np.deg2rad(0.5) + 1e-12))
    if not (np.array_equal(left, expected_left)
            and np.array_equal(right, expected_right)):
        raise ValueError("accept flags must exactly match the 1 mm / 0.5 degree pose tolerances")
    collision = np.asarray(payload["collision"], dtype=bool)
    edge_collision = np.asarray(payload["edge_collision"], dtype=bool)
    topology_valid = np.asarray(payload["topology_valid"], dtype=bool)
    if np.any(collision):
        raise ValueError("fixed-time shard must be collision-free")
    if np.any(edge_collision):
        raise ValueError("fixed-time shard must be edge-collision-free")
    if not np.all(topology_valid):
        raise ValueError("fixed-time shard must be topology-valid")
    qpos = np.asarray(payload.get("qpos"), dtype=float)
    if qpos.ndim != 2 or qpos.shape[0] != count:
        raise ValueError("qpos evidence shape is invalid")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("qpos must contain only finite values")
    velocity = np.asarray(payload.get("velocity_rad_s"), dtype=float)
    acceleration = np.asarray(payload.get("acceleration_rad_s2"), dtype=float)
    if (velocity.ndim != 2 or velocity.shape[0] != count
            or acceleration.shape != velocity.shape):
        raise ValueError("velocity and acceleration evidence shapes are invalid")
    if not (np.all(np.isfinite(velocity))
            and np.all(np.isfinite(acceleration))):
        raise ValueError("velocity and acceleration must contain only finite values")
    return payload


__all__ = [
    "SHARD_SCHEMA",
    "STUDY_MODES",
    "StudyConfig",
    "TrajectorySpec",
    "discover_dual_hand_trajectories",
    "validate_shard",
]
