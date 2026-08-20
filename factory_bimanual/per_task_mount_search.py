"""Shared contracts for independent per-task PiperX mount searches."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json

import numpy as np

from .factory_task_catalog import TaskRepresentative
from .mount_orientation import FACTORY_BATCH_MOUNT_MODES
from .registration import RigidTaskRegistration, register_task
from .source_data import load_factory_task


@dataclass(frozen=True)
class PerTaskSearchConfig:
    modes: tuple[str, ...] = FACTORY_BATCH_MOUNT_MODES
    coarse_budget: int = 36
    dense_budget: int = 6
    local_budget: int = 12
    finalist_budget: int = 4
    minimum_separation_m: float = .60
    schema: str = "piperx-per-task-mount-v1-forward"

    def __post_init__(self):
        if min(self.coarse_budget, self.dense_budget,
               self.local_budget, self.finalist_budget) < 1:
            raise ValueError("all search budgets must be positive")
        if self.minimum_separation_m < .60:
            raise ValueError("minimum separation cannot be below 0.60 m")


def load_registered_representative(representative: TaskRepresentative):
    source = load_factory_task(
        representative.csv_path, representative.task_name,
        max_translation_jump_m=.20, repair_invalid_pose_rows=True)
    points = np.vstack((source.left_position_m, source.right_position_m))
    translation = np.asarray((
        -points[:, 0].mean(), -points[:, 1].mean(), .90 - points[:, 2].min()))
    return register_task(
        source, RigidTaskRegistration(np.eye(3), translation))


def prefix_registered_task(task, rows: int):
    count = min(len(task.time_s), max(2, int(rows)))
    values = {}
    for name, value in task.__dict__.items():
        if isinstance(value, np.ndarray) and value.ndim and len(value) == len(task.time_s):
            values[name] = value[:count].copy()
    return replace(task, **values)


def candidate_fingerprint(source_sha256, mode, mount,
                          config: PerTaskSearchConfig, *, stage="candidate",
                          settings=None):
    payload = {
        "schema": config.schema,
        "source_sha256": str(source_sha256),
        "mode": str(mode), "stage": str(stage),
        "mount": mount, "config": asdict(config),
        "settings": {} if settings is None else settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def rank_full_finalist(record):
    return (
        -float(record["synchronous_strict_coverage"]),
        int(record["longest_failure_frames"]),
        float(record["mean_pair_pose_error"]),
        float(record["p95_pair_pose_error"]),
        -float(record["minimum_joint_limit_margin_rad"]),
        -float(record["p10_pair_singularity_margin"]),
        -float(record["base_distance_m"]),
    )


def summarize_quality_arrays(*, position_error, orientation_error,
                             joint_margins, singularity_margins):
    pose = np.concatenate([
        np.asarray(position_error[side], float)
        + np.asarray(orientation_error[side], float)
        for side in ("left", "right")])
    joints = np.concatenate([
        np.asarray(joint_margins[side], float) for side in ("left", "right")])
    singularity = np.minimum(
        np.asarray(singularity_margins["left"], float),
        np.asarray(singularity_margins["right"], float))
    pose = pose[np.isfinite(pose)]
    joints = joints[np.isfinite(joints)]
    singularity = singularity[np.isfinite(singularity)]
    return {
        "mean_pair_pose_error": float(np.mean(pose)) if len(pose) else 1e9,
        "p95_pair_pose_error": float(np.percentile(pose, 95)) if len(pose) else 1e9,
        "minimum_joint_limit_margin_rad": (
            float(np.min(joints)) if len(joints) else 0.0),
        "p10_pair_singularity_margin": (
            float(np.percentile(singularity, 10)) if len(singularity) else 0.0),
    }


def _is_fully_audited_safe(record):
    source_rows = int(record.get("source_row_count", 0))
    return (
        source_rows > 0
        and int(record.get("audited_source_rows", -1)) == source_rows
        and int(record.get("pair_collision_frames", -1)) == 0
        and int(record.get("pair_edge_collision_frames", -1)) == 0
    )


def select_safe_layout(records):
    safe = [record for record in records if _is_fully_audited_safe(record)]
    if not safe:
        raise RuntimeError("no fully audited collision-free layout")
    return min(safe, key=rank_full_finalist)
