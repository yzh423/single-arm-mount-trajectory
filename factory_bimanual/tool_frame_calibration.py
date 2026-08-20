"""Fixed source-frame to robot-TCP orientation calibration contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


CALIBRATION_TASKS = ("fold_box", "seal_bag")
LOCAL_REFINEMENT_LIMIT_DEG = 15.0


def source_file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proper_axis_rotations():
    """Return all 24 right-handed signed Cartesian-axis mappings."""
    output = []
    for axes in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            for column, (axis, sign) in enumerate(zip(axes, signs)):
                matrix[axis, column] = sign
            if np.linalg.det(matrix) > 0.5:
                output.append(matrix)
    output.sort(key=lambda value: tuple(value.ravel()))
    return np.asarray(output)


def _normalize_quaternions(values):
    quaternions = np.asarray(values, dtype=float)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("source quaternions must have shape (frames, 4)")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(~np.isfinite(quaternions)) or np.any(norms < 1e-12):
        raise ValueError("source quaternions must be finite and nonzero")
    result = quaternions / norms[:, None]
    for row in range(1, len(result)):
        if float(result[row - 1] @ result[row]) < 0.0:
            result[row] *= -1.0
    return result


def _quaternion_multiply(left, right):
    lw, lx, ly, lz = np.asarray(left, dtype=float)
    rw, rx, ry, rz = np.asarray(right, dtype=float)
    return np.asarray((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def apply_fixed_tool_rotation(source_quaternions_wxyz,
                              offset_quaternion_wxyz):
    """Apply one constant source-to-TCP rotation to every source frame."""
    source = _normalize_quaternions(source_quaternions_wxyz)
    offset = np.asarray(offset_quaternion_wxyz, dtype=float)
    if offset.shape != (4,) or np.any(~np.isfinite(offset)):
        raise ValueError("offset quaternion must be finite with shape (4,)")
    norm = float(np.linalg.norm(offset))
    if norm < 1e-12:
        raise ValueError("offset quaternion must be nonzero")
    offset /= norm
    mapped = np.asarray([_quaternion_multiply(value, offset)
                         for value in source])
    return _normalize_quaternions(mapped)


def validate_local_refinement_deg(xyz_deg):
    values = np.asarray(xyz_deg, dtype=float)
    if values.shape != (3,) or np.any(~np.isfinite(values)):
        raise ValueError("local refinement must be three finite angles")
    if np.any(np.abs(values) > LOCAL_REFINEMENT_LIMIT_DEG + 1e-12):
        raise ValueError("local refinement exceeds the 15 degree limit")
    return values


def fixed_offset_quaternion(axis_rotation_index, local_xyz_deg):
    """Compose one proper axis mapping with bounded local XYZ refinement."""
    index = int(axis_rotation_index)
    rotations = proper_axis_rotations()
    if index < 0 or index >= len(rotations):
        raise ValueError("axis rotation index must be in [0, 24)")
    local = validate_local_refinement_deg(local_xyz_deg)
    matrix = rotations[index] @ Rotation.from_euler(
        "xyz", local, degrees=True).as_matrix()
    xyzw = Rotation.from_matrix(matrix).as_quat()
    quaternion = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion / np.linalg.norm(quaternion)


def representative_quaternion_indices(quaternions_wxyz, *, maximum):
    """Choose deterministic endpoints, motion boundaries and farthest samples."""
    values = _normalize_quaternions(quaternions_wxyz)
    limit = int(maximum)
    if limit < 2:
        raise ValueError("maximum must be at least two")
    if len(values) <= limit:
        return np.arange(len(values), dtype=int)
    selected = [0, len(values) - 1]
    if limit > 2:
        # A pure farthest-point sampler can select the quiet frame just after
        # a large orientation jump and omit the transition boundary itself.
        # Seed it with the destination of the largest consecutive rotation so
        # calibration probes explicitly cover at least one motion boundary.
        consecutive_similarity = np.abs(np.sum(values[1:] * values[:-1], axis=1))
        largest_jump = int(np.argmin(consecutive_similarity)) + 1
        if largest_jump not in selected:
            selected.append(largest_jump)
    while len(selected) < limit:
        similarity = np.max(np.abs(values @ values[selected].T), axis=1)
        distance = 1.0 - np.clip(similarity, 0.0, 1.0)
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    return np.asarray(sorted(selected), dtype=int)


def rank_calibration_result(record):
    """Lexically rank one fixed cross-task coordinate calibration."""
    return (
        -float(record.get("synchronous_strict_coverage", 0.0)),
        int(record.get("longest_failure_run_frames", 10**9)),
        -float(record.get("connectable_safe_branch_ratio", 0.0)),
        -float(record.get("minimum_singularity_margin", -np.inf)),
        -float(record.get("minimum_joint_limit_margin_rad", -np.inf)),
        float(record.get("mean_normalized_pose_error", np.inf)),
    )


@dataclass(frozen=True)
class CalibrationArtifact:
    version: int
    robot: str
    tasks: tuple[str, ...]
    source_fingerprints: dict[str, str]
    left_offset_quaternion_wxyz: tuple[float, float, float, float]
    right_offset_quaternion_wxyz: tuple[float, float, float, float]
    left_axis_rotation_index: int
    right_axis_rotation_index: int
    left_local_xyz_deg: tuple[float, float, float]
    right_local_xyz_deg: tuple[float, float, float]
    metrics: dict[str, float]

    def __post_init__(self):
        if self.version != 1 or self.robot != "piperx":
            raise ValueError("unsupported calibration version or robot")
        if tuple(self.tasks) != CALIBRATION_TASKS:
            raise ValueError("calibration must be shared by Fold Box and Seal Bag")
        if set(self.source_fingerprints) != set(CALIBRATION_TASKS):
            raise ValueError("source fingerprints must cover both tasks")
        for name in ("left_offset_quaternion_wxyz",
                     "right_offset_quaternion_wxyz"):
            value = np.asarray(getattr(self, name), dtype=float)
            if (value.shape != (4,) or np.any(~np.isfinite(value)) or
                    not np.isclose(np.linalg.norm(value), 1.0, atol=1e-10)):
                raise ValueError(f"{name} must be a normalized quaternion")
        if not 0 <= self.left_axis_rotation_index < 24:
            raise ValueError("left axis rotation index must be in [0, 24)")
        if not 0 <= self.right_axis_rotation_index < 24:
            raise ValueError("right axis rotation index must be in [0, 24)")
        validate_local_refinement_deg(self.left_local_xyz_deg)
        validate_local_refinement_deg(self.right_local_xyz_deg)

    def write(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    @classmethod
    def read(cls, path, *, expected_source_fingerprints=None):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for name in (
            "tasks", "left_offset_quaternion_wxyz",
            "right_offset_quaternion_wxyz", "left_local_xyz_deg",
            "right_local_xyz_deg",
        ):
            payload[name] = tuple(payload[name])
        artifact = cls(**payload)
        if (expected_source_fingerprints is not None and
                artifact.source_fingerprints != expected_source_fingerprints):
            raise ValueError("calibration source fingerprint mismatch")
        return artifact


__all__ = [
    "CALIBRATION_TASKS",
    "CalibrationArtifact",
    "LOCAL_REFINEMENT_LIMIT_DEG",
    "apply_fixed_tool_rotation",
    "fixed_offset_quaternion",
    "proper_axis_rotations",
    "rank_calibration_result",
    "representative_quaternion_indices",
    "source_file_fingerprint",
    "validate_local_refinement_deg",
]
