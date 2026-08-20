"""Configuration and PDF-recommended shared mounts for dual PiperX."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .task_family import TaskFamily


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs/piperx_recommended_v31.json"


@dataclass(frozen=True)
class StrictAcceptConfig:
    position_tolerance_m: float
    orientation_tolerance_rad: float
    branch_guard_rad: float


@dataclass(frozen=True)
class DLSConfig:
    damping: float
    step_scale: float
    maximum_step_rad: float
    max_iterations: int
    position_error_clip_m: float
    orientation_error_clip_rad: float


@dataclass(frozen=True)
class ExecutionConfig:
    maximum_velocity_rad_s: float
    maximum_acceleration_rad_s2: float
    settle_time_s: float
    decision_time_s: float
    source_rate_hz: float
    dwell_frames: int
    lookahead_frames: int


@dataclass(frozen=True)
class MountFunnelConfig:
    coarse_xy_step_m: float
    bottom_z_offset_m: float
    workspace_samples: int
    workspace_radius_quantile: float
    orientation_cone_m: float
    coarse_stride_frames: int
    anchor_keep: int
    probe_keep: int
    probe_stride_frames: int
    full_keep: int


@dataclass(frozen=True)
class RecommendedMountSpec:
    morphology: str
    mode: str
    left_p_base_m: tuple[float, float, float]
    right_p_base_m: tuple[float, float, float]
    source_take: str
    left_tool_offset_quaternion_wxyz: tuple[float, float, float, float] | None
    right_tool_offset_quaternion_wxyz: tuple[float, float, float, float] | None
    tool_offset_selection: str | None
    left_yaw_deg: float
    right_yaw_deg: float
    coordinate_domain: str
    selection_method: str


@dataclass(frozen=True)
class PiperXRecommendedConfig:
    schema: str
    source_report: str
    accept: StrictAcceptConfig
    dls: DLSConfig
    execution: ExecutionConfig
    funnel: MountFunnelConfig
    anchor_restarts: int
    mounts: Mapping[str, RecommendedMountSpec]


@dataclass(frozen=True)
class WorldMount:
    family: TaskFamily
    morphology: str
    mode: str
    left_xyz_m: np.ndarray
    right_xyz_m: np.ndarray
    shared_base_z_m: float
    source_take: str
    left_yaw_deg: float = 0.0
    right_yaw_deg: float = 0.0
    coordinate_domain: str = "registered_world"
    selection_method: str = (
        "PDF recommended shared base with rigid task registration")

    @property
    def base_distance_m(self) -> float:
        return float(np.linalg.norm(self.left_xyz_m - self.right_xyz_m))

    @property
    def xy(self) -> dict[str, list[float]]:
        return {
            "left": self.left_xyz_m[:2].tolist(),
            "right": self.right_xyz_m[:2].tolist(),
        }

    @property
    def yaw_deg(self) -> dict[str, float]:
        return {
            "left": float(self.left_yaw_deg),
            "right": float(self.right_yaw_deg),
        }

    def as_scene_mount(self) -> dict:
        return {
            "family": self.family.key,
            "morphology": self.morphology,
            "mode": self.mode,
            "xy": self.xy,
            "yaw": self.yaw_deg,
            "shared_base_z_m": self.shared_base_z_m,
            "base_z_m": {
                "left": float(self.left_xyz_m[2]),
                "right": float(self.right_xyz_m[2]),
            },
            "base_distance_m": self.base_distance_m,
            "source_take": self.source_take,
            "coordinate_domain": self.coordinate_domain,
            "selection_method": self.selection_method,
        }


def _positive(value, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _xyz(value, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite coordinates")
    return tuple(float(item) for item in array)


def _finite(value, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_quaternion(value, name: str):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if (array.shape != (4,) or np.any(~np.isfinite(array)) or
            not np.isclose(np.linalg.norm(array), 1.0, atol=1e-10)):
        raise ValueError(f"{name} must be a normalized finite quaternion")
    return tuple(float(item) for item in array)


def load_recommended_config(path: Path = DEFAULT_CONFIG_PATH) -> PiperXRecommendedConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "piperx-recommended-v3.1":
        raise ValueError("unexpected PiperX recommended config schema")
    accept_payload = payload["accept"]
    accept = StrictAcceptConfig(
        _positive(accept_payload["position_tolerance_m"], "position tolerance"),
        np.deg2rad(_positive(
            accept_payload["orientation_tolerance_deg"],
            "orientation tolerance",
        )),
        _positive(accept_payload["branch_guard_rad"], "branch guard"),
    )
    dls_payload = payload["dls"]
    dls = DLSConfig(
        _positive(dls_payload["damping"], "DLS damping"),
        _positive(dls_payload["step_scale"], "DLS step scale"),
        _positive(dls_payload["maximum_step_rad"], "DLS maximum step"),
        int(dls_payload["max_iterations"]),
        _positive(dls_payload["position_error_clip_m"], "position error clip"),
        _positive(
            dls_payload["orientation_error_clip_rad"],
            "orientation error clip",
        ),
    )
    execution_payload = payload["execution"]
    execution = ExecutionConfig(
        _positive(execution_payload["maximum_velocity_rad_s"], "velocity"),
        _positive(
            execution_payload["maximum_acceleration_rad_s2"], "acceleration"
        ),
        _positive(execution_payload["settle_time_s"], "settle time"),
        _positive(execution_payload["decision_time_s"], "decision time"),
        _positive(execution_payload["source_rate_hz"], "source rate"),
        int(execution_payload["dwell_frames"]),
        int(execution_payload["lookahead_frames"]),
    )
    funnel_payload = payload["mount_funnel"]
    funnel = MountFunnelConfig(**{
        name: (int(value) if name in {
            "workspace_samples", "coarse_stride_frames", "anchor_keep",
            "probe_keep", "probe_stride_frames", "full_keep",
        } else float(value))
        for name, value in funnel_payload.items()
    })
    modes = payload["mount_modes"]
    valid_modes = {
        "baseline_frozen", "upright_table", "horizontal_forward", "inverted"
    }
    mounts = {}
    for key, record in payload["task_family_mounts"].items():
        family = TaskFamily.parse(key)
        morphology = str(record["morphology"])
        mode = str(record.get("mode", modes[morphology]))
        if mode not in valid_modes:
            raise ValueError(f"unsupported mount mode for {key}: {mode}")
        mounts[family.key] = RecommendedMountSpec(
            morphology=morphology,
            mode=mode,
            left_p_base_m=_xyz(record["left_p_base_m"], f"{key} left base"),
            right_p_base_m=_xyz(record["right_p_base_m"], f"{key} right base"),
            source_take=str(record["source_take"]),
            left_tool_offset_quaternion_wxyz=_optional_quaternion(
                record.get("left_tool_offset_quaternion_wxyz"),
                f"{key} left tool offset",
            ),
            right_tool_offset_quaternion_wxyz=_optional_quaternion(
                record.get("right_tool_offset_quaternion_wxyz"),
                f"{key} right tool offset",
            ),
            tool_offset_selection=(
                None if record.get("tool_offset_selection") is None
                else str(record["tool_offset_selection"])
            ),
            left_yaw_deg=_finite(
                record.get("left_yaw_deg", 0.0), f"{key} left yaw"),
            right_yaw_deg=_finite(
                record.get("right_yaw_deg", 0.0), f"{key} right yaw"),
            coordinate_domain=str(record.get(
                "coordinate_domain", "source_frame")),
            selection_method=str(record.get(
                "selection_method",
                "PDF recommended shared base with rigid task registration")),
        )
        if mounts[family.key].coordinate_domain not in {
                "source_frame", "registered_world"}:
            raise ValueError(f"unsupported coordinate domain for {key}")
    anchor_restarts = int(payload["anchor_restarts"])
    if anchor_restarts != 40:
        raise ValueError("recommended anchor restart count must remain 40")
    if dls.max_iterations < 1 or min(
        execution.dwell_frames, execution.lookahead_frames,
        funnel.workspace_samples, funnel.coarse_stride_frames,
        funnel.anchor_keep, funnel.probe_keep,
        funnel.probe_stride_frames, funnel.full_keep,
    ) < 1:
        raise ValueError("recommended integer budgets must be positive")
    return PiperXRecommendedConfig(
        schema=payload["schema"],
        source_report=str(payload["source_report"]),
        accept=accept,
        dls=dls,
        execution=execution,
        funnel=funnel,
        anchor_restarts=anchor_restarts,
        mounts=MappingProxyType(mounts),
    )


def world_mount_for_family(
    config: PiperXRecommendedConfig,
    family: TaskFamily,
    registration_rotation: np.ndarray,
    registration_translation_m: np.ndarray,
) -> WorldMount:
    """Apply the task registration to a PDF-provided source-frame base pair."""

    try:
        mount = config.mounts[family.key]
    except KeyError as error:
        raise KeyError(f"no recommended PiperX mount for {family.key}") from error
    rotation = np.asarray(registration_rotation, dtype=float)
    translation = np.asarray(registration_translation_m, dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("registration must contain a 3x3 rotation and 3-vector")
    if np.any(~np.isfinite(rotation)) or np.any(~np.isfinite(translation)):
        raise ValueError("registration must be finite")
    if mount.coordinate_domain == "registered_world":
        left = np.asarray(mount.left_p_base_m, dtype=float)
        right = np.asarray(mount.right_p_base_m, dtype=float)
    else:
        left = rotation @ np.asarray(mount.left_p_base_m) + translation
        right = rotation @ np.asarray(mount.right_p_base_m) + translation
    if not np.isclose(left[2], right[2], rtol=0.0, atol=1e-9):
        raise ValueError("recommended mount must retain a shared height after registration")
    shared_z = float(0.5 * (left[2] + right[2]))
    left = left.copy()
    right = right.copy()
    left[2] = right[2] = shared_z
    return WorldMount(
        family=family,
        morphology=mount.morphology,
        mode=mount.mode,
        left_xyz_m=left,
        right_xyz_m=right,
        shared_base_z_m=shared_z,
        source_take=mount.source_take,
        left_yaw_deg=mount.left_yaw_deg,
        right_yaw_deg=mount.right_yaw_deg,
        coordinate_domain=(
            "registered_world" if mount.coordinate_domain == "source_frame"
            else mount.coordinate_domain),
        selection_method=mount.selection_method,
    )
