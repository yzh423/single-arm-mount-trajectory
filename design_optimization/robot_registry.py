"""Versioned robot-model registry for the thirteen-arm benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    source_model: Path
    model_format: str
    base_link: str
    flange_link: str
    active_joints: tuple[str, ...]
    locked_joints_rad: Mapping[str, tuple[float, float]]
    source_tool_length_m: float = 0.0

    @property
    def active_dof(self) -> int:
        return len(self.active_joints) - len(self.locked_joints_rad)


def load_robot_registry(path: str | Path) -> dict[str, RobotSpec]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("robot registry schema_version must be 1")
    project_root = path.resolve().parent.parent
    registry: dict[str, RobotSpec] = {}
    for robot_id, row in payload.get("robots", {}).items():
        active_joints = tuple(str(name) for name in row["active_joints"])
        if len(active_joints) != len(set(active_joints)):
            raise ValueError(f"{robot_id}: duplicate active joint name")
        locked = {
            str(name): (float(bounds[0]), float(bounds[1]))
            for name, bounds in row.get("locked_joints_rad", {}).items()
        }
        unknown_locked = set(locked) - set(active_joints)
        if unknown_locked:
            raise ValueError(f"{robot_id}: locked joint is not in active chain")
        for name, bounds in locked.items():
            if bounds[0] > bounds[1]:
                raise ValueError(f"{robot_id}: invalid locked range for {name}")
        source = Path(row["source_model"])
        if not source.is_absolute():
            source = project_root / source
        spec = RobotSpec(
            robot_id=str(robot_id),
            source_model=source.resolve(),
            model_format=str(row["model_format"]),
            base_link=str(row["base_link"]),
            flange_link=str(row["flange_link"]),
            active_joints=active_joints,
            locked_joints_rad=locked,
            source_tool_length_m=float(row.get("source_tool_length_m", 0.0)),
        )
        if spec.model_format not in {"urdf", "mjcf"}:
            raise ValueError(f"{robot_id}: unsupported model format {spec.model_format}")
        if spec.active_dof not in {6, 7}:
            raise ValueError(f"{robot_id}: expected six or seven active DOF")
        registry[robot_id] = spec
    return registry
