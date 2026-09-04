"""Immutable robot metadata for the isolated factory bimanual experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import mujoco

from scripts.strict_urdf_model_audit import MODELS
from scripts.strict_urdf_model_audit import load_native_spec


Side = Literal["left", "right"]
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BimanualRobotContract:
    name: str
    source_urdf: Path
    arm_joint_names: tuple[str, ...]
    tcp_link_name: str
    base_link_name: str
    joint_limits_rad: tuple[tuple[float, float], ...] = ()
    tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def dof_per_arm(self) -> int:
        return len(self.arm_joint_names)

    def prefixed_joint_names(self, side: Side) -> tuple[str, ...]:
        if side not in ("left", "right"):
            raise ValueError(f"invalid side: {side!r}")
        return tuple(f"{side}_{name}" for name in self.arm_joint_names)


def _geometry_source_sha256(model_path: Path, asset_roots) -> str:
    """Hash model text and every vendor asset that can affect geometry."""
    digest = hashlib.sha256()
    sources = [("model", Path(model_path))]
    for root_index, root in enumerate(asset_roots):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(
                (item for item in root.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(root).as_posix()):
            if ".git" in path.relative_to(root).parts:
                continue
            sources.append((
                f"asset-root-{root_index}/{path.relative_to(root).as_posix()}",
                path,
            ))
    for label, path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        label_bytes = label.encode("utf-8")
        digest.update(len(label_bytes).to_bytes(8, "big"))
        digest.update(label_bytes)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def robot_geometry_sha256(name: str) -> str:
    """Return a path-independent fingerprint of one native robot snapshot."""
    try:
        entry = MODELS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported bimanual robot: {name!r}") from exc
    roots = []
    if entry.mesh_dir is not None:
        roots.append(entry.mesh_dir)
    roots.extend((entry.package_roots or {}).values())
    return _geometry_source_sha256(entry.path, tuple(dict.fromkeys(roots)))


def _contract(name: str) -> BimanualRobotContract:
    entry = MODELS[name]
    model = load_native_spec(entry).compile()
    limits = []
    for joint_name in entry.joints:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise RuntimeError(f"{name}: missing official joint {joint_name}")
        limits.append(tuple(float(value) for value in model.jnt_range[joint_id]))
    return BimanualRobotContract(
        name=name,
        source_urdf=entry.path.resolve(),
        arm_joint_names=entry.joints,
        tcp_link_name=entry.tcp_parent,
        base_link_name=entry.base_link or "base_link",
        joint_limits_rad=tuple(limits),
        tcp_offset_m=entry.tool_translation_m,
    )


ROBOT_CONTRACTS = MappingProxyType(
    {
        name: _contract(name)
        for name in ("xarm6", "franka_panda", "i2rt_yam", "piperx", "ur5")
    }
)


def get_robot_contract(name: str) -> BimanualRobotContract:
    try:
        return ROBOT_CONTRACTS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported bimanual robot: {name!r}") from exc
