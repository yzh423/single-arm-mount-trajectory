"""Build isolated same-model bimanual MuJoCo scenes from canonical URDFs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .robot_contracts import BimanualRobotContract
from scripts.strict_urdf_model_audit import MODELS, load_native_spec


@dataclass(frozen=True)
class SceneManifest:
    robot: str
    source_urdf: str
    output_xml: str
    spacing_m: float
    table_height_m: float
    left_base_position: tuple[float, float, float]
    right_base_position: tuple[float, float, float]
    left_joint_names: tuple[str, ...]
    right_joint_names: tuple[str, ...]


def _load_child(contract: BimanualRobotContract) -> mujoco.MjSpec:
    entry = MODELS[contract.name]
    if contract.source_urdf.resolve() != entry.path.resolve():
        raise ValueError(f"{contract.name}: bimanual contract diverges from strict registry")
    return load_native_spec(entry)


def _add_tcp_site(child: mujoco.MjSpec, contract: BimanualRobotContract) -> None:
    bodies = {body.name: body for body in child.worldbody.find_all("body")}
    body = bodies.get(contract.tcp_link_name)
    if body is None:
        raise ValueError(
            f"{contract.name} TCP parent body {contract.tcp_link_name!r} is absent"
        )
    body.add_site(
        name="tcp", pos=list(contract.tcp_offset_m),
        size=[0.012, 0.012, 0.012], rgba=[0.1, 0.8, 0.2, 1.0],
    )


def build_same_model_scene(
    contract: BimanualRobotContract,
    spacing_m: float,
    output_path: Path,
    *,
    table_height_m: float = 0.75,
    mount_xy_m: dict[str, tuple[float, float]] | None = None,
    mount_yaw_deg: dict[str, float] | None = None,
    mount_adapter_height_m: float = 0.08,
    mount_base_z_m: float | None = None,
    mount_quaternion_wxyz: dict[str, tuple[float, float, float, float]] | None = None,
    mount_support_mode: str | None = None,
) -> SceneManifest:
    if not spacing_m > 0:
        raise ValueError("spacing_m must be positive")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene = mujoco.MjSpec()
    scene.modelname = f"factory_bimanual_{contract.name}"
    scene.compiler.degree = False
    scene.worldbody.add_geom(
        name="workbench", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, table_height_m - 0.04], size=[0.9, 0.7, 0.04],
        rgba=[0.55, 0.42, 0.3, 1.0],
    )
    default_xy = {"left": (-spacing_m / 2, 0.0), "right": (spacing_m / 2, 0.0)}
    xy = default_xy if mount_xy_m is None else mount_xy_m
    yaw = ({"left": 0.0, "right": 180.0} if mount_yaw_deg is None
           else mount_yaw_deg)
    if mount_support_mode not in (
            None, "upright_table", "horizontal_wall", "horizontal_forward",
            "inverted"):
        raise ValueError("unsupported mount support mode")
    if mount_support_mode == "inverted":
        if mount_base_z_m is None:
            raise ValueError("inverted support requires mount_base_z_m")
        scene.worldbody.add_geom(
            name="mount_ceiling", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0., 0., float(mount_base_z_m) + mount_adapter_height_m + .04],
            size=[.9, .7, .04], rgba=[.30, .32, .36, 1.],
            contype=1, conaffinity=1)
    for side in ("left", "right"):
        x, y = (float(v) for v in xy[side])
        angle = np.deg2rad(float(yaw[side]))
        base_z = (table_height_m + mount_adapter_height_m
                  if mount_base_z_m is None else float(mount_base_z_m))
        quat = (np.asarray([
            float(np.cos(angle / 2)), 0., 0., float(np.sin(angle / 2))])
            if mount_quaternion_wxyz is None
            else np.asarray(mount_quaternion_wxyz[side], dtype=float))
        if quat.shape != (4,) or not np.isfinite(quat).all():
            raise ValueError(f"{side} mount quaternion must be finite shape (4,)")
        quat /= np.linalg.norm(quat)
        rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation, quat)
        axis = rotation.reshape(3, 3)[:, 2]
        adapter_center = np.asarray([x, y, base_z]) - (
            axis * mount_adapter_height_m / 2)
        if mount_support_mode in ("horizontal_wall", "horizontal_forward"):
            pole_xy = np.asarray([x, y]) - axis[:2] * mount_adapter_height_m
            pole_half_height = (base_z - table_height_m) / 2
            if pole_half_height <= 0:
                raise ValueError("horizontal pole mount must be above the table")
            scene.worldbody.add_geom(
                name=f"{side}_mount_pole",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[float(pole_xy[0]), float(pole_xy[1]),
                     table_height_m + pole_half_height],
                size=[.05, pole_half_height, 0.],
                rgba=[.30, .32, .36, 1.],
                contype=1, conaffinity=1)
        scene.worldbody.add_geom(
            name=f"{side}_mount_adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=adapter_center.tolist(), quat=quat.tolist(),
            size=[0.075, mount_adapter_height_m / 2, 0.0],
            rgba=[0.16, 0.18, 0.22, 1.0], contype=1, conaffinity=1,
        )
        mount = scene.worldbody.add_body(
            name=f"{side}_base_mount",
            pos=[x, y, base_z], quat=quat.tolist(),
        )
        frame = mount.add_frame(name=f"{side}_base")
        child = _load_child(contract)
        scene.meshdir = child.meshdir
        _add_tcp_site(child, contract)
        scene.attach(child, prefix=f"{side}_", frame=frame)
        target = scene.worldbody.add_body(
            name=f"{side}_target", pos=[x, y + 0.35, table_height_m + 0.35], mocap=True
        )
        target.add_geom(
            name=f"{side}_target_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.018, 0.018, 0.018], rgba=[0.9, 0.15, 0.1, 0.6],
            contype=0, conaffinity=0,
        )

    scene.compile()
    scene.to_file(str(output_path))
    # MjSpec's writer rounds ranges to six significant digits. Restore the
    # audited vendor radians so reopening an exported scene cannot tighten a
    # joint by serialization error.
    root = ET.fromstring(output_path.read_text(encoding="utf-8"))
    # MjSpec.attach applies the body prefix to mesh filenames as well as mesh
    # names.  The files in the deterministic native-asset cache intentionally
    # retain their source names, so remove only an attach-added side prefix
    # when the corresponding cached file exists.
    mesh_directory = Path(scene.meshdir)
    for mesh in root.findall(".//asset/mesh"):
        filename = mesh.get("file", "")
        for side in ("left", "right"):
            prefix = f"{side}_"
            if filename.startswith(prefix):
                source_filename = filename.removeprefix(prefix)
                if (mesh_directory / source_filename).is_file():
                    mesh.set("file", source_filename)
                break
    native_limits = dict(zip(contract.arm_joint_names, contract.joint_limits_rad))
    for joint in root.findall(".//joint"):
        name = joint.get("name", "")
        for side in ("left", "right"):
            prefix = f"{side}_"
            source_name = name.removeprefix(prefix)
            if name.startswith(prefix) and source_name in native_limits:
                lower, upper = native_limits[source_name]
                joint.set("range", f"{lower:.12g} {upper:.12g}")
                break
    output_path.write_text(
        ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8"
    )
    manifest = SceneManifest(
        robot=contract.name,
        source_urdf=str(contract.source_urdf),
        output_xml=str(output_path),
        spacing_m=float(spacing_m),
        table_height_m=float(table_height_m),
        left_base_position=(float(xy["left"][0]), float(xy["left"][1]),
                            table_height_m + mount_adapter_height_m
                            if mount_base_z_m is None else float(mount_base_z_m)),
        right_base_position=(float(xy["right"][0]), float(xy["right"][1]),
                             table_height_m + mount_adapter_height_m
                             if mount_base_z_m is None else float(mount_base_z_m)),
        left_joint_names=contract.prefixed_joint_names("left"),
        right_joint_names=contract.prefixed_joint_names("right"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return manifest
