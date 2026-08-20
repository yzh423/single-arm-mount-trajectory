"""Qualify all 13 real-mesh models before strict trajectory/video jobs run."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.strict_mujoco_model import sampled_maximum_tcp_reach_m
from design_optimization.robot_contract import COMMON_TOOL_LENGTH_M

@dataclass(frozen=True)
class ModelEntry:
    path: Path
    joints: tuple[str, ...]
    tcp_parent: str
    tool_offset_m: float
    expected_dof: int
    root_body: str | None = None
    mesh_dir: Path | None = None
    tool_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    locked_joint_ranges: dict[str, tuple[float, float]] | None = None
    flange_parent: str | None = None
    geometry_authority: str | None = None
    package_roots: dict[str, Path] | None = None
    base_link: str | None = None
    tcp_authority: str = "legacy_unspecified"
    joint_limit_authority: str = "official_urdf"

    @property
    def tool_translation_m(self) -> tuple[float, float, float]:
        axis = np.asarray(self.tool_axis, dtype=float)
        axis /= np.linalg.norm(axis)
        return tuple(float(value) for value in axis * self.tool_offset_m)

    @property
    def tool_quaternion_wxyz(self) -> tuple[float, float, float, float]:
        """Rotate canonical TCP +Z onto this URDF's physical tool axis."""
        z_axis = np.asarray(self.tool_axis, dtype=float)
        z_axis /= np.linalg.norm(z_axis)
        reference_y = np.asarray((0.0, 1.0, 0.0))
        if abs(float(reference_y @ z_axis)) > 0.95:
            reference_y = np.asarray((1.0, 0.0, 0.0))
        x_axis = np.cross(reference_y, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
        return tuple(float(value) for value in quaternion)


OFFICIAL = ROOT / "third_party/official_robot_models"
MODELS = {
    "doosan": ModelEntry(OFFICIAL / "doosan_m0609.urdf", tuple(f"joint_{i}" for i in range(1, 7)), "tool0", 0.139, 6, package_roots={"dsr_description2": OFFICIAL / "doosan_description"}, base_link="base_link", tcp_authority="specified_139mm"),
    "xarm6": ModelEntry(OFFICIAL / "official_derived/xarm_description/xarm6_official.urdf", tuple(f"joint{i}" for i in range(1, 7)), "link_tcp", 0.0, 6, package_roots={"xarm_description": OFFICIAL / "official_derived/xarm_description"}, base_link="link_base", tcp_authority="official_frame"),
    "ur5": ModelEntry(OFFICIAL / "official_derived/ur_description2/ur5_official.urdf", ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"), "tool0", 0.0, 6, package_roots={"ur_description": OFFICIAL / "official_derived/ur_description2"}, base_link="base_link", tcp_authority="official_frame"),
    "kinova_gen3_lite": ModelEntry(OFFICIAL / "kinova/ros2_kortex-main/kortex_description/robots/gen3_lite.urdf", tuple(f"joint_{i}" for i in range(1, 7)), "tool_frame", 0.0, 6, package_roots={"kortex_description": OFFICIAL / "kinova/ros2_kortex-main/kortex_description"}, base_link="base_link", tcp_authority="official_frame"),
    "arx_x5": ModelEntry(OFFICIAL / "arx/ARX_Model-af6fe43c873008a85bce6195c0f2160f1a1c14ce/X5/X5A/urdf/X5A.urdf", tuple(f"joint{i}" for i in range(1, 7)), "link6", COMMON_TOOL_LENGTH_M, 6, tool_axis=(1.0, 0.0, 0.0), package_roots={"X5A": OFFICIAL / "arx/ARX_Model-af6fe43c873008a85bce6195c0f2160f1a1c14ce/X5/X5A"}, base_link="base_link", tcp_authority="fallback_130mm", joint_limit_authority="official_urdf_unverified_placeholder"),
    "big_yam": ModelEntry(OFFICIAL / "i2rt/i2rt/robot_models/arm/big_yam/v1/big_yam.urdf", tuple(f"joint{i}" for i in range(1, 7)), "gripper", COMMON_TOOL_LENGTH_M, 6, tool_axis=(0.0, 0.0, -1.0), base_link="base", tcp_authority="fallback_130mm"),
    "franka_panda": ModelEntry(OFFICIAL / "official_derived/franka_description/panda_official.urdf", tuple(f"panda_joint{i}" for i in range(1, 8)), "panda_hand_tcp", 0.0, 7, package_roots={"franka_description": OFFICIAL / "official_derived/franka_description"}, base_link="panda_link0", tcp_authority="official_frame"),
    "franka_panda_locked_j3": ModelEntry(OFFICIAL / "official_derived/franka_description/panda_official.urdf", tuple(f"panda_joint{i}" for i in range(1, 8)), "panda_hand_tcp", 0.0, 7, locked_joint_ranges={"panda_joint3": (-0.0001, 0.0001)}, package_roots={"franka_description": OFFICIAL / "official_derived/franka_description"}, base_link="panda_link0", tcp_authority="official_frame"),
    "i2rt_yam": ModelEntry(OFFICIAL / "i2rt/i2rt/robot_models/arm/yam/v1/yam_linear_4310_d405.urdf", tuple(f"joint{i}" for i in range(1, 7)), "gripper", COMMON_TOOL_LENGTH_M, 6, tool_axis=(0.0, 0.0, -1.0), base_link="base", tcp_authority="fallback_130mm"),
    "nero": ModelEntry(OFFICIAL / "nero/urdf/nero.urdf", tuple(f"joint{i}" for i in range(1, 8)), "gripper_base", COMMON_TOOL_LENGTH_M, 7, mesh_dir=OFFICIAL / "nero/meshes", base_link="base_link", tcp_authority="fallback_130mm", joint_limit_authority="pinned_vendor_snapshot"),
    # The checked-in single-arm derivative expands the official v1 parallel
    # gripper xacro with its declared tcp_xyz default (0 0 0.0835).  The TCP is
    # therefore already represented by the hand_tcp body and must not receive
    # the generic 130 mm fallback a second time.
    "openarm": ModelEntry(OFFICIAL / "official_derived/openarm_description/openarm_v1_left_single.urdf", tuple(f"openarm_left_joint{i}" for i in range(1, 8)), "openarm_left_hand_tcp", 0.0, 7, root_body="openarm_left_link0", package_roots={"openarm_description": OFFICIAL / "openarm/openarm_description-main"}, base_link="openarm_left_link0", tcp_authority="official_xacro_default_83p5mm"),
    # PiPER-X is not kinematically interchangeable with the earlier PiPER.
    # This pinned native snapshot retains the PiPER-X J4/J5 geometry and its
    # declared gripper-centre ee_frame (115 mm from gripper_base).
    "piperx": ModelEntry(OFFICIAL / "piperx/PiperX.urdf", tuple(f"joint{i}" for i in range(1, 7)), "ee_frame", 0.0, 6, mesh_dir=OFFICIAL / "piperx/meshes", base_link="base_link", tcp_authority="model_ee_frame_115mm"),
    "willow": ModelEntry(OFFICIAL / "willow/urdf/Willow_mujoco.urdf", tuple(f"joint{i}" for i in range(1, 7)), "link6", COMMON_TOOL_LENGTH_M, 6, mesh_dir=OFFICIAL / "willow/meshes", tool_axis=(-1.0, 0.0, 0.0), package_roots={"Ragtime_Willow_description": OFFICIAL / "willow"}, base_link="base_link", tcp_authority="fallback_130mm", joint_limit_authority="pinned_vendor_snapshot"),
}


def load_native_spec(entry: ModelEntry) -> mujoco.MjSpec:
    """Load one workspace-local real-mesh arm from URDF or MJCF."""
    if not entry.path.is_file():
        raise FileNotFoundError(entry.path)
    if entry.path.suffix.lower() == ".urdf":
        # MuJoCo treats missing URDF visual/collision names as duplicate empty
        # geom names. Add deterministic import-only names without altering any
        # geometry, collision, inertia, or joint data in the source asset.
        root = ET.fromstring(entry.path.read_text(encoding="utf-8"))
        # MuJoCo defaults to discardvisual=true for URDF imports. Preserve the
        # vendor's official visual mesh layer; collision proxies must never be
        # promoted into rendered robot appearance.
        mujoco_extension = root.find("mujoco")
        if mujoco_extension is None:
            mujoco_extension = ET.SubElement(root, "mujoco")
        compiler_extension = mujoco_extension.find("compiler")
        if compiler_extension is None:
            compiler_extension = ET.SubElement(mujoco_extension, "compiler")
        compiler_extension.set("discardvisual", "false")
        # Keep each vendor visual asset authoritative. DAE files are resolved
        # and converted below from their exact package path; matching a visual
        # to a same-stem STL is unsafe because multi-model packages (notably
        # Universal Robots) reuse names such as upperarm.stl for every variant.
        for link in root.findall("link"):
            link_name = link.get("name", "link")
            for kind in ("visual", "collision"):
                for index, element in enumerate(link.findall(kind)):
                    if not element.get("name"):
                        element.set("name", f"{link_name}_{kind}_{index}")
        xml = ET.tostring(root, encoding="unicode")
        assets = {}
        for mesh_index, mesh_element in enumerate(root.findall(".//mesh")):
            filename = mesh_element.get("filename")
            if not filename:
                continue
            lookup_names = [filename]
            if filename.startswith("visual__"):
                lookup_names.append(filename.removeprefix("visual__"))
            candidates = []
            for lookup_filename in lookup_names:
                relative = Path(lookup_filename.replace("package://", ""))
                candidates.extend((entry.path.parent / relative, entry.path.parent / "meshes" / relative))
                if lookup_filename.startswith("package://") and relative.parts:
                    package_root = (entry.package_roots or {}).get(relative.parts[0])
                    if package_root is not None:
                        candidates.append(package_root / Path(*relative.parts[1:]))
                if lookup_filename.startswith("package://") and len(relative.parts) > 1:
                    candidates.append(entry.path.parent / Path(*relative.parts[1:]))
                if entry.mesh_dir is not None:
                    candidates.extend((entry.mesh_dir / relative, entry.mesh_dir / relative.name))
                for package_root in (entry.package_roots or {}).values():
                    candidates.extend(package_root.rglob(relative.name))
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"{entry.path}: missing mesh {filename}")
            if path.suffix.lower() == ".dae":
                import hashlib
                import trimesh
                digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
                converted = OFFICIAL / "official_derived/converted_meshes" / f"{path.stem}_{digest}.stl"
                converted.parent.mkdir(parents=True, exist_ok=True)
                if not converted.is_file():
                    scene = trimesh.load(str(path), force="scene")
                    scene.to_mesh().export(str(converted))
                path = converted
            asset_name = f"official_mesh_{mesh_index:04d}{path.suffix.lower()}"
            mesh_element.set("filename", asset_name)
            assets[asset_name] = path.read_bytes()
        xml = ET.tostring(root, encoding="unicode")
        spec = mujoco.MjSpec.from_string(xml, assets=assets)
        # In-memory assets support immediate compilation. Persist the same
        # bytes under a deterministic third-party cache as well so a composed
        # MjSpec exported to XML can be reopened without any Assets lookup.
        import hashlib
        cache_key = hashlib.sha256(str(entry.path.resolve()).encode("utf-8")).hexdigest()[:16]
        runtime_meshes = OFFICIAL / "official_derived/resolved_meshes" / cache_key
        runtime_meshes.mkdir(parents=True, exist_ok=True)
        for asset_name, content in assets.items():
            destination = runtime_meshes / asset_name
            if not destination.is_file() or destination.read_bytes() != content:
                destination.write_bytes(content)
        spec.meshdir = str(runtime_meshes)
    else:
        spec = mujoco.MjSpec.from_file(str(entry.path))
    spec.compiler.fusestatic = False
    spec.compiler.balanceinertia = True
    if entry.mesh_dir is not None:
        for mesh in spec.meshes:
            mesh_path = Path(mesh.file)
            if not mesh_path.is_absolute():
                candidate = entry.mesh_dir / mesh_path
                if not candidate.is_file():
                    candidate = entry.mesh_dir / mesh_path.name
                if candidate.is_file():
                    mesh.file = str(candidate)
    if entry.root_body is not None:
        roots = [body for body in spec.bodies
                 if body.name != "world" and body.parent is not None and body.parent.name == "world"]
        keep = spec.body(entry.root_body)
        if keep is None:
            raise RuntimeError(f"missing canonical root {entry.root_body}")
        for root in roots:
            if root.name != entry.root_body:
                spec.detach_body(root)
        for geom in spec.geoms:
            if geom.parent is not None and geom.parent.name == "world":
                geom.pos[2] = -100.0
                geom.contype = 0
                geom.conaffinity = 0
                geom.rgba[3] = 0.0
    for body in spec.bodies:
        if body.name == "world" or body.first_joint() is None:
            continue
        if float(body.mass) <= 1e-5:
            body.mass = 0.01
        has_full_inertia = all(np.isfinite(float(value)) for value in body.fullinertia)
        if not has_full_inertia and min(float(value) for value in body.inertia) <= 1e-8:
            body.inertia[:] = [1e-4, 1e-4, 1e-4]
    for joint_name, joint_range in (entry.locked_joint_ranges or {}).items():
        joint = spec.joint(joint_name)
        if joint is None:
            raise RuntimeError(f"missing locked joint {joint_name}")
        joint.range[:] = joint_range
        joint.limited = True
    return spec


def qualify(name: str, entry: ModelEntry) -> dict[str, object]:
    spec = load_native_spec(entry)
    parent = spec.body(entry.tcp_parent)
    if parent is None:
        raise RuntimeError(f"missing TCP parent {entry.tcp_parent}")
    flange_parent = spec.body(entry.flange_parent or entry.tcp_parent)
    if flange_parent is None:
        raise RuntimeError(f"missing flange parent {entry.flange_parent}")
    flange_parent.add_site(name="strict_flange", pos=[0.0, 0.0, 0.0], size=[0.006, 0.0, 0.0])
    if not entry.path.resolve().is_relative_to(OFFICIAL.resolve()):
        raise RuntimeError(f"model source must be inside third_party: {entry.path}")
    parent.add_site(name="strict_tracking_tcp", pos=entry.tool_translation_m,
                    quat=entry.tool_quaternion_wxyz,
                    size=[0.008, 0.0, 0.0], rgba=[0.0, 0.8, 0.35, 1.0])
    model = spec.compile()
    roots = [body for body in range(1, model.nbody) if int(model.body_parentid[body]) == 0]
    if len(roots) != 1:
        raise RuntimeError(f"expected one root, found {roots}")
    if model.nmesh <= 0:
        raise RuntimeError("model has no mesh assets")
    addresses: list[int] = []
    limits: list[tuple[float, float]] = []
    for joint_name in entry.joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"missing active joint {joint_name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
        if bool(model.jnt_limited[joint_id]):
            limits.append(tuple(float(value) for value in model.jnt_range[joint_id]))
        else:
            limits.append((-np.pi, np.pi))
    if len(addresses) != entry.expected_dof:
        raise RuntimeError(f"expected {entry.expected_dof} active joints, found {len(addresses)}")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "strict_tracking_tcp")
    flange_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "strict_flange")
    if site_id < 0:
        raise RuntimeError("strict_tracking_tcp site was fused or lost")
    data = mujoco.MjData(model)
    rng = np.random.default_rng(20260807)
    tcp_positions = []
    flange_tcp_distance_errors = []
    flange_tcp_rotation_errors = []
    expected_tool_rotation = np.zeros(9)
    mujoco.mju_quat2Mat(expected_tool_rotation, np.asarray(entry.tool_quaternion_wxyz))
    expected_tool_rotation = expected_tool_rotation.reshape(3, 3)
    for sample in range(65):
        for address, (lower, upper) in zip(addresses, limits):
            fraction = 0.5 if sample == 0 else float(rng.uniform(0.1, 0.9))
            data.qpos[address] = lower + fraction * (upper - lower)
        mujoco.mj_forward(model, data)
        if not (np.isfinite(data.xpos).all() and np.isfinite(data.site_xpos).all()):
            raise RuntimeError(f"non-finite transform at sample {sample}")
        tcp_positions.append(data.site_xpos[site_id].copy())
        flange_tcp_distance_errors.append(abs(float(np.linalg.norm(data.site_xpos[site_id] - data.site_xpos[flange_site_id])) - entry.tool_offset_m))
        relative_rotation = data.site_xmat[flange_site_id].reshape(3, 3).T @ data.site_xmat[site_id].reshape(3, 3)
        rotation_error = expected_tool_rotation.T @ relative_rotation
        flange_tcp_rotation_errors.append(float(np.arccos(np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0))))
    tcp_positions = np.asarray(tcp_positions)
    moving_span = float(np.linalg.norm(tcp_positions.max(axis=0) - tcp_positions.min(axis=0)))
    if moving_span < 0.05:
        raise RuntimeError(f"TCP span too small: {moving_span}")
    return {
        "status": "pass",
        "source_model": str(entry.path),
        "source_format": entry.path.suffix.lower().lstrip("."),
        "root_body_count": len(roots),
        "root_body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, roots[0]),
        "active_dof": len(addresses),
        "active_joint_names": list(entry.joints),
        "tcp_parent": entry.tcp_parent,
        "tcp_authority": entry.tcp_authority,
        "joint_limit_authority": entry.joint_limit_authority,
        "tool_offset_m": entry.tool_offset_m,
        "flange_tcp_translation_m": list(entry.tool_translation_m),
        "flange_tcp_quaternion_wxyz": list(entry.tool_quaternion_wxyz),
        "native_geometry_dimensions": True,
        "experiment_geometry_policy": "pinned vendor-native dimensions; audited physical TCP transform attached",
        "native_sampled_maximum_flange_reach_m": sampled_maximum_tcp_reach_m(model, entry.joints, "strict_flange"),
        "tcp_convention": "T_base_tcp = T_base_flange @ T_flange_tcp; fixed tool offset is not morphology-scaled",
        "flange_tcp_translation_error_max_m": max(flange_tcp_distance_errors),
        "flange_tcp_rotation_error_max_rad": max(flange_tcp_rotation_errors),
        "bodies": int(model.nbody - 1),
        "joints_total": int(model.njnt),
        "mesh_count": int(model.nmesh),
        "geom_count": int(model.ngeom),
        "random_fk_samples": len(tcp_positions),
        "all_transforms_finite": True,
        "tcp_workspace_span_m": moving_span,
    }


def main() -> None:
    report: dict[str, object] = {"schema_version": 1, "robots": {}}
    for name, entry in MODELS.items():
        try:
            row = qualify(name, entry)
        except Exception as error:
            row = {"status": "fail", "source_model": str(entry.path), "error": f"{type(error).__name__}: {error}"}
        report["robots"][name] = row
        print(name, row["status"], row.get("error", ""), flush=True)
    passed = sum(row["status"] == "pass" for row in report["robots"].values())
    report["passed"] = passed
    report["total"] = len(MODELS)
    report["status"] = "pass" if passed == len(MODELS) else "fail"
    output = ROOT / "reports/single_arm/strict_urdf_model_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
