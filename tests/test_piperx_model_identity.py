from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import pytest
import yaml

from scripts.strict_urdf_model_audit import MODELS
from scripts.official_model_manifest import OFFICIAL_MODELS


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_URDF_SHA256 = "9c5433d94fdad29c050d00c1cdd86e02807f87b51abc429697129d2a6232f663"


def test_piperx_registry_uses_the_native_piper_x_model() -> None:
    entry = MODELS["piperx"]

    assert entry.path.resolve() == (
        ROOT / "third_party/official_robot_models/piperx/PiperX.urdf"
    ).resolve()
    assert entry.base_link == "base_link"
    assert entry.tcp_parent == "ee_frame"
    assert entry.tool_offset_m == pytest.approx(0.0)
    assert entry.tcp_authority == "model_ee_frame_115mm"


def test_piperx_asset_is_native_and_complete() -> None:
    entry = MODELS["piperx"]
    assert hashlib.sha256(entry.path.read_bytes()).hexdigest() == EXPECTED_URDF_SHA256
    root = ET.parse(entry.path).getroot()

    assert root.get("name") == "piper_x"
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    assert tuple(f"joint{i}" for i in range(1, 7)) == entry.joints
    assert all(joints[name].get("type") == "revolute" for name in entry.joints)
    assert joints["ee_frame_fixed"].find("parent").get("link") == "gripper_base"
    assert joints["ee_frame_fixed"].find("child").get("link") == "ee_frame"
    assert joints["ee_frame_fixed"].find("origin").get("xyz") == "0 0 0.115"

    mesh_names = {Path(mesh.get("filename")).name for mesh in root.findall(".//mesh")}
    assert mesh_names
    assert entry.mesh_dir == ROOT / "third_party/official_robot_models/piperx/meshes"
    assert all((entry.mesh_dir / name).is_file() for name in mesh_names)


def test_piperx_provenance_does_not_claim_the_older_piper_urdf() -> None:
    manifest = OFFICIAL_MODELS["piperx"]

    assert manifest["variant"].startswith("PiPER-X")
    assert "piper_ros" not in manifest.get("repository", "")
    assert manifest["source_artifact_sha256"] == (
        "9d5b0490df5d3469fa08fae355d5dbda3761af5dace5624d49eda56896b72ecb"
    )
    assert "agilexrobotics" in manifest["vendor_reference"].lower()


def test_piperx_provenance_pins_official_agx_arm_urdf_revision() -> None:
    manifest = OFFICIAL_MODELS["piperx"]

    assert manifest["repository"] == (
        "https://github.com/agilexrobotics/agx_arm_urdf"
    )
    assert manifest["revision"] == (
        "f6642ce0d7872c686f29c99e9e10cd23d1d49313"
    )
    assert manifest["source_subtree"] == "piper_x"


def test_piperx_yaml_registry_matches_runtime_model() -> None:
    registry = yaml.safe_load(
        (ROOT / "configs/robot_registry_13.yaml").read_text(encoding="utf-8")
    )["robots"]["piperx"]

    assert registry["source_model"] == (
        "third_party/official_robot_models/piperx/PiperX.urdf"
    )
    assert registry["base_link"] == "base_link"
    assert registry["flange_link"] == "ee_frame"
