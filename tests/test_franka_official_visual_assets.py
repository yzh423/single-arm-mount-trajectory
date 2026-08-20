from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.strict_urdf_model_audit import MODELS


ROOT = Path(__file__).resolve().parents[1]


def test_both_franka_variants_use_self_contained_official_dae_visuals():
    for robot in ("franka_panda", "franka_panda_locked_j3"):
        entry = MODELS[robot]
        tree = ET.parse(entry.path)
        visual_meshes = [mesh.attrib["filename"] for mesh in tree.findall(".//visual/geometry/mesh")]
        assert visual_meshes
        assert all(name.startswith("package://franka_description/") for name in visual_meshes)
        package_root = entry.package_roots["franka_description"]
        assert all((package_root / name.removeprefix("package://franka_description/")).is_file()
                   for name in visual_meshes)
