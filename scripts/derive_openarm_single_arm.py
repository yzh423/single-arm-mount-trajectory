"""Derive the official OpenArm v1 left arm with the gripper xacro TCP default."""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/official_robot_models/openarm/openarm_description-main/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
OUTPUT = ROOT / "third_party/official_robot_models/official_derived/openarm_description/openarm_v1_left_single.urdf"
BASE_LINK = "openarm_left_link0"
OFFICIAL_PARALLEL_GRIPPER_TCP_XYZ = "0 0 0.0835"


def derive(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    root = ET.parse(source).getroot()
    links = {element.get("name"): element for element in root.findall("link")}
    joints = list(root.findall("joint"))
    children: dict[str, list[tuple[str, ET.Element]]] = {}
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            children.setdefault(parent.get("link"), []).append((child.get("link"), joint))

    retained_links = {BASE_LINK}
    pending = [BASE_LINK]
    retained_joints: set[str] = set()
    while pending:
        parent = pending.pop()
        for child, joint in children.get(parent, []):
            retained_joints.add(joint.get("name"))
            if child not in retained_links:
                retained_links.add(child)
                pending.append(child)

    if "openarm_left_hand_tcp" not in retained_links:
        raise RuntimeError("official OpenArm left TCP is not below the single-arm base")
    if any(name.startswith("openarm_right_") for name in retained_links):
        raise RuntimeError("right-arm element leaked into left single-arm derivative")

    derived = ET.Element("robot", dict(root.attrib))
    derived.set("name", "openarm_v1_left_single_official_derived")
    for element in root:
        if element.tag == "link" and element.get("name") in retained_links:
            derived.append(copy.deepcopy(element))
        elif element.tag == "joint" and element.get("name") in retained_joints:
            derived.append(copy.deepcopy(element))
        elif element.tag not in {"link", "joint"}:
            derived.append(copy.deepcopy(element))

    # The checked-in vendor example was expanded with tcp_xyz=0, while the
    # authoritative parallel-gripper xacro declares 0 0 0.0835 as its default.
    # Recreate that official default when producing the single-arm derivative.
    tcp_joint = next(
        (joint for joint in derived.findall("joint")
         if joint.get("name") == "openarm_left_hand_tcp_joint"), None)
    if tcp_joint is None or tcp_joint.find("origin") is None:
        raise RuntimeError("official OpenArm TCP joint/origin is missing")
    tcp_joint.find("origin").set("xyz", OFFICIAL_PARALLEL_GRIPPER_TCP_XYZ)

    ET.indent(derived, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ET.tostring(derived, encoding="unicode") + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(derive())
