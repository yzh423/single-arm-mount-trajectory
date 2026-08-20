"""Pinned manufacturer/model-source authority for native robot experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party" / "official_robot_models"

OFFICIAL_MODELS = {
    "doosan": {"repository": "https://github.com/DoosanRobotics/doosan-robot2", "revision": "ec9242546ec6202835900dbcd8498e2daabfa6a6", "variant": "M0609"},
    "xarm6": {"repository": "https://github.com/xArm-Developer/xarm_ros2", "revision": "62936f7ea1846a85f7350de2c4c18f39e6d19715", "variant": "xArm6"},
    "ur5": {"repository": "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description", "revision": "18e6f603b3ebc2ec479fecb62d6be544b15755e9", "variant": "UR5 (CB3)"},
    "kinova_gen3_lite": {"repository": "https://github.com/Kinovarobotics/ros2_kortex", "revision": "c50057a02fb64e854b2759261994f43173bec703", "variant": "Gen3 Lite + 2F gripper"},
    "arx_x5": {"repository": "https://github.com/ARXroboticsX/ARX_Model", "revision": "af6fe43c873008a85bce6195c0f2160f1a1c14ce", "variant": "X5A"},
    "franka_panda": {"repository": "https://github.com/frankaemika/franka_ros", "revision": "ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98", "variant": "Panda + Franka hand"},
    "franka_panda_locked_j3": {"repository": "https://github.com/frankaemika/franka_ros", "revision": "ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98", "variant": "Panda + Franka hand; J3 limit override only"},
    "i2rt_yam": {"repository": "https://github.com/i2rt-robotics/i2rt", "revision": "1276f63d640eb45c226efd3dc08430b810372e94", "variant": "YAM v1 linear 4310 D405"},
    "openarm": {"repository": "https://github.com/enactic/openarm_description", "revision": "1fba2cbc05001f05b4514120b70130b4ac06f409", "variant": "OpenArm v1 left single arm + parallel gripper (deterministic subtree derivative)"},
    "piperx": {
        "repository": "https://github.com/agilexrobotics/agx_arm_urdf",
        "revision": "f6642ce0d7872c686f29c99e9e10cd23d1d49313",
        "source_subtree": "piper_x",
        "variant": "PiPER-X native 6-DoF model + parallel gripper",
        "source_artifact_sha256": "9d5b0490df5d3469fa08fae355d5dbda3761af5dace5624d49eda56896b72ecb",
        "vendor_reference": "https://www.linkedin.com/posts/agilexrobotics_piperx-agilexrobotics-highprecision-activity-7392038315203510272-2EIO",
    },
}

MODEL_SOURCES = {
    **OFFICIAL_MODELS,
    "big_yam": {
        "repository": "https://github.com/i2rt-robotics/i2rt",
        "revision": "1276f63d640eb45c226efd3dc08430b810372e94",
        "variant": "Big YAM v1 arm and gripper",
    },
    "nero": {
        "repository": "workspace-vendored-snapshot",
        "revision": "9ddd8bc3f614b1dea55d7fc65eb5d75ae121c089dbf8d4f033e11ec08ad1ad4",
        "variant": "Nero seven-axis arm and gripper",
        "source_status": "pinned_vendor_snapshot_without_public_revision",
    },
    "willow": {
        "repository": "workspace-vendored-snapshot",
        "revision": "bedb1139ea9a298ad5663a5b5c903a8ca2a0f49d1508a48c54d49e3b6f2b5e74",
        "variant": "Ragtime Willow six-axis arm; MuJoCo mesh-split derivative",
        "source_status": "pinned_vendor_snapshot_without_public_revision",
    },
}
