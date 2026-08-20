from pathlib import Path
import json

from factory_bimanual.robot_contracts import ROBOT_CONTRACTS, get_robot_contract


ROOT = Path(__file__).resolve().parents[2]


def test_supported_robot_contracts_are_exact_and_native_dof():
    assert tuple(ROBOT_CONTRACTS) == ("xarm6", "franka_panda", "i2rt_yam", "piperx", "ur5")
    assert {name: contract.dof_per_arm for name, contract in ROBOT_CONTRACTS.items()} == {
        "xarm6": 6,
        "franka_panda": 7,
        "i2rt_yam": 6,
        "piperx": 6,
        "ur5": 6,
    }


def test_contract_assets_exist_and_are_not_locked_panda():
    expected_root = ROOT / "third_party/official_robot_models"
    for name, contract in ROBOT_CONTRACTS.items():
        assert contract.source_urdf.is_file()
        assert contract.source_urdf.is_relative_to(expected_root)
        assert len(contract.arm_joint_names) == contract.dof_per_arm
        assert len(set(contract.arm_joint_names)) == contract.dof_per_arm

    panda = get_robot_contract("franka_panda")
    assert panda.source_urdf == (
        ROOT / "third_party/official_robot_models/official_derived/"
        "franka_description/panda_official.urdf"
    )
    assert "locked" not in str(panda.source_urdf).lower()


def test_learning_reference_tcp_contracts_are_explicit():
    panda = get_robot_contract("franka_panda")
    yam = get_robot_contract("i2rt_yam")
    assert panda.tcp_link_name == "panda_hand_tcp"
    assert panda.tcp_offset_m == (0.0, 0.0, 0.0)
    assert yam.tcp_link_name == "gripper"
    assert yam.tcp_offset_m == (0.0, 0.0, -0.13)
    piper = get_robot_contract("piperx")
    assert piper.source_urdf == ROOT / "third_party/official_robot_models/piperx/PiperX.urdf"
    assert piper.tcp_link_name == "ee_frame"
    assert piper.tcp_offset_m == (0.0, 0.0, 0.0)


def test_piperx_controller_profile_uses_native_velocity_limit():
    payload = json.loads((ROOT / "factory_bimanual/controller_profiles.json").read_text())
    profile = payload["profiles"]["piperx"]
    assert profile["dof"] == 6
    assert profile["joint_velocity_rad_s"] == [3.0] * 6


def test_generated_names_are_unique_and_side_prefixed():
    for contract in ROBOT_CONTRACTS.values():
        left = contract.prefixed_joint_names("left")
        right = contract.prefixed_joint_names("right")
        assert all(name.startswith("left_") for name in left)
        assert all(name.startswith("right_") for name in right)
        assert set(left).isdisjoint(right)


def test_contracts_retain_authoritative_native_joint_limits_in_radians():
    for contract in ROBOT_CONTRACTS.values():
        assert len(contract.joint_limits_rad) == contract.dof_per_arm
        assert all(lower < upper for lower, upper in contract.joint_limits_rad)
