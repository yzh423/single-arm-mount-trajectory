from pathlib import Path

from design_optimization.robot_registry import load_robot_registry
from design_optimization.urdf_chain import (
    load_native_chain,
    load_urdf_chain,
    sampled_maximum_reach,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "doosan",
    "xarm6",
    "ur5",
    "kinova_gen3_lite",
    "arx_x5",
    "big_yam",
    "franka_panda",
    "franka_panda_locked_j3",
    "i2rt_yam",
    "nero",
    "openarm",
    "piperx",
    "willow",
}


def test_registry_is_complete_and_source_models_exist():
    registry = load_robot_registry(ROOT / "configs/robot_registry_13.yaml")
    assert set(registry) == EXPECTED
    assert all(spec.source_model.exists() for spec in registry.values())


def test_registry_preserves_native_dof_and_locked_franka_contract():
    registry = load_robot_registry(ROOT / "configs/robot_registry_13.yaml")
    assert registry["franka_panda"].active_dof == 7
    assert registry["franka_panda_locked_j3"].active_dof == 6
    locked = registry["franka_panda_locked_j3"]
    assert locked.locked_joints_rad["panda_joint3"] == (-0.0001, 0.0001)


def test_registry_rejects_duplicate_active_joint_names(tmp_path):
    config = tmp_path / "registry.yaml"
    config.write_text(
        """schema_version: 1
robots:
  broken:
    source_model: model.urdf
    model_format: urdf
    base_link: base
    flange_link: flange
    active_joints: [joint1, joint1]
""",
        encoding="utf-8",
    )
    try:
        load_robot_registry(config)
    except ValueError as error:
        assert "duplicate active joint" in str(error)
    else:
        raise AssertionError("duplicate active joints must be rejected")


def test_urdf_chain_extracts_native_six_and_seven_joint_models():
    registry = load_robot_registry(ROOT / "configs/robot_registry_13.yaml")
    xarm = load_urdf_chain(registry["xarm6"])
    franka = load_urdf_chain(registry["franka_panda"])
    assert xarm.axes.shape == (6, 3)
    assert xarm.deltas.shape == (6, 3)
    assert franka.axes.shape == (7, 3)
    assert franka.deltas.shape == (7, 3)
    assert (xarm.q_min < xarm.q_max).all()
    assert (franka.q_min < franka.q_max).all()


def test_locked_franka_applies_epsilon_j3_range():
    registry = load_robot_registry(ROOT / "configs/robot_registry_13.yaml")
    chain = load_urdf_chain(registry["franka_panda_locked_j3"])
    index = chain.joint_names.index("panda_joint3")
    assert chain.q_min[index] == -0.0001
    assert chain.q_max[index] == 0.0001


def test_native_chain_reach_is_measured_without_mutating_geometry():
    registry = load_robot_registry(ROOT / "configs/robot_registry_13.yaml")
    source = load_urdf_chain(registry["xarm6"])
    before = source.deltas.copy()
    maximum = sampled_maximum_reach(source, sample_count=4096)
    assert maximum > 0.75
    assert (source.deltas == before).all()


def test_mjcf_backends_extract_ur5_and_kinova_bare_flange_chains():
    registry = load_robot_registry(ROOT / "configs/robot_registry_13.yaml")
    ur5 = load_native_chain(registry["ur5"])
    kinova = load_native_chain(registry["kinova_gen3_lite"])
    assert ur5.axes.shape == (6, 3)
    assert kinova.axes.shape == (6, 3)
    assert registry["ur5"].source_tool_length_m == 0.0
    assert registry["kinova_gen3_lite"].source_tool_length_m == 0.0


def test_official_model_audit_certifies_current_ten_arm_native_geometry():
    from scripts.audit_thirteen_robot_models import audit_registry

    payload = audit_registry(
        ROOT / "configs/robot_registry_13.yaml", sample_count=256
    )
    assert set(payload["robots"]) == set(__import__("scripts.strict_urdf_model_audit", fromlist=["MODELS"]).MODELS)
    assert payload["status"] == "pass"
    assert payload["geometry_policy"] == "official_vendor_native_dimensions"
