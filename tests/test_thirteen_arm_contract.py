import json
from pathlib import Path

import pytest
import mujoco
import numpy as np

from design_optimization.robot_contract import (
    DEFAULT_FLANGE_TCP_TRANSLATION_M,
    PANDA_LOCKED_J3_RANGE_RAD,
)
from design_optimization.installation_search_space import mount_transform_from_record
from design_optimization.topology import load_templates
from design_optimization.urdf_chain import fk_flange
from scripts.audit_thirteen_robot_models import canonical_chain_from_entry
from scripts.strict_urdf_model_audit import MODELS, load_native_spec
from scripts.solve_strict_urdf_task_cache import build_model


ROOT = Path(__file__).resolve().parents[1]


def test_contract_preserves_official_tcp_frames_and_explicit_tool_offsets():
    assert DEFAULT_FLANGE_TCP_TRANSLATION_M == pytest.approx((0.0, 0.0, 0.130))
    assert all(MODELS[name].tool_offset_m == pytest.approx(0.0)
               for name in {"xarm6", "ur5", "kinova_gen3_lite", "franka_panda", "franka_panda_locked_j3"})
    assert MODELS["doosan"].tool_translation_m == pytest.approx((0.0, 0.0, 0.139))
    assert MODELS["openarm"].tool_translation_m == pytest.approx((0.0, 0.0, 0.0))
    assert MODELS["i2rt_yam"].tool_translation_m == pytest.approx((0.0, 0.0, -0.130))
    assert MODELS["piperx"].tool_translation_m == pytest.approx((0.0, 0.0, 0.0))
    assert MODELS["arx_x5"].tool_translation_m == pytest.approx((0.130, 0.0, 0.0))


def test_franka_and_openarm_use_vendor_defined_tcp_frames():
    for robot in ("franka_panda", "franka_panda_locked_j3"):
        assert MODELS[robot].tcp_parent == "panda_hand_tcp"
        assert MODELS[robot].tool_offset_m == pytest.approx(0.0)
    assert MODELS["openarm"].tcp_parent == "openarm_left_hand_tcp"
    assert MODELS["openarm"].tool_offset_m == pytest.approx(0.0)
    assert MODELS["openarm"].tcp_authority == "official_xacro_default_83p5mm"


def test_locked_panda_reuses_identical_official_geometry_and_only_constrains_j3():
    normal = MODELS["franka_panda"]
    locked = MODELS["franka_panda_locked_j3"]
    assert locked.path == normal.path
    assert locked.locked_joint_ranges == {"panda_joint3": PANDA_LOCKED_J3_RANGE_RAD}
    assert not normal.locked_joint_ranges


def test_search_templates_use_audited_tcp_lengths():
    templates = load_templates(ROOT / "reports/single_arm/model_audit.json")
    assert templates["doosan"].tool_length_m == pytest.approx(0.139)
    assert templates["arx_x5"].tool_length_m == pytest.approx(0.130)
    assert templates["i2rt_yam"].tool_length_m == pytest.approx(0.130)
    for name in {
        "xarm6", "ur5", "kinova_gen3_lite", "franka_panda",
        "franka_panda_locked_j3", "openarm", "piperx",
    }:
        assert templates[name].tool_length_m == pytest.approx(0.0)


def test_locked_panda_keeps_seven_joint_chain_with_tight_j3_range():
    row = json.loads((ROOT / "reports/single_arm/model_audit.json").read_text())["robots"][
        "franka_panda_locked_j3"
    ]
    assert len(row["joint_names"]) == 7
    assert len(row["axes"]) == 7
    assert row["q_min_rad"][2] == pytest.approx(PANDA_LOCKED_J3_RANGE_RAD[0])
    assert row["q_max_rad"][2] == pytest.approx(PANDA_LOCKED_J3_RANGE_RAD[1])


def test_all_authoritative_models_are_workspace_local_and_present():
    for name, entry in MODELS.items():
        assert entry.path.is_relative_to(ROOT), f"{name}: external model path {entry.path}"
        assert entry.path.is_file(), f"{name}: missing model {entry.path}"


def test_all_canonical_specs_compile_as_one_real_mesh_arm():
    for name, entry in MODELS.items():
        spec = load_native_spec(entry)
        model = spec.compile()
        roots = [body for body in range(1, model.nbody) if int(model.body_parentid[body]) == 0]
        assert len(roots) == 1, f"{name}: roots={roots}"
        assert model.nmesh > 0, f"{name}: real mesh missing"
        assert all(spec.joint(joint) is not None for joint in entry.joints), name


def test_strict_task_model_uses_official_tcp_frames():
    for robot in ("xarm6", "ur5", "kinova_gen3_lite"):
        model = build_model(robot, np.zeros(3), 0.0)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "strict_flange")
        tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "strict_tracking_tcp")
        assert np.linalg.norm(data.site_xpos[tcp] - data.site_xpos[flange]) == pytest.approx(0.0)


def test_saved_mount_record_applies_xyz_tilt_yaw_and_roll():
    record = {
        "base_xyz_m": [0.1, -0.2, 0.3],
        "tilt_deg": 20.0,
        "yaw_deg": -15.0,
        "roll_deg": 10.0,
    }
    transform = mount_transform_from_record(record)
    np.testing.assert_allclose(transform[:3, 3], record["base_xyz_m"])
    assert not np.allclose(transform[:3, :3], np.eye(3))


def test_search_fk_matches_authoritative_xarm6_mujoco_model():
    entry = MODELS["xarm6"]
    chain, model, flange_site_id = canonical_chain_from_entry("xarm6", entry)
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in entry.joints]
    addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    root = int(model.jnt_bodyid[joint_ids[0]])
    while int(model.body_parentid[root]) != 0:
        root = int(model.body_parentid[root])
    data.qpos[addresses] = 0.0
    mujoco.mj_forward(model, data)
    base_rotation = data.xmat[root].reshape(3, 3).copy()
    base_position = data.xpos[root].copy()
    rng = np.random.default_rng(20260808)
    for q in rng.uniform(chain.q_min, chain.q_max, size=(32, chain.dof)):
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        expected = np.eye(4)
        expected[:3, :3] = base_rotation.T @ data.site_xmat[flange_site_id].reshape(3, 3)
        expected[:3, 3] = base_rotation.T @ (data.site_xpos[flange_site_id] - base_position)
        actual = fk_flange(chain, q)
        assert np.linalg.norm(actual[:3, 3] - expected[:3, 3]) < 1e-7
        assert np.linalg.norm(actual[:3, :3] - expected[:3, :3]) < 1e-7
