from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig, MuJoCoCandidateGenerator,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.robot_contracts import BimanualRobotContract
from factory_bimanual.scene_builder import build_same_model_scene


def test_rotation_error_is_expressed_in_world_jacobian_frame():
    from factory_bimanual.mujoco_candidate_generator import _world_rotation_error
    current = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
    target = np.array([.5, .5, .5, .5])
    assert _world_rotation_error(target, current) == pytest.approx(
        [0, np.pi / 2, 0], abs=1e-7
    )


def test_public_warm_start_candidate_uses_half_degree_strict_box(tmp_path):
    contract = ROBOT_CONTRACTS["piperx"]
    xml = tmp_path / "piperx_warm.xml"
    build_same_model_scene(contract, .8, xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = {"left": {
        "joints": contract.prefixed_joint_names("left"),
        "site": "left_tcp",
    }}
    joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in names["left"]["joints"]
    ])
    qids = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")
    target_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(target_quaternion, data.site_xmat[site])
    generator = MuJoCoCandidateGenerator(
        model, data, contract, name_map=names,
        config=CandidateGeneratorConfig(
            position_tolerance_m=.001,
            orientation_tolerance_rad=np.deg2rad(.5),
            global_seed_count=0,
        ),
    )

    result = generator.generate_warm_start_candidate(
        "left",
        data.site_xpos[site].copy(),
        target_quaternion,
        reference_q=data.qpos[qids].copy(),
    )

    assert result is not None
    assert result.branch_index == -1
    assert result.position_error_m <= .001
    assert result.orientation_error_rad <= np.deg2rad(.5)


@pytest.mark.parametrize("robot", tuple(ROBOT_CONTRACTS))
def test_neutral_tcp_produces_strict_candidate_for_all_scenes(tmp_path, robot):
    contract = ROBOT_CONTRACTS[robot]
    xml = tmp_path / f"{robot}.xml"
    build_same_model_scene(contract, .8, xml)
    model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    task_values = {}
    names = {side: {"joints": contract.prefixed_joint_names(side), "site": f"{side}_tcp"}
             for side in ("left", "right")}
    for side in ("left", "right"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        task_values[f"{side}_position_m"] = np.asarray([data.site_xpos[sid].copy()])
        quat = np.empty(4); mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
        task_values[f"{side}_quaternion_wxyz"] = np.asarray([quat])
    task = SimpleNamespace(**task_values)
    generator = MuJoCoCandidateGenerator(model, data, contract, name_map=names)

    for side in ("left", "right"):
        candidates = generator(model, contract, task, 0, side)
        assert candidates
        assert candidates[0].position_error_m <= .001
        assert candidates[0].orientation_error_rad <= np.deg2rad(1.5)
        assert candidates[0].actual_tcp.shape == (7,)


def test_seed_bank_and_branch_order_are_deterministic(tmp_path):
    contract = ROBOT_CONTRACTS["xarm6"]
    xml = tmp_path / "x.xml"; build_same_model_scene(contract, .8, xml)
    model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")
    quat = np.empty(4); mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
    task = SimpleNamespace(left_position_m=np.asarray([data.site_xpos[sid].copy()]),
                           left_quaternion_wxyz=np.asarray([quat]))
    names = {"left": {"joints": contract.prefixed_joint_names("left"), "site": "left_tcp"}}
    generator = MuJoCoCandidateGenerator(model, data, contract, name_map=names)
    first = generator(model, contract, task, 0, "left")
    second = generator(model, contract, task, 0, "left")
    assert [c.branch_index for c in first] == [c.branch_index for c in second]
    assert all(np.array_equal(a.q, b.q) for a, b in zip(first, second))


def test_candidate_generator_keeps_distinct_wide_limited_joint_hard_stop_solutions():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/><worldbody><body>
      <joint name="j1" type="hinge" axis="0 0 1" range="-6.283185 6.283185"/>
      <geom type="sphere" size=".01" mass=".1"/><site name="tcp" pos=".2 0 0"/>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    quaternion = np.empty(4); mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
    task = SimpleNamespace(
        left_position_m=np.asarray([data.site_xpos[site].copy()]),
        left_quaternion_wxyz=np.asarray([quaternion]),
    )
    generator = MuJoCoCandidateGenerator(
        model, data, SimpleNamespace(),
        name_map={"left": {"joints": ("j1",), "site": "tcp"}},
        config=CandidateGeneratorConfig(dedup_rad=np.deg2rad(1.0)),
    )
    candidates = generator(model, None, task, 0, "left")
    # A finite [-2pi, 2pi] hinge is not periodic topology: configurations near
    # the two hard stops must remain distinct even if their TCP poses coincide.
    assert len(candidates) == 3
    assert candidates[0].q[0] == pytest.approx(0.0)
    assert min(item.q[0] for item in candidates) < -6.27
    assert max(item.q[0] for item in candidates) > 6.27


def test_position_priority_target_generation_reports_unmatched_orientation():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/><worldbody><body>
      <joint name="j1" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="sphere" size=".01" mass=".1"/><site name="tcp" pos=".2 0 0"/>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    target_quaternion = np.array([0., 1., 0., 0.])  # impossible x half-turn
    generator = MuJoCoCandidateGenerator(
        model, data, SimpleNamespace(),
        name_map={"left": {"joints": ("j1",), "site": "tcp"}},
        config=CandidateGeneratorConfig(
            orientation_weight=0.0,
            orientation_tolerance_rad=np.pi,
            global_seed_count=0,
        ),
    )

    candidates = generator.generate_target(
        "left", data.site_xpos[site].copy(), target_quaternion)

    assert candidates
    assert candidates[0].position_error_m <= .001
    assert candidates[0].orientation_error_rad > np.deg2rad(90.)


def test_global_seed_bank_does_not_repeat_three_bit_joint_signs(tmp_path):
    contract = ROBOT_CONTRACTS["xarm6"]
    xml = tmp_path / "x.xml"; build_same_model_scene(contract, .8, xml)
    model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
    names = {"left": {"joints": contract.prefixed_joint_names("left"),
                       "site": "left_tcp"}}
    generator = MuJoCoCandidateGenerator(
        model, data, contract, name_map=names,
        config=CandidateGeneratorConfig(global_seed_count=12))
    seeds = np.asarray(generator._seeds("left")[-12:])
    midpoint = np.mean(model.jnt_range[[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names["left"]["joints"]]], axis=1)
    signs = seeds > midpoint
    assert np.any(signs[:, 0] != signs[:, 3])
    assert np.any(signs[:, 1] != signs[:, 4])
    assert np.any(signs[:, 2] != signs[:, 5])


def test_forced_seed_refresh_includes_stratified_branches_off_schedule(
        tmp_path):
    contract = ROBOT_CONTRACTS["piperx"]
    xml = tmp_path / "piperx.xml"
    build_same_model_scene(contract, .8, xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    names = {"left": {
        "joints": contract.prefixed_joint_names("left"),
        "site": "left_tcp",
    }}
    generator = MuJoCoCandidateGenerator(
        model, data, contract, name_map=names,
        config=CandidateGeneratorConfig(
            global_seed_count=0, stratified_seed_enabled=True,
            stratified_refresh_interval=100,
        ))
    generator._rolling["left"] = [np.zeros(6)]
    generator._seed_calls["left"] = 1

    ordinary = generator._seeds("left")
    forced = generator._seeds("left", force_stratified=True)

    assert len(forced) > len(ordinary)
    assert generator._last_used_stratified["left"]


def test_candidate_search_disables_contacts_but_restores_collision_masks(tmp_path, monkeypatch):
    """IK is kinematic; contact generation belongs to the later edge filter."""
    contract = ROBOT_CONTRACTS["xarm6"]
    xml = tmp_path / "x.xml"; build_same_model_scene(contract, .8, xml)
    model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")
    quat = np.empty(4); mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
    task = SimpleNamespace(left_position_m=np.asarray([data.site_xpos[sid].copy()]),
                           left_quaternion_wxyz=np.asarray([quat]))
    names = {"left": {"joints": contract.prefixed_joint_names("left"),
                       "site": "left_tcp"}}
    generator = MuJoCoCandidateGenerator(model, data, contract, name_map=names,
                                          config=CandidateGeneratorConfig(global_seed_count=0))
    original_contype = model.geom_contype.copy()
    original_conaffinity = model.geom_conaffinity.copy()

    def observe_masks(*_args, **_kwargs):
        assert not np.any(model.geom_contype)
        assert not np.any(model.geom_conaffinity)
        return None

    monkeypatch.setattr(generator, "_solve", observe_masks)
    assert generator(model, contract, task, 0, "left") == []
    assert np.array_equal(model.geom_contype, original_contype)
    assert np.array_equal(model.geom_conaffinity, original_conaffinity)


def test_constrained_fallback_recovers_real_tolerance_boundary_false_negative(
        tmp_path):
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    from scripts.strict_urdf_model_audit import MODELS
    legacy = BimanualRobotContract(
        "piperx_scaled_legacy",
        MODELS["piperx"].path,
        tuple(f"joint{i}" for i in range(1, 7)),
        "gripper_base", "base_link",
        ((-2.6179938, 2.6179938), (0.0, 3.1415926),
         (-2.9670597, 0.0), (-1.553343, 1.553343),
         (-1.553343, 1.553343), (-2.0943951, 2.0943951)),
        (0.0, 0.0, 0.13),
    )
    # Preserve the exact historical scaled scene which exposed the numerical
    # boundary miss.  New scenes intentionally reject unregistered legacy
    # assets and use the official native-scale Piper X instead.
    xml = (root / "reports/factory_bimanual/seal_bag_dual_piperx"
           / "piperx_selected_mount.scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    task = SimpleNamespace(
        left_position_m=np.asarray([[
            0.06312342461090609, -0.24703320367035672, 1.227163873,
        ]]),
        left_quaternion_wxyz=np.asarray([[
            0.47757521388755036, 0.17889655332932233,
            0.7437187524778983, -0.43220406696359104,
        ]]),
    )
    names = {"left": {
        "joints": legacy.prefixed_joint_names("left"), "site": "left_tcp",
    }}
    common = dict(max_iterations=240, global_seed_count=64,
                  maximum_candidates=8)
    ordinary = MuJoCoCandidateGenerator(
        model, data, legacy, name_map=names,
        config=CandidateGeneratorConfig(
            **common, constrained_fallback_enabled=False),
    )
    recovered = MuJoCoCandidateGenerator(
        model, data, legacy, name_map=names,
        config=CandidateGeneratorConfig(
            **common, constrained_fallback_enabled=True),
    )

    assert ordinary(model, legacy, task, 0, "left") == []
    first = recovered(model, legacy, task, 0, "left")
    recovered.reset()
    second = recovered(model, legacy, task, 0, "left")
    assert first
    assert first[0].position_error_m <= .001
    assert first[0].orientation_error_rad <= np.deg2rad(1.5)
    assert [item.q.tolist() for item in first] == [
        item.q.tolist() for item in second
    ]
