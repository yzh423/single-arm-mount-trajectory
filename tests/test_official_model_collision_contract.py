from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.solve_strict_urdf_task_cache import build_model, collision_flags
from scripts.strict_urdf_model_audit import MODELS, load_native_spec


ROOT = Path(__file__).resolve().parents[1]


def _mesh_aabb(model: mujoco.MjModel, geom_name: str) -> tuple[np.ndarray, np.ndarray]:
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh = int(model.geom_dataid[geom])
    start = int(model.mesh_vertadr[mesh])
    vertices = model.mesh_vert[start:start + int(model.mesh_vertnum[mesh])]
    mesh_rotation = np.empty(9)
    mujoco.mju_quat2Mat(mesh_rotation, model.mesh_quat[mesh])
    geom_rotation = np.empty(9)
    mujoco.mju_quat2Mat(geom_rotation, model.geom_quat[geom])
    vertices = vertices @ mesh_rotation.reshape(3, 3).T + model.mesh_pos[mesh]
    vertices = vertices @ geom_rotation.reshape(3, 3).T + model.geom_pos[geom]
    return vertices.min(axis=0), vertices.max(axis=0)


def test_doosan_uses_the_required_139mm_tcp_from_tool0() -> None:
    entry = MODELS["doosan"]
    assert entry.tcp_parent == "tool0"
    assert entry.tool_translation_m == pytest.approx((0.0, 0.0, 0.139))
    assert entry.tcp_authority == "specified_139mm"


def test_openarm_uses_the_official_xacro_default_83p5mm_grasp_tcp() -> None:
    entry = MODELS["openarm"]
    assert entry.tcp_parent == "openarm_left_hand_tcp"
    assert entry.tool_translation_m == pytest.approx((0.0, 0.0, 0.0))
    assert entry.tcp_authority == "official_xacro_default_83p5mm"
    spec = load_native_spec(entry)
    tcp = spec.body("openarm_left_hand_tcp")
    assert tcp.pos == pytest.approx((0.0, 0.0, 0.0835))


def test_ur5_visual_meshes_are_from_the_ur5_variant() -> None:
    model = load_native_spec(MODELS["ur5"]).compile()
    for link in ("upper_arm_link", "forearm_link"):
        visual_low, visual_high = _mesh_aabb(model, f"{link}_visual_0")
        collision_low, collision_high = _mesh_aabb(model, f"{link}_collision_0")
        visual_center = 0.5 * (visual_low + visual_high)
        collision_center = 0.5 * (collision_low + collision_high)
        assert np.linalg.norm(visual_center - collision_center) < 0.10, link


def test_panda_renders_official_visual_meshes_not_collision_proxies() -> None:
    model = load_native_spec(MODELS["franka_panda"]).compile()
    visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "panda_link1_visual_0")
    collision = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "panda_link1_collision_0")
    visual_mesh = int(model.geom_dataid[visual])
    collision_mesh = int(model.geom_dataid[collision])
    assert int(model.mesh_vertnum[visual_mesh]) != int(model.mesh_vertnum[collision_mesh])


def test_panda_official_self_collision_layers_do_not_collide_at_neutral() -> None:
    model = build_model("franka_panda", np.asarray((0.0, -0.2, 0.4)), 0.0)
    q = np.asarray((0.0, 0.0, 0.0, -np.pi / 2, 0.0, 1.868, 0.0))
    mount, self_collision = collision_flags(model, MODELS["franka_panda"].joints, q[None, :])
    assert not bool(mount[0])
    assert not bool(self_collision[0])


def test_openarm_official_wrist_overlap_is_not_reported_as_self_collision() -> None:
    model = build_model("openarm", np.asarray((0.0, -0.2, 0.4)), 0.0)
    q = np.asarray((-1.047, -1.571, 0.0, 1.222, 0.0, 0.0, 0.0))
    mount, self_collision = collision_flags(model, MODELS["openarm"].joints, q[None, :])
    assert not bool(mount[0])
    assert not bool(self_collision[0])


def test_first_actuated_link_collision_with_mount_adapter_is_rejected() -> None:
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="strict_mount_adapter" type="cylinder" size=".08 .04" pos="0 0 .3"/>
      <body name="fixed_root">
        <body name="first_moving" pos="0 0 .3">
          <joint name="j1"/><geom name="moving_geom" type="sphere" size=".1"/>
        </body>
      </body>
    </worldbody></mujoco>""")
    mount, _ = collision_flags(model, ("j1",), np.zeros((1, 1)))
    assert bool(mount[0])
