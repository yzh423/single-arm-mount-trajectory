import mujoco

from scripts.render_strict_single_arm_task import configure_visual_only_render


def test_configure_visual_only_render_hides_collision_geoms_but_keeps_scene():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="strict_table" type="box" size="1 1 .1"/>
      <body><geom name="link_visual_0" type="sphere" size=".1" group="1"/>
            <geom name="link_collision_0" type="capsule" size=".1 .2"/></body>
    </worldbody></mujoco>""")

    option = configure_visual_only_render(model)

    collision = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link_collision_0")
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strict_table")
    assert model.geom_group[collision] == 3
    assert model.geom_group[table] == 0
    assert option.geomgroup[3] == 0


def test_configure_visual_only_render_hides_collision_suffix_without_index():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body>
      <geom name="openarm_link_visual" type="sphere" size=".1" contype="0" conaffinity="0"/>
      <geom name="openarm_link_collision" type="sphere" size=".1"/>
    </body></worldbody></mujoco>""")

    option = configure_visual_only_render(model)
    collision = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "openarm_link_collision")

    assert model.geom_group[collision] == 3
    assert option.geomgroup[3] == 0
