"""Common bright table scene for all canonical single-arm renders."""
from __future__ import annotations

import mujoco
import numpy as np


def _geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[start:start + count]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        return float((vertices @ rotation.T + data.geom_xpos[geom_id]).min(axis=0)[2])
    return float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id])


def align_robot_to_pedestal(spec: mujoco.MjSpec, joint_names, *, pedestal_top_z: float = 0.036) -> float:
    """Place the base visual surface on the pedestal without root-origin guessing."""
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    first_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[0])
    first_body = int(model.jnt_bodyid[first_joint])
    base_bodies = {first_body}
    cursor = first_body
    while int(model.body_parentid[cursor]) != 0:
        cursor = int(model.body_parentid[cursor])
        base_bodies.add(cursor)
    visual = [g for g in range(model.ngeom)
              if int(model.geom_bodyid[g]) in base_bodies and int(model.geom_group[g]) == 1]
    if not visual:
        visual = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in base_bodies]
    if not visual:
        raise ValueError("robot base has no renderable geometry")
    lowest = min(_geom_min_z(model, data, geom) for geom in visual)
    offset = pedestal_top_z - lowest
    roots = [body for body in spec.bodies
             if body.name != "world" and body.parent is not None and body.parent.name == "world"]
    if len(roots) != 1:
        raise ValueError(f"expected one robot root, got {[body.name for body in roots]}")
    roots[0].pos[2] += offset
    return float(offset)


def base_mount_depth(spec: mujoco.MjSpec, joint_names) -> float:
    """Distance from the root origin to the lowest vendor base visual surface."""
    model = spec.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    first_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[0])
    first_body = int(model.jnt_bodyid[first_joint]); base_bodies = {first_body}; cursor = first_body
    while int(model.body_parentid[cursor]) != 0:
        cursor = int(model.body_parentid[cursor]); base_bodies.add(cursor)
    visual = [g for g in range(model.ngeom)
              if int(model.geom_bodyid[g]) in base_bodies and int(model.geom_group[g]) == 1]
    if not visual:
        visual = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in base_bodies]
    root = cursor
    lowest = min(_geom_min_z(model, data, geom) for geom in visual)
    return max(0.0, float(data.xpos[root, 2] - lowest))


def decorate_high_quality_scene(spec: mujoco.MjSpec) -> None:
    """Add renderer-only geometry without changing native robot dimensions."""
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 900
    spec.visual.global_.azimuth = 130
    spec.visual.global_.elevation = -22
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient[:] = [0.38, 0.38, 0.38]
    spec.visual.headlight.diffuse[:] = [0.58, 0.58, 0.58]
    spec.visual.headlight.specular[:] = [0.12, 0.12, 0.12]
    spec.visual.rgba.haze[:] = [0.72, 0.78, 0.86, 1.0]
    spec.add_texture(name="render_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.82, 0.86, 0.92], rgb2=[0.48, 0.56, 0.68],
                     width=512, height=3072)

    world = spec.worldbody
    # Group 2 is visible in the default renderer while remaining distinct
    # from robot collision (0) and robot visual (1) geometry.
    common = dict(contype=0, conaffinity=0, group=2)
    world.add_geom(name="render_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   pos=[0, 0, -0.76], size=[3.0, 3.0, 0.05], rgba=[0.69, 0.72, 0.76, 1], **common)
    world.add_geom(name="render_tabletop", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0, -0.20, -0.045], size=[0.78, 0.62, 0.045],
                   rgba=[0.78, 0.75, 0.68, 1], **common)
    for x in (-0.66, 0.66):
        for y in (-0.68, 0.28):
            world.add_geom(name=f"render_leg_{x}_{y}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[x, y, -0.40], size=[0.045, 0.045, 0.35],
                           rgba=[0.25, 0.27, 0.30, 1], **common)
    world.add_geom(name="render_pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[0, 0, 0.018], size=[0.105, 0.018, 0],
                   rgba=[0.27, 0.29, 0.32, 1], **common)
    world.add_light(name="render_key", pos=[1.1, -1.2, 1.8], dir=[-0.5, 0.45, -1],
                    directional=True, diffuse=[0.85, 0.85, 0.85], specular=[0.2, 0.2, 0.2], castshadow=True)
    world.add_light(name="render_fill", pos=[-1.4, -0.2, 1.3], dir=[0.7, 0.1, -1],
                    directional=True, diffuse=[0.48, 0.50, 0.54], specular=[0.05, 0.05, 0.05], castshadow=False)
    world.add_light(name="render_rim", pos=[0.2, 1.3, 1.6], dir=[-0.1, -0.7, -1],
                    directional=True, diffuse=[0.38, 0.42, 0.48], specular=[0.08, 0.08, 0.08], castshadow=False)
