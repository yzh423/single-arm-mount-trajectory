from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig,
    MuJoCoCandidateGenerator,
)


def test_reference_candidate_minimizes_motion_inside_strict_pose_box():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/><worldbody><body>
      <joint name="j1" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="sphere" size=".01" mass=".1"/>
      <site name="tcp" pos=".2 0 0"/>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[0] = .004
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    target_position = data.site_xpos[site].copy()
    target_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(target_quaternion, data.site_xmat[site])
    generator = MuJoCoCandidateGenerator(
        model, data, SimpleNamespace(),
        name_map={"left": {"joints": ("j1",), "site": "tcp"}},
        config=CandidateGeneratorConfig(global_seed_count=0),
    )

    candidate = generator.generate_reference_candidate(
        "left", target_position, target_quaternion,
        reference_q=np.array([0.0]))

    assert candidate is not None
    assert candidate.q[0] == pytest.approx(0.0, abs=1e-7)
    assert candidate.position_error_m <= .001
    assert candidate.orientation_error_rad <= np.deg2rad(1.5)
