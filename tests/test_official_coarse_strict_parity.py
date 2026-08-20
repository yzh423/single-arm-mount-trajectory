import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from design_optimization.kinematics import assemble_from_deltas, fk_tcp
from scripts.run_thirteen_arm_dense_search import load_official_search_templates
from scripts.solve_strict_urdf_task_cache import build_model
from scripts.strict_urdf_model_audit import MODELS


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_proxy_tcp_fk_matches_strict_official_mujoco():
    templates = load_official_search_templates(device="cpu", dtype=torch.float64)
    rng = np.random.default_rng(20260813)
    for name, template in templates.items():
        model = build_model(name, np.zeros(3), 0.0)
        data = mujoco.MjData(model)
        entry = MODELS[name]
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                     for joint in entry.joints]
        addresses = [int(model.jnt_qposadr[joint]) for joint in joint_ids]
        root = int(model.jnt_bodyid[joint_ids[0]])
        while int(model.body_parentid[root]) != 0:
            root = int(model.body_parentid[root])
        tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "strict_tracking_tcp")
        design = assemble_from_deltas(template, template.deltas)
        samples = rng.uniform(template.q_min.numpy(), template.q_max.numpy(), size=(16, template.dof))
        actual = fk_tcp(design, torch.tensor(samples, dtype=torch.float64))[0].numpy()
        for q, transform in zip(samples, actual):
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            base_rotation = data.xmat[root].reshape(3, 3)
            base_position = data.xpos[root]
            expected_position = base_rotation.T @ (data.site_xpos[tcp_id] - base_position)
            expected_rotation = base_rotation.T @ data.site_xmat[tcp_id].reshape(3, 3)
            assert np.linalg.norm(transform[:3, 3] - expected_position) < 1e-4, name
            relative = transform[:3, :3].T @ expected_rotation
            angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            assert np.degrees(angle) < 0.01, name


def test_collision_profiles_are_derived_from_current_official_models():
    payload = json.loads(
        (ROOT / "reports/single_arm/collision_proxy_profiles.json").read_text(encoding="utf-8")
    )
    assert payload["geometry_policy"] == "strict_official_model_registry"
    for name, entry in MODELS.items():
        row = payload["robots"][name]
        assert Path(row["source_model"]).resolve() == entry.path.resolve()
        assert row["tcp_authority"] == entry.tcp_authority
