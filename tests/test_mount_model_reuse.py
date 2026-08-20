import mujoco
import numpy as np

from scripts.solve_strict_urdf_task_cache import (
    MountModelTemplate,
    build_model,
    build_mount_model_template,
)


def named_id(model, kind, name):
    return mujoco.mj_name2id(model, kind, name)


def assert_mount_equivalent(robot, mount):
    legacy = build_model(robot, mount[:3], mount[3], mount[4], mount[5])
    template = build_mount_model_template(robot)
    reused, reused_data = template.apply(mount[:3], mount[3], mount[4], mount[5])
    legacy_data = mujoco.MjData(legacy)
    mujoco.mj_forward(legacy, legacy_data)

    legacy_root = next(body for body in range(1, legacy.nbody) if legacy.body_parentid[body] == 0)
    reused_root = next(body for body in range(1, reused.nbody) if reused.body_parentid[body] == 0)
    np.testing.assert_allclose(reused.body_pos[reused_root], legacy.body_pos[legacy_root], atol=1e-12)
    np.testing.assert_allclose(reused.body_quat[reused_root], legacy.body_quat[legacy_root], atol=1e-12)
    for name in ("strict_pedestal", "strict_mount_adapter"):
        left = named_id(legacy, mujoco.mjtObj.mjOBJ_GEOM, name)
        right = named_id(reused, mujoco.mjtObj.mjOBJ_GEOM, name)
        np.testing.assert_allclose(reused.geom_pos[right], legacy.geom_pos[left], atol=1e-12)
        np.testing.assert_allclose(reused.geom_size[right], legacy.geom_size[left], atol=1e-12)
        reused_axis = reused_data.geom_xmat[right].reshape(3, 3)[:, 2]
        legacy_axis = legacy_data.geom_xmat[left].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(
            np.outer(reused_axis, reused_axis), np.outer(legacy_axis, legacy_axis), atol=1e-12,
        )


def test_mount_template_matches_legacy_xarm6_at_multiple_mounts():
    assert MountModelTemplate is not None
    for mount in (
        np.array([0.0, -0.15, 0.335, 0.0, 0.0, 0.0]),
        np.array([0.1, -0.2, 0.4, 30.0, 20.0, 10.0]),
        np.array([-0.2, 0.1, 0.2, -45.0, 80.0, -20.0]),
    ):
        assert_mount_equivalent("xarm6", mount)
