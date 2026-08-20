from pathlib import Path

import numpy as np

from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.source_data import load_factory_task


ROOT = Path(__file__).parents[2]


def test_registration_preserves_time_and_bimanual_distances():
    task = load_factory_task(
        ROOT / "data/factory/8-11/Screw_Cap/handheld_20260811_162854.csv",
        "screw_cap")
    angle = np.deg2rad(35.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    registered = register_task(
        task, RigidTaskRegistration(rotation, np.array([0.2, -0.3, 1.0])))
    np.testing.assert_array_equal(registered.time_s, task.time_s)
    source_distance = np.linalg.norm(task.left_position_m-task.right_position_m, axis=1)
    registered_distance = np.linalg.norm(
        registered.left_position_m-registered.right_position_m, axis=1)
    np.testing.assert_allclose(registered_distance, source_distance, atol=1e-12)
    np.testing.assert_allclose(registered.registration.inverse().matrix
                               @ registered.registration.matrix, np.eye(4), atol=1e-12)


def test_registration_rejects_scale_and_reflection():
    for invalid in (np.diag([2.0, 1.0, 1.0]), np.diag([-1.0, 1.0, 1.0])):
        try:
            RigidTaskRegistration(invalid, np.zeros(3))
        except ValueError as exc:
            assert "rotation" in str(exc)
        else:
            raise AssertionError("non-rigid transform was accepted")
