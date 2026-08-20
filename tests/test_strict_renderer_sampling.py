import numpy as np

from scripts.render_strict_single_arm_task import interpolate_joint_positions, path_render_indices


def test_path_render_indices_preserve_endpoints_and_bound_geometry():
    indices = path_render_indices(5000, maximum_points=400)
    assert len(indices) <= 400
    assert indices[0] == 0 and indices[-1] == 4999
    assert np.all(np.diff(indices) > 0)


def test_joint_interpolation_does_not_spin_through_periodic_boundary():
    lower = np.asarray((np.pi - 0.1, 0.0))
    upper = np.asarray((-np.pi + 0.1, 1.0))
    periodic = np.asarray((True, False))
    halfway = interpolate_joint_positions(lower, upper, 0.5, periodic)
    assert abs(abs(halfway[0]) - np.pi) < 1e-10
    assert halfway[1] == 0.5
