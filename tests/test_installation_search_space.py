import numpy as np

from design_optimization.installation_search_space import first_version_bounds, mount_rotation_matrix, tabletop_mount_feasible
from design_optimization.search_policy import dimension_aware_candidate_budget


def test_installation_space_includes_pitch_yaw_and_roll():
    space = first_version_bounds(geometry_dimensions=7)
    np.testing.assert_allclose(space.lower[-6:], [-0.50, -0.55, 0.02, -90.0, -180.0, -180.0])
    np.testing.assert_allclose(space.upper[-6:], [0.50, 0.25, 0.65, 90.0, 180.0, 180.0])
    assert space.names[-6:] == ("base_x_m", "base_y_m", "base_z_m", "tilt_pitch_deg", "yaw_deg", "roll_deg")
    np.testing.assert_allclose(space.lower[-3:], (-90.0, -180.0, -180.0))
    np.testing.assert_allclose(space.upper[-3:], (90.0, 180.0, 180.0))


def test_mount_rotation_uses_yaw_pitch_roll_order():
    rotation = mount_rotation_matrix(tilt_pitch_deg=0.0, yaw_deg=90.0, roll_deg=0.0)
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_tabletop_feasibility_rejects_downward_or_off_table_support():
    assert tabletop_mount_feasible([0.0, -0.2, 0.4], mount_rotation_matrix(tilt_pitch_deg=30, yaw_deg=0, roll_deg=0))
    assert not tabletop_mount_feasible([0.0, -0.2, 0.4], mount_rotation_matrix(tilt_pitch_deg=0, yaw_deg=0, roll_deg=180))
    assert not tabletop_mount_feasible([1.0, 0.7, 0.4], mount_rotation_matrix(tilt_pitch_deg=-45, yaw_deg=0, roll_deg=0))


def test_thirteen_dimensional_search_uses_expanded_budget():
    assert dimension_aware_candidate_budget(13) >= 8192
