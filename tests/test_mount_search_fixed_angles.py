import numpy as np
import scripts.run_thirteen_arm_dense_search as gpu_search

from scripts.search_strict_urdf_mount import (
    _compress_yaw_only_mounts,
    _expand_yaw_only_mounts,
)


def test_yaw_only_mount_adapter_removes_fixed_dimensions_without_mutating_inputs():
    lower = np.array([-0.5, -0.55, 0.02, -90.0, -180.0, -180.0])
    upper = np.array([0.5, 0.25, 0.65, 90.0, 180.0, 180.0])
    warm_starts = np.array([
        [0.1, -0.2, 0.3, 25.0, 40.0, -35.0],
        [-0.1, -0.1, 0.4, -15.0, -80.0, 20.0],
    ])
    original_lower = lower.copy()
    original_upper = upper.copy()
    original_warm_starts = warm_starts.copy()

    active_lower = _compress_yaw_only_mounts(lower)
    active_upper = _compress_yaw_only_mounts(upper)
    active_warm_starts = _compress_yaw_only_mounts(warm_starts)
    restored = _expand_yaw_only_mounts(active_warm_starts)

    assert active_lower.shape == (4,)
    assert active_upper.shape == (4,)
    assert np.all(active_lower < active_upper)
    np.testing.assert_array_equal(active_warm_starts, warm_starts[:, [0, 1, 2, 4]])
    np.testing.assert_array_equal(restored[:, [3, 5]], 0.0)
    np.testing.assert_array_equal(restored[:, [0, 1, 2, 4]], active_warm_starts)
    np.testing.assert_array_equal(lower, original_lower)
    np.testing.assert_array_equal(upper, original_upper)
    np.testing.assert_array_equal(warm_starts, original_warm_starts)


def test_gpu_coarse_search_exposes_only_xyz_yaw_coordinates():
    assert hasattr(gpu_search, "yaw_only_mount_space")
    names, lower, upper, incumbent = gpu_search.yaw_only_mount_space()
    assert names == ("base_x_m", "base_y_m", "base_z_m", "yaw_deg")
    assert lower.shape == upper.shape == incumbent.shape == (4,)
    assert np.all(lower < upper)


def test_gpu_coarse_search_accepts_one_explicit_episode():
    args = gpu_search.build_parser().parse_args([
        "--episode-artifact", "episodes/example.npz", "--candidates", "4096",
    ])
    assert str(args.episode_artifact) == "episodes\\example.npz" or str(args.episode_artifact) == "episodes/example.npz"
    assert args.candidates == 4096
