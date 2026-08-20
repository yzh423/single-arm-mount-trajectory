import numpy as np

from scripts.strict_trajectory_sources import RelativeTrajectory, resample_trajectory


def test_resample_trajectory_caps_frames_and_preserves_endpoints():
    count = 1000
    trajectory = RelativeTrajectory(
        time_s=np.linspace(0.0, 10.0, count),
        position_m=np.column_stack((np.linspace(0.0, 1.0, count), np.zeros((count, 2)))),
        quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        hand="right",
        source="fixture",
    )

    sampled = resample_trajectory(trajectory, maximum_frames=300)

    assert len(sampled.time_s) == 300
    np.testing.assert_array_equal(sampled.time_s[[0, -1]], [0.0, 10.0])
    np.testing.assert_array_equal(sampled.position_m[[0, -1], 0], [0.0, 1.0])
