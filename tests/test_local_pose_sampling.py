import numpy as np

from design_optimization.local_pose_sampling import trim_episode_edges


def test_trim_episode_edges_removes_first_and_last_second():
    time = np.arange(0.0, 4.01, 0.1)
    keep = trim_episode_edges(time, seconds=1.0)
    assert time[keep][0] >= 1.0
    assert time[keep][-1] <= 3.0
    assert not keep[0] and not keep[-1]


def test_default_trim_removes_first_and_last_point_fifteen_seconds():
    time = np.arange(0.0, 2.01, 0.1)
    keep = trim_episode_edges(time)
    assert time[keep][0] >= 0.15
    assert time[keep][-1] <= 1.85


def test_trim_episode_edges_rejects_episode_without_interior():
    time = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    keep = trim_episode_edges(time, seconds=1.0)
    assert keep.sum() < 2
