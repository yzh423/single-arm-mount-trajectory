import json

import numpy as np
import pytest

from factory_bimanual.orientation_comparison_metrics import derive_orientation_metrics


def test_metrics_are_recomputed_from_row_aligned_arrays(tmp_path):
    path = tmp_path / "run.npz"
    np.savez_compressed(
        path,
        time_s=np.asarray([0., 1., 2., 4.]),
        followed=np.asarray([True, False, False, True]),
        q=np.asarray([[0., 0.], [.2, -.1], [.4, -.2], [.8, -.3]]),
        joint_lower=np.asarray([-1., -1.]), joint_upper=np.asarray([1., 1.]),
        position_error_left_m=np.asarray([0., .001, .002, .003]),
        position_error_right_m=np.asarray([.001, .002, .003, .004]),
        orientation_error_left_rad=np.asarray([0., .1, .2, .3]),
        orientation_error_right_rad=np.asarray([.1, .2, .3, .4]),
        singularity_left=np.asarray([.4, .3, .2, .1]),
        singularity_right=np.asarray([.5, .4, .3, .2]),
        state_collision=np.asarray([False, False, False, False]),
        edge_collision=np.asarray([False, True, False, False]),
    )
    metrics = derive_orientation_metrics(path)
    assert metrics["source_rows"] == 4
    assert metrics["coverage"] == .5
    assert metrics["longest_failure_frames"] == 2
    assert metrics["longest_failure_s"] == 3.
    assert metrics["position_error_mm"]["max"] == 4.
    assert metrics["minimum_joint_margin_rad"] == pytest.approx(.2)
    assert metrics["minimum_singularity_margin"] == .1
    assert metrics["edge_collision_frames"] == 1
    assert metrics["joint_velocity_rad_s"]["max"] == .2
