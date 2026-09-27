import numpy as np
import pytest
import mujoco

from scripts.render_piperx_portfolio_video import (
    _extract_tracks, _load_validated_shard, _trace_segments,
)
from scripts.run_piperx_controller_event_v4 import ROOT


def test_extract_tracks_keeps_raw_targets_separate_from_executed_tcp():
    arrays = {
        "source_time_s": np.array([0.0, 0.5, 1.0]),
        "left_target_position_m": np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        "right_target_position_m": np.array([[0., 1., 0.], [1., 1., 0.], [2., 1., 0.]]),
        "left_actual_tcp": np.array([[0., 0., 0., 1., 0., 0., 0.],
                                     [0., 0., 0., 1., 0., 0., 0.],
                                     [2., 0., 0., 1., 0., 0., 0.]]),
        "right_actual_tcp": np.array([[0., 1., 0., 1., 0., 0., 0.],
                                      [1., 1., 0., 1., 0., 0., 0.],
                                      [2., 1., 0., 1., 0., 0., 0.]]),
    }
    tracks = _extract_tracks(arrays)
    np.testing.assert_array_equal(tracks["left"][0][1], [1., 0., 0.])
    np.testing.assert_array_equal(tracks["left"][1][1], [0., 0., 0.])
    np.testing.assert_array_equal(tracks["right"][0][1], tracks["right"][1][1])


def test_trace_segments_keep_full_reference_but_stop_actual_at_current_time():
    times = np.array([0., 1., 2., 3., 4.])
    points = np.column_stack((times, np.zeros_like(times), np.zeros_like(times)))
    reference = _trace_segments(points, times, 1.5, full_path=True, dashed=False)
    actual = _trace_segments(points, times, 1.5, full_path=False, dashed=False,
                             current_point=np.array([1.5, 0., 0.]))
    np.testing.assert_allclose(reference[-1, 1], [4., 0., 0.])
    np.testing.assert_allclose(actual[-1, 1], [1.5, 0., 0.])
    assert np.max(actual[:, :, 0]) <= 1.5


def test_dashed_reference_omits_segments_without_changing_its_coordinates():
    times = np.arange(8., dtype=float)
    points = np.column_stack((times, np.zeros_like(times), np.zeros_like(times)))
    solid = _trace_segments(points, times, 2., full_path=True, dashed=False)
    dashed = _trace_segments(points, times, 2., full_path=True, dashed=True)
    assert len(dashed) < len(solid)
    assert len(dashed) > 0
    for segment in dashed:
        assert any(np.array_equal(segment, candidate) for candidate in solid)


def test_solid_trace_preserves_every_recorded_event_segment():
    times = np.arange(256., dtype=float)
    points = np.column_stack((times, np.sin(times), np.zeros_like(times)))
    segments = _trace_segments(points, times, 120., full_path=True, dashed=False)
    assert len(segments) == len(times) - 1
    np.testing.assert_array_equal(segments[127], points[127:129])


def test_extract_tracks_rejects_nonfinite_actual_tcp():
    arrays = {
        "source_time_s": np.array([0., 1.]),
        "left_target_position_m": np.zeros((2, 3)),
        "right_target_position_m": np.zeros((2, 3)),
        "left_actual_tcp": np.array([[0., 0., 0., 1., 0., 0., 0.],
                                     [np.nan, 0., 0., 1., 0., 0., 0.]]),
        "right_actual_tcp": np.zeros((2, 7)),
    }
    with pytest.raises(ValueError, match="finite"):
        _extract_tracks(arrays)


def test_published_actual_track_matches_mujoco_forward_kinematics():
    summary_path = ROOT / (
        "reports/piperx_controller_event_v4/shards/8-11/Fold_Box/161044/baseline/"
        "8-11_Fold_Box_161044_baseline_event_v4.summary.json")
    _, arrays, scene_path, _ = _load_validated_shard(summary_path)
    tracks = _extract_tracks(arrays)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    site_ids = {side: model.site(f"{side}_tcp").id for side in tracks}
    for event_index, qpos in enumerate(arrays["qpos"]):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for side in tracks:
            np.testing.assert_allclose(data.site_xpos[site_ids[side]],
                                       tracks[side][1][event_index], atol=1e-9)
