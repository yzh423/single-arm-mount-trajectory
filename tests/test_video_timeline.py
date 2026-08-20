import numpy as np

from scripts.video_timeline import showcase_timeline, validation_timeline


def test_validation_timeline_is_exact_identity():
    source = np.array([0.0, 0.1, 0.25, 0.5])
    np.testing.assert_array_equal(validation_timeline(source), source)


def test_showcase_compresses_only_long_stationary_runs():
    source = np.arange(8, dtype=float) * 0.1
    moving = np.array([True, False, False, False, True, False, True])
    result = showcase_timeline(source, moving)
    display = np.asarray(result["display_time_s"])
    assert np.all(np.diff(display) > 0)
    assert np.isclose(display[4] - display[1], 0.1)
    assert np.isclose(display[6] - display[5], 0.1)
    assert result["dwell_threshold_s"] == 0.2
    assert result["compressed_dwell_s"] == 0.1
    assert result["display_only"] is True
