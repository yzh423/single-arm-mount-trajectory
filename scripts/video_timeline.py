"""Evidence-preserving validation and display-only showcase timelines."""
from __future__ import annotations

import numpy as np


def _source_time(source_time_s: np.ndarray) -> np.ndarray:
    source = np.asarray(source_time_s, dtype=float)
    if source.ndim != 1 or len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError("source_time_s must be a strictly increasing 1-D timeline")
    return source


def validation_timeline(source_time_s: np.ndarray) -> np.ndarray:
    """Return an exact copy: validation playback is timing evidence."""
    return _source_time(source_time_s).copy()


def showcase_timeline(
    source_time_s: np.ndarray,
    moving_intervals: np.ndarray,
    *,
    dwell_threshold_s: float = 0.20,
    compressed_dwell_s: float = 0.10,
) -> dict[str, object]:
    """Compress long consecutive stationary intervals without altering source data."""
    source = _source_time(source_time_s)
    moving = np.asarray(moving_intervals, dtype=bool)
    if moving.shape != (len(source) - 1,):
        raise ValueError("moving_intervals must describe every source interval")
    if dwell_threshold_s <= 0 or compressed_dwell_s <= 0:
        raise ValueError("dwell durations must be positive")
    source_dt = np.diff(source)
    display_dt = source_dt.copy()
    runs: list[dict[str, object]] = []
    start = 0
    while start < len(moving):
        if moving[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(moving) and not moving[stop]:
            stop += 1
        duration = float(source_dt[start:stop].sum())
        compressed = duration > dwell_threshold_s
        if compressed:
            display_dt[start:stop] *= compressed_dwell_s / duration
        runs.append({
            "source_interval_start": start,
            "source_interval_stop": stop,
            "source_duration_s": duration,
            "display_duration_s": float(display_dt[start:stop].sum()),
            "compressed": compressed,
        })
        start = stop
    display = np.r_[0.0, np.cumsum(display_dt)] + source[0]
    return {
        "mode": "showcase_dwell_compressed",
        "display_only": True,
        "dwell_threshold_s": float(dwell_threshold_s),
        "compressed_dwell_s": float(compressed_dwell_s),
        "source_time_s": source.tolist(),
        "display_time_s": display.tolist(),
        "stationary_runs": runs,
    }
