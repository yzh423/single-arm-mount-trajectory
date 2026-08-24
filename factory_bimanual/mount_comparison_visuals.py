"""Stable visual semantics for four-mount fixed-time comparisons."""
from __future__ import annotations

import numpy as np


STUDY_PANEL_ORDER = (
    "baseline",
    "upright_table",
    "horizontal_wall",
    "inverted",
)
MOUNT_LABELS = {
    "baseline": "BASELINE",
    "upright_table": "TABLE / UPRIGHT",
    "horizontal_wall": "WALL / HORIZONTAL",
    "inverted": "CEILING / INVERTED",
}
MOUNT_AXIS_LABELS = {
    "baseline": "configured installation",
    "upright_table": "mount axis +Z upward",
    "horizontal_wall": "mount axis horizontal",
    "inverted": "mount axis -Z downward",
}
MOUNT_COLORS = {
    "baseline": "#7B8794",
    "upright_table": "#2F80ED",
    "horizontal_wall": "#F2994A",
    "inverted": "#9B51E0",
}


def comparison_timeline(source_schedules, *, fps=30):
    schedules = [np.asarray(value, dtype=float) for value in source_schedules]
    if len(schedules) != 4:
        raise ValueError("four-panel comparison requires four source schedules")
    reference = schedules[0]
    if (reference.ndim != 1 or len(reference) < 2
            or not np.all(np.diff(reference) > 0)):
        raise ValueError("source schedule must be one strictly increasing vector")
    if any(not np.array_equal(value, reference) for value in schedules[1:]):
        raise ValueError("all panels must use the same source schedule")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    start = float(reference[0])
    duration = float(reference[-1] - start)
    return np.arange(0.0, duration + 1e-12, 1.0 / float(fps))


def source_frame_indices(source_time_s, video_time_s):
    source = np.asarray(source_time_s, dtype=float)
    timeline = np.asarray(video_time_s, dtype=float)
    if source.ndim != 1 or timeline.ndim != 1:
        raise ValueError("source and video timelines must be vectors")
    relative = source - source[0]
    if (len(source) < 2 or not np.all(np.diff(source) > 0)
            or np.any(timeline < 0)
            or (len(timeline) and timeline[-1] > relative[-1] + 1e-12)):
        raise ValueError("invalid fixed-time interpolation domain")
    return np.searchsorted(relative, timeline, side="right").clip(
        1, len(source)) - 1


def hex_to_bgr(value):
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError("color must use six hexadecimal digits")
    red, green, blue = (int(value[index:index + 2], 16)
                        for index in (0, 2, 4))
    return blue, green, red


def audited_mount_rank(row):
    """Prefer a fully safe result, then maximize synchronous coverage."""
    unsafe = (int(float(row.get("collision_frames", 0)))
              + int(float(row.get("edge_collision_frames", 0)))
              + int(float(row.get("topology_invalid_frames", 0))))
    return (
        unsafe > 0,
        -float(row["both_accept_coverage"]),
        unsafe,
        STUDY_PANEL_ORDER.index(row["mode"]),
    )


__all__ = [
    "MOUNT_AXIS_LABELS",
    "MOUNT_COLORS",
    "MOUNT_LABELS",
    "STUDY_PANEL_ORDER",
    "audited_mount_rank",
    "comparison_timeline",
    "hex_to_bgr",
    "source_frame_indices",
]
