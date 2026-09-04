import numpy as np
import pytest

from factory_bimanual.mount_comparison_visuals import (
    MOUNT_COLORS,
    STUDY_PANEL_ORDER,
    comparison_timeline,
    source_frame_indices,
)
from scripts.plot_piperx_multitask_mount_results import (
    coverage_matrix,
    winner_counts,
)


def test_four_panel_timeline_is_exact_source_domain():
    source_time = np.asarray([0.0, 0.1, 0.25, 0.31])
    timeline = comparison_timeline([source_time] * 4, fps=30)

    assert timeline[0] == 0.0
    assert timeline[-1] <= source_time[-1]
    assert timeline[-1] + 1 / 30 > source_time[-1]


def test_comparison_rejects_different_source_schedules():
    with pytest.raises(ValueError, match="same source schedule"):
        comparison_timeline([
            np.asarray([0.0, 0.1]), np.asarray([0.0, 0.2]),
            np.asarray([0.0, 0.1]), np.asarray([0.0, 0.1])], fps=30)


def test_mount_visual_semantics_are_stable():
    assert STUDY_PANEL_ORDER == (
        "baseline", "upright_table", "horizontal_wall", "inverted")
    assert MOUNT_COLORS["upright_table"] == "#2F80ED"
    assert MOUNT_COLORS["horizontal_wall"] == "#F2994A"
    assert MOUNT_COLORS["inverted"] == "#9B51E0"


def test_source_frame_indices_never_look_ahead():
    source = np.asarray([0.0, 0.1, 0.25, 0.31])
    timeline = np.asarray([0.0, 0.05, 0.1, 0.2, 0.3])

    np.testing.assert_array_equal(
        source_frame_indices(source, timeline), [0, 0, 1, 1, 2])


def test_report_tables_use_fixed_mount_order_and_one_winner_per_trajectory():
    rows = [
        {"trajectory": "a", "mode": mode,
         "both_accept_coverage": value,
         "collision_frames": 0, "edge_collision_frames": 0}
        for mode, value in zip(STUDY_PANEL_ORDER, (0.5, 0.9, 0.8, 0.1))
    ] + [
        {"trajectory": "b", "mode": mode,
         "both_accept_coverage": value,
         "collision_frames": 0, "edge_collision_frames": 0}
        for mode, value in zip(STUDY_PANEL_ORDER, (0.7, 0.6, 0.95, 0.2))
    ]

    labels, matrix = coverage_matrix(rows)
    counts = winner_counts(rows)

    assert labels == ("a", "b")
    np.testing.assert_allclose(matrix, [[0.5, 0.9, 0.8, 0.1],
                                        [0.7, 0.6, 0.95, 0.2]])
    assert counts["upright_table"] == 1
    assert counts["horizontal_wall"] == 1
    assert sum(counts.values()) == 2


def test_winner_prefers_safe_mount_over_higher_coverage_collision():
    rows = [
        {"trajectory": "a", "mode": mode,
         "both_accept_coverage": (1.0 if mode == "baseline" else 0.8),
         "collision_frames": (1 if mode == "baseline" else 0),
         "edge_collision_frames": 0, "topology_invalid_frames": 0}
        for mode in STUDY_PANEL_ORDER
    ]

    counts = winner_counts(rows)

    assert counts["baseline"] == 0
    assert counts["upright_table"] == 1


def test_heatmap_matrix_represents_missing_metric_as_nan():
    rows = [
        {"trajectory": "a", "mode": mode,
         "maximum_accepted_position_error_mm": (
             None if mode == "baseline" else 0.5)}
        for mode in STUDY_PANEL_ORDER
    ]

    _labels, matrix = coverage_matrix(
        rows, "maximum_accepted_position_error_mm")

    assert np.isnan(matrix[0, 0])
    np.testing.assert_allclose(matrix[0, 1:], 0.5)
