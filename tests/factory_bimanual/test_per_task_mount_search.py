from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.factory_task_catalog import build_factory_task_catalog
from factory_bimanual.per_task_mount_search import (
    PerTaskSearchConfig, candidate_fingerprint,
    load_registered_representative, prefix_registered_task,
    rank_full_finalist, select_safe_layout, summarize_quality_arrays,
)


def test_config_locks_hierarchical_budgets_and_modes():
    config = PerTaskSearchConfig()
    assert (config.coarse_budget, config.dense_budget,
            config.local_budget, config.finalist_budget) == (36, 6, 12, 4)
    assert config.modes == (
        "upright_table", "horizontal_forward", "inverted")
    assert config.minimum_separation_m == .60


def test_fingerprint_changes_with_source_mode_rules_settings_and_mount():
    config = PerTaskSearchConfig()
    mount = {"xy": {"left": [-.4, 0], "right": [.4, 0]},
             "yaw": {"left": 0, "right": 180}, "shared_base_z_m": .81}
    base = candidate_fingerprint("abc", "upright_table", mount, config)
    assert base != candidate_fingerprint("def", "upright_table", mount, config)
    assert base != candidate_fingerprint("abc", "inverted", mount, config)
    assert base != candidate_fingerprint("abc", "upright_table",
                                         mount | {"shared_base_z_m": .82}, config)
    assert base != candidate_fingerprint(
        "abc", "upright_table", mount,
        replace(config, minimum_separation_m=.61))


def test_full_finalist_ranking_and_selection_require_zero_collisions():
    common = {"synchronous_strict_coverage": .8, "longest_failure_frames": 4,
              "mean_pair_pose_error": .002, "p95_pair_pose_error": .004,
              "minimum_joint_limit_margin_rad": .2,
              "p10_pair_singularity_margin": .1, "base_distance_m": .7,
              "audited_source_rows": 100, "source_row_count": 100,
              "pair_edge_collision_frames": 0}
    colliding = common | {"pair_collision_frames": 1,
                          "synchronous_strict_coverage": .99}
    safe_low = common | {"pair_collision_frames": 0,
                         "synchronous_strict_coverage": .7}
    safe_high = common | {"pair_collision_frames": 0}
    assert rank_full_finalist(safe_high) < rank_full_finalist(safe_low)
    assert select_safe_layout([colliding, safe_low, safe_high]) is safe_high
    with pytest.raises(RuntimeError, match="no fully audited collision-free layout"):
        select_safe_layout([colliding])


def test_quality_summary_uses_real_error_and_margin_arrays():
    metrics = summarize_quality_arrays(
        position_error={"left": np.asarray([.001, .003]),
                        "right": np.asarray([.002, .004])},
        orientation_error={"left": np.asarray([.01, .03]),
                           "right": np.asarray([.02, .04])},
        joint_margins={"left": np.asarray([.2, .1]),
                       "right": np.asarray([.3, .15])},
        singularity_margins={"left": np.asarray([.05, .03]),
                             "right": np.asarray([.08, .04])})
    assert metrics["minimum_joint_limit_margin_rad"] == pytest.approx(.1)
    assert metrics["p10_pair_singularity_margin"] > 0
    assert metrics["p95_pair_pose_error"] >= metrics["mean_pair_pose_error"]


def test_real_representative_loader_preserves_every_source_row():
    root = Path(__file__).resolve().parents[2] / "data" / "factory"
    representative = next(
        item for item in build_factory_task_catalog(root).representatives
        if item.task_name == "Seal_Bag")
    task = load_registered_representative(representative)
    assert len(task.time_s) == representative.source_rows
    assert np.array_equal(task.source_row_index,
                          np.arange(representative.source_rows))
    points = np.vstack((task.left_position_m, task.right_position_m))
    np.testing.assert_allclose(points[:, :2].mean(axis=0), 0., atol=1e-12)
    assert points[:, 2].min() == pytest.approx(.90)
    prefix = prefix_registered_task(task, 2)
    assert type(prefix) is type(task)
    assert len(prefix.time_s) == 2
    assert np.array_equal(prefix.source_row_index, np.asarray([0, 1]))
