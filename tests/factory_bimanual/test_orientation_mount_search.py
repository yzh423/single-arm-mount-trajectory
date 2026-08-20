import dataclasses
import hashlib
import json

import numpy as np
from pathlib import Path

from factory_bimanual.orientation_mount_search import (
    OrientationSearchConfig, evenly_spaced, generate_mounts, mount_fingerprint,
)
from scripts.search_fold_box_piperx_paired_mount import _valid_mount
from factory_bimanual.factory_task_catalog import build_factory_task_catalog
from factory_bimanual.per_task_mount_search import load_registered_representative


@dataclasses.dataclass
class _Task:
    left_position_m: np.ndarray
    right_position_m: np.ndarray


def _task():
    return _Task(
        np.asarray([[-.1, .1, 1.], [.1, .2, 1.1]]),
        np.asarray([[.1, -.1, 1.], [.2, -.2, 1.1]]))


def test_modes_use_identical_budget_bounds_and_shared_z():
    config = OrientationSearchConfig(maximum_candidates=15)
    groups = {mode: generate_mounts(mode, _task(), config)
              for mode in config.modes}
    assert {len(records) for records in groups.values()} == {15}
    for records in groups.values():
        assert all(record["base_z_m"]["left"] == record["base_z_m"]["right"]
                   for record in records)
        assert all(np.linalg.norm(np.subtract(record["xy"]["left"],
                                               record["xy"]["right"])) >= .60
                   for record in records)
    stripped = lambda rows: [
        {"xy": row["xy"], "yaw": row["yaw"]} for row in rows]
    assert stripped(groups[config.modes[0]]) == stripped(groups[config.modes[1]])


def test_physical_installation_heights_are_mode_specific():
    config = OrientationSearchConfig(maximum_candidates=15)
    task = _task()
    upright = generate_mounts("upright_table", task, config)
    wall = generate_mounts("horizontal_wall", task, config)
    inverted = generate_mounts("inverted", task, config)
    assert {m["shared_base_z_m"] for m in upright} == {.81}
    expected_vertical_midpoint = .5 * (
        min(task.left_position_m[:, 2].min(), task.right_position_m[:, 2].min())
        + max(task.left_position_m[:, 2].max(), task.right_position_m[:, 2].max()))
    assert {m["shared_base_z_m"] for m in wall} == {
        expected_vertical_midpoint}
    assert {m["shared_base_z_m"] for m in inverted} == {1.41}


def test_generation_and_fingerprints_are_deterministic():
    config = OrientationSearchConfig(maximum_candidates=12)
    first = generate_mounts("inverted", _task(), config)
    second = generate_mounts("inverted", _task(), config)
    assert first == second
    assert [mount_fingerprint(row) for row in first] == [
        mount_fingerprint(row) for row in second]
    legacy = hashlib.sha256(json.dumps(
        first[0], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    assert mount_fingerprint(first[0]) != legacy


def test_orientation_experiment_accepts_common_elevated_z_region():
    mount = generate_mounts(
        "horizontal_wall", _task(),
        OrientationSearchConfig(maximum_candidates=5))[4]
    assert mount["shared_base_z_m"] == 1.05
    assert _valid_mount(mount)


def test_orientation_mount_rejects_close_bases():
    mount = generate_mounts(
        "upright_table", _task(),
        OrientationSearchConfig(maximum_candidates=5))[0]
    mount["xy"] = {"left": [0.0, 0.0], "right": [.59, 0.0]}
    assert not _valid_mount(mount)


def test_horizontal_forward_grid_uses_task_vertical_midpoint_and_safe_spacing():
    task = _task()
    mounts = generate_mounts(
        "horizontal_forward", task,
        OrientationSearchConfig(
            modes=("horizontal_forward",), maximum_candidates=36))
    assert len(mounts) == 36
    assert {m["shared_base_z_m"] for m in mounts} == {1.05}
    assert all(np.linalg.norm(np.subtract(m["xy"]["left"],
                                           m["xy"]["right"])) >= .60
               for m in mounts)
    task_center_y = np.vstack((task.left_position_m,
                               task.right_position_m))[:, 1].mean()
    assert all(np.mean([m["xy"]["left"][1], m["xy"]["right"][1]])
               <= task_center_y - .15 for m in mounts)


def test_every_factory_representative_has_full_forward_coarse_budget():
    root = Path(__file__).resolve().parents[2] / "data" / "factory"
    for representative in build_factory_task_catalog(root).representatives:
        task = load_registered_representative(representative)
        mounts = generate_mounts(
            "horizontal_forward", task,
            OrientationSearchConfig(
                modes=("horizontal_forward",), maximum_candidates=36))
        assert len(mounts) == 36, representative.task_name


def test_evenly_spaced_reduction_keeps_endpoints_and_budget():
    values = list(range(36))
    selected = evenly_spaced(values, 12)
    assert len(selected) == 12
    assert selected[0] == 0 and selected[-1] == 35
    assert selected == evenly_spaced(values, 12)
