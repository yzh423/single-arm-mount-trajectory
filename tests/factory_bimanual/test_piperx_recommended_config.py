from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.piperx_recommended import (
    DEFAULT_CONFIG_PATH,
    load_recommended_config,
    world_mount_for_family,
)
from factory_bimanual.task_family import TaskFamily, family_from_path


def test_family_keeps_same_task_on_different_dates_separate(tmp_path):
    root = tmp_path / "factory"
    first = root / "8-11" / "Fold_Box" / "a.csv"
    second = root / "8-12" / "Fold_Box" / "b.csv"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("time\n0\n", encoding="utf-8")
    second.write_text("time\n0\n", encoding="utf-8")

    family_a = family_from_path(first, root)
    family_b = family_from_path(second, root)

    assert family_a.key == "8-11/Fold_Box"
    assert family_b.key == "8-12/Fold_Box"
    assert family_a != family_b


def test_family_path_must_have_date_task_and_file(tmp_path):
    root = tmp_path / "factory"
    root.mkdir()
    invalid = root / "episode.csv"
    invalid.write_text("time\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="date"):
        family_from_path(invalid, root)


def test_config_has_all_pdf_piperx_task_families():
    config = load_recommended_config(DEFAULT_CONFIG_PATH)
    assert set(config.mounts) == {
        "8-11/AluminumFoilPouch_BoxPackaging",
        "8-11/Fold_Box",
        "8-11/Screw_Cap",
        "8-11/Seal_Bag",
        "8-11/WaterSoluble_AluminumPouch",
        "8-12/Bag_BoxPacking",
        "8-12/Fold_Box",
        "8-12/InsertIntoBottle",
        "8-12/InsertSmallPackageIntoMachine",
        "8-12/PackIntoBox",
        "8-12/PourRawMaterial",
        "8-12/PutIntoBox",
    }


def test_fold_box_pdf_base_is_registered_into_world():
    config = load_recommended_config(DEFAULT_CONFIG_PATH)
    mount = world_mount_for_family(
        config,
        TaskFamily("8-11", "Fold_Box"),
        np.eye(3),
        np.array([-0.3995700068, 0.0200368514, 1.444496408]),
    )

    assert mount.mode == "upright_table"
    assert mount.source_take == "161044"
    assert np.allclose(
        mount.left_xyz_m,
        [-0.2495700068, 0.3200368514, 0.761496408],
    )
    assert np.allclose(
        mount.right_xyz_m,
        [-0.3495700068, -0.2799631486, 0.761496408],
    )
    assert np.isclose(mount.shared_base_z_m, 0.761496408)
    assert np.isclose(mount.base_distance_m, np.sqrt(0.37))


def test_world_mount_rejects_mismatched_registered_heights():
    config = load_recommended_config(DEFAULT_CONFIG_PATH)
    rotation = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.2, 0.0, 1.0)))
    with pytest.raises(ValueError, match="shared height"):
        world_mount_for_family(
            config, TaskFamily("8-11", "Fold_Box"), rotation, np.zeros(3)
        )


def test_strict_thresholds_and_funnel_are_not_relaxed():
    config = load_recommended_config(DEFAULT_CONFIG_PATH)
    assert config.accept.position_tolerance_m == 0.001
    assert np.isclose(
        config.accept.orientation_tolerance_rad,
        np.deg2rad(0.5),
        rtol=0.0,
        atol=1e-15,
    )
    assert config.accept.branch_guard_rad == 0.30
    assert config.anchor_restarts == 40
    assert config.funnel.anchor_keep == 150
    assert config.funnel.probe_keep == 12
    assert config.funnel.full_keep == 3
    assert config.funnel.probe_stride_frames == 60


def test_unknown_family_is_explicit_error():
    config = load_recommended_config(DEFAULT_CONFIG_PATH)
    with pytest.raises(KeyError, match="8-13/Unknown"):
        world_mount_for_family(
            config, TaskFamily("8-13", "Unknown"), np.eye(3), np.zeros(3)
        )
