from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.source_data import load_factory_task


ROOT = Path(__file__).parents[2]


def test_screw_cap_retains_every_source_row_and_marks_missing_gripper():
    task = load_factory_task(
        ROOT / "data/factory/8-11/Screw_Cap/handheld_20260811_162854.csv",
        "screw_cap",
    )
    assert len(task.time_s) == 2524
    np.testing.assert_array_equal(task.source_row_index, np.arange(2524))
    assert np.all(np.diff(task.time_s) > 0)
    assert task.left_gripper_angle_rad is None
    assert task.right_gripper_angle_rad is None
    assert task.coordinate_frame == "vr_world"


def test_pour_retains_every_source_row_and_gripper_channels():
    task = load_factory_task(
        ROOT / "data/factory/8-12/PourRawMaterial/handheld_20260812_111542.csv",
        "pour_raw_material",
    )
    assert len(task.time_s) == 5942
    np.testing.assert_array_equal(task.source_row_index, np.arange(5942))
    assert task.left_gripper_angle_rad.shape == (5942,)
    assert task.right_gripper_angle_rad.shape == (5942,)
    for quaternion in (task.left_quaternion_wxyz, task.right_quaternion_wxyz):
        np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-8)


def test_fold_box_requires_explicit_jump_threshold_override():
    path = ROOT / "data/factory/8-11/Fold_Box/handheld_20260811_160754.csv"
    with pytest.raises(ValueError, match="translation jump"):
        load_factory_task(path, "fold_box")
    task = load_factory_task(path, "fold_box", max_translation_jump_m=.07)
    assert len(task.time_s) == 1964


def test_optional_repair_fills_only_invalid_nonfinite_pose_rows():
    path = ROOT / "data/factory/8-12/PutIntoBox/handheld_20260812_113145.csv"
    with pytest.raises(ValueError, match="non-finite"):
        load_factory_task(path, "put_into_box", max_translation_jump_m=.20)
    task = load_factory_task(
        path, "put_into_box", max_translation_jump_m=.20,
        repair_invalid_pose_rows=True)
    assert len(task.time_s) == 4804
    assert np.isfinite(task.left_position_m).all()
    assert np.isfinite(task.right_position_m).all()
    assert not task.left_valid[331:768].any()
    assert not task.right_valid[331:768].any()
