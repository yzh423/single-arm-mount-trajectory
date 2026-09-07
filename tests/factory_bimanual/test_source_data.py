from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.source_data import load_factory_task


ROOT = Path(__file__).parents[2]


def _write_controller_update_fixture(path: Path, *, asynchronous: bool = False):
    import pandas as pd

    rows = []
    left_counters = [10, 10, 11, 12]
    right_counters = [20, 20, 21, 22]
    if asynchronous:
        right_counters[2] = 20
    for index, (left_counter, right_counter) in enumerate(
            zip(left_counters, right_counters)):
        update_index = [0, 0, 1, 2][index]
        row = {
            "t": [0.0, 0.01, 0.02, 0.03][index],
            "coordinate_frame": "vr_world",
            "left_tcp_valid": True,
            "right_tcp_valid": True,
            "left_frame_counter": left_counter,
            "right_frame_counter": right_counter,
            "left_receive_monotonic_s": [10.0, 10.0, 10.02, 10.04][index],
            "right_receive_monotonic_s": [10.0, 10.0, 10.02, 10.04][index],
        }
        for side in ("left", "right"):
            row.update({
                f"{side}_tcp_pos_x": 0.001 * update_index,
                f"{side}_tcp_pos_y": 0.0,
                f"{side}_tcp_pos_z": 0.2,
                f"{side}_tcp_quat_w": 1.0,
                f"{side}_tcp_quat_x": 0.0,
                f"{side}_tcp_quat_y": 0.0,
                f"{side}_tcp_quat_z": 0.0,
            })
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


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


def test_controller_update_timing_collapses_only_duplicate_paired_frames(tmp_path):
    path = tmp_path / "controller_updates.csv"
    _write_controller_update_fixture(path)

    task = load_factory_task(path, "fixture", timing_mode="controller_updates")

    np.testing.assert_array_equal(task.source_row_index, [0, 2, 3])
    np.testing.assert_allclose(task.time_s, [0.0, 0.02, 0.04])
    np.testing.assert_allclose(task.left_position_m[:, 0], [0.0, 0.001, 0.002])
    assert task.timing_source == "paired_controller_receive"
    assert task.source_poll_row_count == 4


def test_controller_update_timing_rejects_asynchronous_arm_updates(tmp_path):
    path = tmp_path / "asynchronous.csv"
    _write_controller_update_fixture(path, asynchronous=True)

    with pytest.raises(ValueError, match="asynchronous left/right controller frames"):
        load_factory_task(path, "fixture", timing_mode="controller_updates")
