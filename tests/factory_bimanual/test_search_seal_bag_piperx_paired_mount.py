from scripts.search_seal_bag_piperx_paired_mount import (
    CSV, _registered_seal_bag_task,
)


def test_official_piperx_mount_search_uses_complete_seal_bag_task():
    task = _registered_seal_bag_task()
    assert CSV.name == "handheld_20260811_161504.csv"
    assert len(task.time_s) > 1000
    assert len(task.left_position_m) == len(task.time_s)
    assert len(task.right_position_m) == len(task.time_s)
