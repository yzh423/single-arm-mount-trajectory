"""Whole-trajectory paired mount search for official native-scale Piper X."""
from __future__ import annotations

import numpy as np

from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.source_data import load_factory_task
from scripts import search_fold_box_piperx_paired_mount as search


ROOT = search.ROOT
CSV = ROOT / "data/factory/8-11/Seal_Bag/handheld_20260811_161504.csv"
search.OUT = (ROOT / "reports/factory_bimanual/seal_bag_dual_piperx"
              / "official_native_scale_paired_mount_search.json")
search.WORK = ROOT / ".tmp/piperx_seal_bag_official_paired_mount"


def _registered_seal_bag_task():
    source = load_factory_task(CSV, "seal_bag")
    points = np.vstack((source.left_position_m, source.right_position_m))
    translation = np.array([
        -points[:, 0].mean(),
        -points[:, 1].mean(),
        search.TABLE_HEIGHT_M + .15 - points[:, 2].min(),
    ])
    return register_task(
        source, RigidTaskRegistration(np.eye(3), translation))


if __name__ == "__main__":
    search._registered_task = _registered_seal_bag_task
    search.main()
