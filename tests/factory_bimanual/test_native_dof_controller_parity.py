from __future__ import annotations

from dataclasses import fields
import sys
from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.native_dof_mpc import MPCConfig, solve_small_system


EXTERNAL_ROOT = Path(r"E:\YZH123123\GoodGoodArmDayDayUp-Learning")


def test_mpc_config_preserves_external_configuration_surface():
    sys.path.insert(0, str(EXTERNAL_ROOT))
    try:
        from doosan_teleop.mpc_pvt import MPCConfig as ExternalMPCConfig
    finally:
        sys.path.remove(str(EXTERNAL_ROOT))
    assert {f.name for f in fields(MPCConfig)} == {f.name for f in fields(ExternalMPCConfig)}


def test_generic_solve_matches_external_six_dof_solve():
    sys.path.insert(0, str(EXTERNAL_ROOT))
    try:
        from doosan_teleop.easy_ik import _solve_6x6
    finally:
        sys.path.remove(str(EXTERNAL_ROOT))
    rng = np.random.default_rng(20260812)
    matrix = rng.normal(size=(6, 6))
    matrix = matrix @ matrix.T + 0.1 * np.eye(6)
    rhs = rng.normal(size=6)
    assert solve_small_system(matrix, rhs) == pytest.approx(_solve_6x6(matrix, rhs), abs=1e-11)
