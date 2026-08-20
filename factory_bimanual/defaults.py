"""Single source of truth for the isolated factory-bimanual preflight."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .registration import RigidTaskRegistration


ROOT = Path(__file__).resolve().parents[1]


def derive_initial_registration(task) -> RigidTaskRegistration:
    """Deterministic translation-only placement; identical for every robot."""
    points = np.vstack((task.left_position_m, task.right_position_m))
    translation = np.array([
        -float(points[:, 0].mean()),
        -float(points[:, 1].mean()),
        0.90 - float(points[:, 2].min()),
    ])
    return RigidTaskRegistration(np.eye(3), translation)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    csv_path: Path

    def registration(self, task) -> RigidTaskRegistration:
        return derive_initial_registration(task)


TASK_SPECS = (
    TaskSpec("screw_cap", ROOT / "data/factory/8-11/Screw_Cap/handheld_20260811_162854.csv"),
    TaskSpec("pour_raw_material", ROOT / "data/factory/8-12/PourRawMaterial/handheld_20260812_111542.csv"),
)

# These are scan inputs, not selected mounts. Preflight must stay blocked until
# a measured spacing result is recorded for every robot.
SPACING_CANDIDATES_M = tuple(float(v) for v in np.arange(0.60, 1.01, 0.05))
# Deliberately unset until workspace feasibility passes.  The representative
# scan produced zero synchronous coverage for every candidate, so selecting a
# tie winner would fabricate readiness rather than establish an installation.
SELECTED_SPACING_M: dict[str, float] = {}

CONTROLLER_PROFILES_PATH = ROOT / "factory_bimanual/controller_profiles.json"
OUTPUT_ROOT = ROOT / "reports/factory_bimanual"
