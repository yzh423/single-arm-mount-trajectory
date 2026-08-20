"""Pure scheduling gate for conditional bimanual MPC comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ArmPair = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class BranchDecision:
    left_max_delta_rad: float | None
    right_max_delta_rad: float | None
    equivalent: bool
    required_mpc_modes: tuple[str, ...]
    canonical_initializer: str | None


def _arm_delta(a: np.ndarray, b: np.ndarray, periodic: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    if a.ndim != 1 or a.shape != b.shape or periodic.shape != a.shape:
        raise ValueError("initializer and periodic arrays must have matching 1-D shapes")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("initializer joints must be finite")
    difference = a - b
    difference[periodic] = (difference[periodic] + np.pi) % (2 * np.pi) - np.pi
    return float(np.max(np.abs(difference), initial=0.0))


def compare_initial_branches(
    a_q: ArmPair | None,
    easy_q: ArmPair | None,
    periodic: ArmPair | None = None,
    *,
    threshold_deg: float = 5.0,
) -> BranchDecision:
    """Apply the strict-below-threshold gate independently to both arms."""
    if threshold_deg <= 0:
        raise ValueError("threshold_deg must be positive")
    if a_q is None or easy_q is None:
        modes = (() if a_q is None and easy_q is None else
                 ("mpc_easyik",) if a_q is None else ("mpc_a",))
        initializer = None if not modes else ("easyik" if a_q is None else "a")
        return BranchDecision(None, None, False, modes, initializer)
    if len(a_q) != 2 or len(easy_q) != 2:
        raise ValueError("each initializer must contain left and right joints")
    if periodic is None:
        periodic = tuple(np.ones_like(np.asarray(arm), dtype=bool) for arm in a_q)  # type: ignore[assignment]
    if len(periodic) != 2:
        raise ValueError("periodic must contain left and right masks")
    left = _arm_delta(a_q[0], easy_q[0], periodic[0])
    right = _arm_delta(a_q[1], easy_q[1], periodic[1])
    threshold = np.deg2rad(threshold_deg)
    # Modulo wrapping can move an exact threshold by an ulp; equality belongs
    # to the split branch by contract (only strictly below five degrees folds).
    below = lambda value: value < threshold and not np.isclose(
        value, threshold, rtol=0.0, atol=1e-12
    )
    equivalent = below(left) and below(right)
    modes = ("canonical_a",) if equivalent else ("mpc_a", "mpc_easyik")
    return BranchDecision(left, right, equivalent, modes, "a" if equivalent else None)
