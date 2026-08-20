"""Dimension-aware global-plus-local candidate policies for morphology search."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc


def dimension_aware_candidate_budget(dimension: int) -> int:
    """Return a power-of-two budget with at least 128 samples per dimension."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    # Full three-axis installation search is substantially harder than the
    # fixed-yaw/roll screen, so do not pretend 2,048 points cover 13-D space.
    requested = max(8192 if dimension >= 12 else 1024, 128 * int(dimension))
    return 1 << int(math.ceil(math.log2(requested)))


def staged_sobol_candidates(
    lower: np.ndarray,
    upper: np.ndarray,
    incumbent: np.ndarray,
    *,
    budget: int,
    seed: int,
    local_fraction: float = 0.25,
) -> np.ndarray:
    """Combine broad Sobol coverage with a bounded local cloud around incumbent."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    incumbent = np.asarray(incumbent, dtype=float)
    if lower.shape != upper.shape or lower.shape != incumbent.shape or lower.ndim != 1:
        raise ValueError("bounds and incumbent must have matching one-dimensional shapes")
    if np.any(lower >= upper) or budget < 2:
        raise ValueError("invalid bounds or budget")
    dimension = len(lower)
    local_count = max(1, int(round(budget * local_fraction)))
    global_count = budget - local_count
    global_power = int(math.ceil(math.log2(max(global_count, 1))))
    global_unit = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(global_power)[:global_count]
    broad = qmc.scale(global_unit, lower, upper)
    local_power = int(math.ceil(math.log2(local_count)))
    local_unit = qmc.Sobol(dimension, scramble=True, seed=seed + 1).random_base2(local_power)[:local_count]
    span = upper - lower
    local = incumbent + (2.0 * local_unit - 1.0) * (0.15 * span)
    local = np.clip(local, lower, upper)
    candidates = np.vstack((incumbent[None], broad, local))[:budget]
    if len(candidates) < budget:
        candidates = np.vstack((candidates, broad[: budget - len(candidates)]))
    return candidates
