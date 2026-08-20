"""Candidate generation and diversity policy for hierarchical mount search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class MountSearchBudget:
    """Formal six-dimensional installation-search budget."""

    global_candidates: int = 4096
    global_retain: int = 128
    local_regions: int = 16
    local_per_region: int = 64
    medium_retain: int = 64
    strict_retain: int = 16
    final_centers: int = 4
    final_per_center: int = 128


def _validated_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.ndim != 1 or lower.shape != upper.shape or np.any(lower >= upper):
        raise ValueError("lower and upper must be matching finite-width vectors")
    return lower, upper


def global_mount_candidates(
    lower: np.ndarray,
    upper: np.ndarray,
    incumbent: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Return an incumbent plus broad scrambled-Sobol installation candidates."""
    lower, upper = _validated_bounds(lower, upper)
    incumbent = np.asarray(incumbent, dtype=float)
    warm_starts = incumbent[None] if incumbent.ndim == 1 else incumbent
    if warm_starts.ndim != 2 or warm_starts.shape[1:] != lower.shape or count < max(2, len(warm_starts)):
        raise ValueError("incumbent warm starts must match bounds and fit within count")
    if np.any(warm_starts < lower) or np.any(warm_starts > upper):
        raise ValueError("incumbent lies outside bounds")
    remaining = count - len(warm_starts)
    if remaining == 0:
        return warm_starts.copy()
    power = int(np.ceil(np.log2(remaining)))
    unit = qmc.Sobol(len(lower), scramble=True, seed=seed).random_base2(power)[:remaining]
    return np.vstack((warm_starts, qmc.scale(unit, lower, upper)))


def diverse_region_indices(
    candidates: np.ndarray,
    ranks: Sequence[tuple[float, ...]],
    *,
    shortlist: int,
    regions: int,
    lower: np.ndarray,
    upper: np.ndarray,
    minimum_distance: float = 0.12,
) -> list[int]:
    """Select high-ranking candidates from distinct normalized mount basins."""
    candidates = np.asarray(candidates, dtype=float)
    lower, upper = _validated_bounds(lower, upper)
    if candidates.ndim != 2 or candidates.shape[1] != len(lower) or len(ranks) != len(candidates):
        raise ValueError("candidates, ranks and bounds have incompatible shapes")
    if shortlist < 1 or regions < 1:
        raise ValueError("shortlist and regions must be positive")
    ordered = sorted(range(len(candidates)), key=lambda index: ranks[index], reverse=True)
    pool = ordered[: min(shortlist, len(ordered))]
    normalized = (candidates - lower) / (upper - lower)
    selected: list[int] = []
    for index in pool:
        if not selected or all(
            np.linalg.norm(normalized[index] - normalized[other]) >= minimum_distance
            for other in selected
        ):
            selected.append(index)
            if len(selected) == min(regions, len(pool)):
                return selected
    for index in pool:
        if index not in selected:
            selected.append(index)
            if len(selected) == min(regions, len(pool)):
                break
    return selected


def local_mount_candidates(
    centers: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    per_center: int,
    radii: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Generate one clipped scrambled-Sobol cloud around every center."""
    centers = np.asarray(centers, dtype=float)
    lower, upper = _validated_bounds(lower, upper)
    radii = np.asarray(radii, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != len(lower):
        raise ValueError("centers must have one row per installation and match bounds")
    if radii.shape != lower.shape or np.any(radii <= 0) or per_center < 1:
        raise ValueError("radii must be positive and per_center must be positive")
    clouds = []
    for offset, center in enumerate(centers):
        if np.any(center < lower) or np.any(center > upper):
            raise ValueError("center lies outside bounds")
        if per_center == 1:
            clouds.append(center[None])
            continue
        power = int(np.ceil(np.log2(per_center - 1)))
        unit = qmc.Sobol(len(lower), scramble=True, seed=seed + offset).random_base2(power)
        perturbation = (2.0 * unit[: per_center - 1] - 1.0) * radii
        cloud = np.vstack((center, np.clip(center + perturbation, lower, upper)))
        clouds.append(cloud)
    return np.vstack(clouds) if clouds else np.empty((0, len(lower)), dtype=float)
