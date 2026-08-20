"""Deterministic policy primitives for budgeted best-first mount search."""
from __future__ import annotations

import enum
import heapq
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import qmc


class Fidelity(enum.IntEnum):
    GATE = 0
    RANK = 1
    MEDIUM = 2
    DENSE = 3
    FINAL = 4


@dataclass(frozen=True)
class SearchBudget:
    global_candidates: int = 128
    retained_regions: int = 8
    local_per_region: int = 8
    rank_frames: int = 24
    medium_frames: int = 64
    dense_before_success: int = 3
    dense_fallback: int = 5
    post_success_expansions: int = 8
    post_success_dense: int = 2
    post_success_time_fraction: float = 0.25
    post_success_regions: int = 3
    non_improving_limit: int = 4
    final_candidates: int = 2

    def __post_init__(self) -> None:
        integer_values = (
            self.global_candidates, self.retained_regions, self.local_per_region,
            self.rank_frames, self.medium_frames, self.dense_before_success,
            self.dense_fallback, self.post_success_expansions,
            self.post_success_dense,
            self.post_success_regions, self.non_improving_limit,
            self.final_candidates,
        )
        if any(value < 1 for value in integer_values):
            raise ValueError("search budget values must be positive")
        if self.retained_regions * self.local_per_region != 64:
            raise ValueError("local population must contain exactly 64 candidates")
        if not 0.0 < self.post_success_time_fraction <= 1.0:
            raise ValueError("post-success time fraction must be in (0, 1]")


@dataclass
class CandidateState:
    candidate_id: int
    mount: np.ndarray
    parent_id: int | None
    region_id: int
    fidelity: Fidelity
    score: tuple[float, ...]
    optimistic_score: tuple[float, ...]
    closed: bool = False

    def __post_init__(self) -> None:
        mount = np.asarray(self.mount, dtype=float)
        if mount.shape != (6,):
            raise ValueError("mount must contain six installation parameters")
        self.mount = mount


class Frontier:
    """Max-priority frontier with deterministic ascending-id tie breaking."""

    def __init__(self) -> None:
        self._heap: list[tuple[tuple[float, ...], int, CandidateState]] = []

    def push(self, candidate: CandidateState) -> None:
        priority = tuple(-float(value) for value in candidate.optimistic_score)
        heapq.heappush(self._heap, (priority, candidate.candidate_id, candidate))

    def pop(self) -> CandidateState:
        while self._heap:
            _, _, candidate = heapq.heappop(self._heap)
            if not candidate.closed:
                return candidate
        raise IndexError("pop from empty frontier")

    def __len__(self) -> int:
        return sum(not item[2].closed for item in self._heap)


def promotion_target(fidelity: Fidelity) -> Fidelity | None:
    if fidelity >= Fidelity.FINAL:
        return None
    return Fidelity(int(fidelity) + 1)


def should_stop_after_success(
    *,
    budget: SearchBudget,
    expansions: int,
    pre_success_elapsed_s: float,
    post_success_elapsed_s: float,
    verified_regions: int,
    non_improving_expansions: int,
    frontier_can_improve: bool,
    frontier_empty: bool,
) -> str | None:
    if expansions >= budget.post_success_expansions:
        return "expansion_budget"
    if post_success_elapsed_s >= budget.post_success_time_fraction * pre_success_elapsed_s:
        return "time_budget"
    if frontier_empty:
        return "frontier_empty"
    if (verified_regions >= budget.post_success_regions
            and non_improving_expansions >= budget.non_improving_limit
            and not frontier_can_improve):
        return "stable_multi_region"
    if not frontier_can_improve:
        return "frontier_dominated"
    return None


def representative_frame_indices(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    *,
    requested: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    quaternions = np.asarray(quaternions_wxyz, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (frames, 3)")
    if quaternions.shape != (len(positions), 4) or len(positions) == 0:
        raise ValueError("quaternions must have shape (frames, 4)")
    if requested < 2:
        raise ValueError("at least two representative frames are required")
    count = min(requested, len(positions))
    if count == len(positions):
        return np.arange(len(positions), dtype=int)

    translation_step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    normalized = quaternions / np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12)
    dots = np.abs(np.sum(normalized[:-1] * normalized[1:], axis=1)).clip(0.0, 1.0)
    rotation_step = 2.0 * np.arccos(dots)
    translation_scale = max(float(np.median(translation_step[translation_step > 0]))
                            if np.any(translation_step > 0) else 1.0, 1e-12)
    rotation_scale = max(float(np.median(rotation_step[rotation_step > 0]))
                         if np.any(rotation_step > 0) else 1.0, 1e-12)
    step = translation_step / translation_scale + rotation_step / rotation_scale
    frame_motion = np.zeros(len(positions), dtype=float)
    frame_motion[1:] += step
    frame_motion[:-1] += step

    selected = {0, len(positions) - 1}
    peak_budget = min(max(1, count // 4), count - len(selected))
    peak_order = sorted(range(1, len(positions) - 1),
                        key=lambda index: (-frame_motion[index], index))
    selected.update(peak_order[:peak_budget])

    cumulative = np.r_[0.0, np.cumsum(translation_step + rotation_step)]
    if cumulative[-1] > 0.0:
        targets = np.linspace(0.0, cumulative[-1], count)
        selected.update(int(np.argmin(np.abs(cumulative - target))) for target in targets)
    else:
        selected.update(np.linspace(0, len(positions) - 1, count).round().astype(int).tolist())

    if len(selected) < count:
        uniform = np.linspace(0, len(positions) - 1, count * 2).round().astype(int)
        selected.update(uniform.tolist())
    if len(selected) < count:
        selected.update(range(len(positions)))

    mandatory = {0, len(positions) - 1, *peak_order[:peak_budget]}
    ranked = sorted(selected, key=lambda index: (
        0 if index in mandatory else 1,
        -frame_motion[index],
        index,
    ))[:count]
    return np.asarray(sorted(ranked), dtype=int)


def representative_frame_windows(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    *,
    requested_windows: int = 3,
    window_length: int = 8,
) -> tuple[np.ndarray, ...]:
    """Choose deterministic contiguous windows without inventing cross-window continuity."""
    frame_count = len(np.asarray(positions))
    if requested_windows < 1 or window_length < 2:
        raise ValueError("window count must be positive and window length at least two")
    if frame_count < 2:
        raise ValueError("at least two frames are required")
    length = min(window_length, frame_count)
    requested = min(requested_windows, max(1, int(np.ceil(frame_count / length))))
    anchors = representative_frame_indices(
        positions, quaternions_wxyz, requested=max(2, requested + 2))
    starts = [0, frame_count - length]
    for anchor in anchors[1:-1]:
        starts.append(int(np.clip(anchor - length // 2, 0, frame_count - length)))
    starts = sorted(set(starts))
    if len(starts) > requested:
        mandatory = {0, frame_count - length}
        interior = [start for start in starts if start not in mandatory]
        peak_frames = set(representative_frame_indices(
            positions, quaternions_wxyz, requested=max(2, requested)))
        interior.sort(key=lambda start: (
            -sum(start <= frame < start + length for frame in peak_frames), start))
        starts = sorted(list(mandatory) + interior[:max(0, requested - len(mandatory))])
    if len(starts) < requested:
        for start in np.linspace(0, frame_count - length, requested).round().astype(int):
            starts.append(int(start))
            starts = sorted(set(starts))
            if len(starts) == requested:
                break
    return tuple(np.arange(start, start + length, dtype=int) for start in starts[:requested])


def select_diverse_regions(
    mounts: np.ndarray,
    scores: Sequence[tuple[float, ...]],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int = 8,
    shortlist: int = 24,
) -> np.ndarray:
    mounts = np.asarray(mounts, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (mounts.ndim != 2 or lower.shape != upper.shape
            or lower.shape != (mounts.shape[1],) or len(scores) != len(mounts)):
        raise ValueError("mounts, scores, and bounds must have matching dimensions")
    if count < 1 or count > len(mounts):
        raise ValueError("region count must fit the candidate population")
    ranked = sorted(range(len(mounts)), key=lambda index: (scores[index], -index), reverse=True)
    pool = ranked[:max(count, min(shortlist, len(ranked)))]
    scale = np.where(upper > lower, upper - lower, 1.0)
    normalized = (mounts - lower) / scale
    selected = [pool.pop(0)]
    while len(selected) < count:
        next_index = max(pool, key=lambda index: (
            min(float(np.linalg.norm(normalized[index] - normalized[prior])) for prior in selected),
            scores[index],
            -index,
        ))
        selected.append(next_index)
        pool.remove(next_index)
    return np.asarray(selected, dtype=int)


def generate_local_population(
    centers: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    per_region: int = 8,
    seed: int = 751,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (centers.ndim != 2 or lower.shape != upper.shape
            or lower.shape != (centers.shape[1],)):
        raise ValueError("centers and bounds must have matching dimensions")
    if per_region < 1:
        raise ValueError("per-region budget must be positive")
    span = upper - lower
    if centers.shape[1] == 4:
        radii = np.asarray((0.08, 0.08, 0.06, 20.0), dtype=float)
    elif centers.shape[1] == 6:
        radii = np.asarray((0.08, 0.08, 0.06, 15.0, 20.0, 12.0), dtype=float)
    else:
        raise ValueError("mount search supports four or six coordinates")
    radii = np.minimum(radii, span / 2.0)
    population: list[np.ndarray] = []
    region_ids: list[int] = []
    parent_ids: list[int] = []
    for region_id, center in enumerate(centers):
        population.append(np.clip(center, lower, upper))
        region_ids.append(region_id)
        parent_ids.append(region_id)
        remaining = per_region - 1
        if remaining:
            power = int(np.ceil(np.log2(remaining)))
            unit = qmc.Sobol(
                centers.shape[1], scramble=True, seed=seed + region_id
            ).random_base2(power)[:remaining]
            perturbations = (2.0 * unit - 1.0) * radii
            for candidate in np.clip(center + perturbations, lower, upper):
                population.append(candidate)
                region_ids.append(region_id)
                parent_ids.append(region_id)
    return (np.asarray(population), np.asarray(region_ids, dtype=int),
            np.asarray(parent_ids, dtype=int))
