"""Deterministic shortlist selection and coarse/strict fidelity diagnostics."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr


def _score_tuple(value: Sequence[float] | float) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return (float(array),)
    if array.ndim != 1 or not len(array) or np.any(~np.isfinite(array)):
        raise ValueError("scores must be finite scalars or non-empty vectors")
    return tuple(float(item) for item in array)


def _ranked_indices(scores: Sequence[Sequence[float] | float]) -> list[int]:
    normalized = [_score_tuple(score) for score in scores]
    return sorted(
        range(len(normalized)),
        key=lambda index: (normalized[index], -index),
        reverse=True,
    )


def select_diverse_candidates(
    mounts: np.ndarray,
    scores: Sequence[Sequence[float] | float],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    pool_size: int,
) -> np.ndarray:
    """Keep the best candidate, then cover distant basins inside a ranked pool."""
    values = np.asarray(mounts, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if (values.ndim != 2 or lower.shape != upper.shape
            or lower.shape != (values.shape[1],) or len(scores) != len(values)):
        raise ValueError("mounts, scores, and bounds must have matching shapes")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(lower + upper)):
        raise ValueError("mounts and bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("every mount bound must have positive width")
    if not 1 <= count <= pool_size <= len(values):
        raise ValueError("count and pool_size must fit the candidate population")

    ordered = _ranked_indices(scores)
    normalized = (values - lower) / (upper - lower)
    score_values = [_score_tuple(score) for score in scores]
    remaining = np.asarray(ordered[:pool_size], dtype=int)
    selected = [int(remaining[0])]
    remaining = remaining[1:]
    minimum_distance = np.linalg.norm(
        normalized[remaining] - normalized[selected[0]], axis=1)
    while len(selected) < count:
        farthest_distance = float(np.max(minimum_distance))
        tied_positions = np.flatnonzero(minimum_distance == farthest_distance)
        chosen_position = max(
            tied_positions.tolist(),
            key=lambda position: (
                score_values[int(remaining[position])],
                -int(remaining[position]),
            ),
        )
        chosen = int(remaining[chosen_position])
        selected.append(chosen)
        keep = np.ones(len(remaining), dtype=bool)
        keep[chosen_position] = False
        remaining = remaining[keep]
        minimum_distance = minimum_distance[keep]
        if len(remaining):
            distance_to_chosen = np.linalg.norm(
                normalized[remaining] - normalized[chosen], axis=1)
            minimum_distance = np.minimum(
                minimum_distance, distance_to_chosen)
    return np.asarray(selected, dtype=int)


def cross_fidelity_metrics(
    candidate_ids: np.ndarray,
    coarse_scores: Sequence[Sequence[float] | float],
    strict_candidate_ids: np.ndarray,
    strict_scores: Sequence[Sequence[float] | float],
    *,
    top_k: int,
) -> dict[str, object]:
    """Measure whether coarse ranking recalls the candidates favored by strict IK."""
    coarse_ids = np.asarray(candidate_ids)
    strict_ids = np.asarray(strict_candidate_ids)
    if coarse_ids.ndim != 1 or strict_ids.ndim != 1:
        raise ValueError("candidate IDs must be one-dimensional")
    if len(coarse_ids) != len(coarse_scores) or len(strict_ids) != len(strict_scores):
        raise ValueError("candidate IDs and scores must have matching lengths")
    if len(set(coarse_ids.tolist())) != len(coarse_ids):
        raise ValueError("coarse candidate IDs must be unique")
    coarse_lookup = {value: index for index, value in enumerate(coarse_ids.tolist())}
    if any(value not in coarse_lookup for value in strict_ids.tolist()):
        raise ValueError("strict candidate IDs must be a subset of coarse candidate IDs")
    if len(set(strict_ids.tolist())) != len(strict_ids):
        raise ValueError("strict candidate IDs must be unique")
    if not 1 <= top_k <= min(len(coarse_ids), len(strict_ids)):
        raise ValueError("top_k must fit both candidate populations")

    coarse_order = _ranked_indices(coarse_scores)
    strict_order = _ranked_indices(strict_scores)
    coarse_rank = {coarse_ids[index].item(): rank
                   for rank, index in enumerate(coarse_order)}
    strict_rank = {strict_ids[index].item(): rank
                   for rank, index in enumerate(strict_order)}
    common = strict_ids.tolist()
    coarse_ranks = np.asarray([coarse_rank[value] for value in common], dtype=float)
    strict_ranks = np.asarray([strict_rank[value] for value in common], dtype=float)
    coarse_common_scores = [
        _score_tuple(coarse_scores[coarse_lookup[value]]) for value in common]
    strict_common_scores = [_score_tuple(value) for value in strict_scores]
    if (len(set(coarse_common_scores)) < 2
            or len(set(strict_common_scores)) < 2):
        correlation_value = None
    else:
        correlation = spearmanr(coarse_ranks, strict_ranks).statistic
        correlation_value = None if not np.isfinite(correlation) else float(correlation)

    coarse_top = [coarse_ids[index].item() for index in coarse_order[:top_k]]
    strict_top = [strict_ids[index].item() for index in strict_order[:top_k]]
    strict_score_values = [_score_tuple(value) for value in strict_scores]
    coarse_score_values = [_score_tuple(value) for value in coarse_scores]
    optimistic = sum(
        coarse_score_values[coarse_lookup[value]][0] > 0.5
        and strict_score_values[index][0] <= 0.5
        for index, value in enumerate(strict_ids.tolist())
    )
    return {
        "compared_candidate_count": len(strict_ids),
        "strict_best_candidate_id": strict_top[0],
        "coarse_top_candidate_ids": coarse_top,
        "strict_top_candidate_ids": strict_top,
        "top_k": int(top_k),
        "top_k_recall": float(len(set(coarse_top) & set(strict_top)) / top_k),
        "spearman_rank_correlation": correlation_value,
        "optimistic_episode_pass_count": int(optimistic),
        "optimistic_episode_pass_rate": float(optimistic / len(strict_ids)),
    }
