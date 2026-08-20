import numpy as np
import pytest

from design_optimization.fidelity_funnel import (
    cross_fidelity_metrics,
    select_diverse_candidates,
)


def test_diverse_selector_keeps_best_and_distant_mount_basins() -> None:
    mounts = np.asarray([
        (0.00, 0.00),
        (0.01, 0.01),
        (0.95, 0.00),
        (0.00, 0.95),
    ])
    scores = [(4.0,), (3.0,), (2.0,), (1.0,)]

    selected = select_diverse_candidates(
        mounts, scores, lower=np.zeros(2), upper=np.ones(2),
        count=3, pool_size=4)

    assert selected.tolist() == [0, 2, 3]


def test_diverse_selector_is_deterministic_for_equal_distances() -> None:
    mounts = np.asarray(((0.0,), (0.5,), (1.0,)))
    scores = [(1.0,), (1.0,), (1.0,)]

    first = select_diverse_candidates(
        mounts, scores, lower=np.zeros(1), upper=np.ones(1), count=2, pool_size=3)
    second = select_diverse_candidates(
        mounts, scores, lower=np.zeros(1), upper=np.ones(1), count=2, pool_size=3)

    assert first.tolist() == second.tolist() == [0, 2]


def test_diverse_selector_rejects_incompatible_inputs() -> None:
    with pytest.raises(ValueError, match="matching"):
        select_diverse_candidates(
            np.zeros((3, 2)), [(1.0,)], lower=np.zeros(2), upper=np.ones(2),
            count=2, pool_size=3)


def test_incremental_diverse_selector_matches_bruteforce_reference() -> None:
    rng = np.random.default_rng(17)
    mounts = rng.uniform(0.0, 1.0, size=(100, 4))
    scores = [(float(value),) for value in rng.normal(size=100)]
    ordered = sorted(range(len(mounts)), key=lambda index: (scores[index], -index),
                     reverse=True)
    pool = ordered[:80]
    expected = [pool.pop(0)]
    while len(expected) < 20:
        chosen = max(pool, key=lambda index: (
            min(float(np.linalg.norm(mounts[index] - mounts[prior]))
                for prior in expected),
            scores[index], -index))
        expected.append(chosen); pool.remove(chosen)

    actual = select_diverse_candidates(
        mounts, scores, lower=np.zeros(4), upper=np.ones(4),
        count=20, pool_size=80)

    assert actual.tolist() == expected


def test_cross_fidelity_metrics_trace_ids_and_measure_recall() -> None:
    candidate_ids = np.asarray((10, 11, 12, 13))
    coarse_scores = [(1, .99), (1, .98), (0, .97), (0, .96)]
    strict_ids = np.asarray((10, 11, 12, 13))
    strict_scores = [(0, .20), (1, 1.0), (0, .30), (0, .10)]

    metrics = cross_fidelity_metrics(
        candidate_ids, coarse_scores, strict_ids, strict_scores, top_k=1)

    assert metrics["strict_best_candidate_id"] == 11
    assert metrics["coarse_top_candidate_ids"] == [10]
    assert metrics["strict_top_candidate_ids"] == [11]
    assert metrics["top_k_recall"] == 0.0
    assert metrics["optimistic_episode_pass_count"] == 1
    assert metrics["compared_candidate_count"] == 4
    assert np.isfinite(metrics["spearman_rank_correlation"])


def test_cross_fidelity_metrics_reports_undefined_constant_ranking() -> None:
    metrics = cross_fidelity_metrics(
        np.asarray((1, 2)), [(0,), (0,)],
        np.asarray((1, 2)), [(0,), (0,)], top_k=1)

    assert metrics["spearman_rank_correlation"] is None


def test_cross_fidelity_metrics_rejects_untraceable_strict_id() -> None:
    with pytest.raises(ValueError, match="subset"):
        cross_fidelity_metrics(
            np.asarray((1, 2)), [(1,), (0,)],
            np.asarray((2, 3)), [(1,), (0,)], top_k=1)
