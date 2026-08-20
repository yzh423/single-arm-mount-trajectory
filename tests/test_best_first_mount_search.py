import numpy as np
import pytest

from design_optimization.best_first_mount_search import (
    CandidateState,
    Fidelity,
    Frontier,
    SearchBudget,
    generate_local_population,
    promotion_target,
    representative_frame_indices,
    representative_frame_windows,
    select_diverse_regions,
    should_stop_after_success,
)


def candidate(identifier: int, optimistic: tuple[float, ...], region: int = 0) -> CandidateState:
    return CandidateState(
        candidate_id=identifier,
        mount=np.zeros(6),
        parent_id=None,
        region_id=region,
        fidelity=Fidelity.RANK,
        score=(0.0,),
        optimistic_score=optimistic,
    )


def test_frontier_uses_optimistic_score_then_candidate_id():
    frontier = Frontier()
    frontier.push(candidate(2, (0.8,)))
    frontier.push(candidate(1, (0.8,)))
    frontier.push(candidate(3, (0.9,)))
    assert [frontier.pop().candidate_id for _ in range(3)] == [3, 1, 2]


def test_fidelity_promotion_is_monotonic():
    assert promotion_target(Fidelity.RANK) is Fidelity.MEDIUM
    assert promotion_target(Fidelity.MEDIUM) is Fidelity.DENSE
    assert promotion_target(Fidelity.DENSE) is Fidelity.FINAL
    assert promotion_target(Fidelity.FINAL) is None


def test_post_success_hard_budget_stops_at_eight_expansions():
    budget = SearchBudget()
    assert should_stop_after_success(
        budget=budget,
        expansions=8,
        pre_success_elapsed_s=100.0,
        post_success_elapsed_s=10.0,
        verified_regions=2,
        non_improving_expansions=1,
        frontier_can_improve=True,
        frontier_empty=False,
    ) == "expansion_budget"


def test_post_success_stops_after_stable_multi_region_evidence():
    assert should_stop_after_success(
        budget=SearchBudget(),
        expansions=4,
        pre_success_elapsed_s=100.0,
        post_success_elapsed_s=10.0,
        verified_regions=3,
        non_improving_expansions=4,
        frontier_can_improve=False,
        frontier_empty=False,
    ) == "stable_multi_region"


def test_budget_rejects_non_64_local_population():
    with pytest.raises(ValueError, match="64"):
        SearchBudget(retained_regions=7, local_per_region=8)


def test_representative_frames_include_ends_and_motion_peak():
    positions = np.c_[np.arange(100), np.zeros(100), np.zeros(100)].astype(float)
    positions[50, 1] = 10.0
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
    indices = representative_frame_indices(positions, quaternions, requested=24)
    assert len(indices) == 24
    assert indices[0] == 0
    assert indices[-1] == 99
    assert 50 in indices
    assert np.all(np.diff(indices) > 0)


def test_representative_frames_fill_static_trajectory_deterministically():
    positions = np.zeros((10, 3))
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
    first = representative_frame_indices(positions, quaternions, requested=6)
    second = representative_frame_indices(positions, quaternions, requested=6)
    assert np.array_equal(first, second)
    assert len(first) == 6


def test_representative_windows_are_contiguous_deterministic_and_cover_ends():
    positions = np.zeros((40, 3))
    positions[18:23, 0] = np.arange(5)
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (40, 1))
    first = representative_frame_windows(
        positions, quaternions, requested_windows=3, window_length=6)
    second = representative_frame_windows(
        positions, quaternions, requested_windows=3, window_length=6)
    assert len(first) == 3
    assert all(np.array_equal(np.diff(window), np.ones(len(window) - 1, dtype=int))
               for window in first)
    assert first[0][0] == 0
    assert first[-1][-1] == 39
    assert any(np.any((window >= 18) & (window <= 22)) for window in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_diverse_regions_keep_best_then_spatially_separate_candidates():
    mounts = np.array([
        [0.0, 0, 0, 0, 0, 0],
        [0.01, 0, 0, 0, 0, 0],
        [1.0, 0, 0, 0, 0, 0],
    ])
    scores = [(3.0,), (2.0,), (1.0,)]
    selected = select_diverse_regions(
        mounts, scores, lower=np.zeros(6), upper=np.ones(6), count=2, shortlist=3,
    )
    assert selected.tolist() == [0, 2]


def test_diverse_regions_accept_active_xyz_yaw_coordinates():
    mounts = np.array([[0.0, 0, 0, 0], [0.01, 0, 0, 0], [1.0, 0, 0, 0]])
    selected = select_diverse_regions(
        mounts, [(3.0,), (2.0,), (1.0,)], lower=np.zeros(4),
        upper=np.ones(4), count=2, shortlist=3)
    assert selected.tolist() == [0, 2]


def test_local_population_is_eight_regions_times_eight_candidates():
    centers = np.arange(48, dtype=float).reshape(8, 6)
    lower = np.full(6, -100.0)
    upper = np.full(6, 100.0)
    population, region_ids, parent_ids = generate_local_population(
        centers, lower, upper, per_region=8, seed=751,
    )
    assert population.shape == (64, 6)
    assert np.bincount(region_ids).tolist() == [8] * 8
    assert np.array_equal(population[::8], centers)
    assert np.array_equal(parent_ids[::8], np.arange(8))


def test_local_population_accepts_active_xyz_yaw_coordinates():
    centers = np.arange(32, dtype=float).reshape(8, 4)
    population, region_ids, parent_ids = generate_local_population(
        centers, np.full(4, -100.0), np.full(4, 100.0),
        per_region=8, seed=751)
    assert population.shape == (64, 4)
    assert np.bincount(region_ids).tolist() == [8] * 8
    assert np.array_equal(parent_ids[::8], np.arange(8))
