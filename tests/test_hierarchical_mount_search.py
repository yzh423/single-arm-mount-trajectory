import numpy as np

from design_optimization.hierarchical_mount_search import (
    MountSearchBudget,
    diverse_region_indices,
    global_mount_candidates,
    local_mount_candidates,
)
from scripts.search_strict_urdf_mount import build_parser, resolved_hierarchy_budget


def test_formal_budget_matches_hierarchical_protocol():
    budget = MountSearchBudget()
    assert budget.global_candidates == 4096
    assert budget.global_retain == 128
    assert budget.local_regions == 16
    assert budget.local_per_region == 64
    assert budget.medium_retain == 64
    assert budget.strict_retain == 16
    assert budget.final_centers == 4
    assert budget.final_per_center == 128


def test_global_candidates_keep_incumbent_and_cover_bounds_deterministically():
    lower = np.array([-0.5, -0.55, 0.02, -90.0, -180.0, -180.0])
    upper = np.array([0.5, 0.25, 0.65, 90.0, 180.0, 180.0])
    incumbent = np.array([0.0, -0.16, 0.18, 0.0, 0.0, 0.0])

    first = global_mount_candidates(lower, upper, incumbent, count=32, seed=750)
    second = global_mount_candidates(lower, upper, incumbent, count=32, seed=750)

    assert first.shape == (32, 6)
    np.testing.assert_allclose(first[0], incumbent)
    np.testing.assert_allclose(first, second)
    assert np.all(first >= lower)
    assert np.all(first <= upper)
    assert np.ptp(first[:, 0]) > 0.7
    assert np.ptp(first[:, 1]) > 0.5


def test_global_candidates_preserve_multiple_real_and_proxy_warm_starts():
    lower = np.full(3, -1.0); upper = np.full(3, 1.0)
    warm_starts = np.asarray(((0.1, 0.2, 0.3), (-0.4, 0.5, -0.6), (0.0, 0.0, 0.0)))
    candidates = global_mount_candidates(lower, upper, warm_starts, count=16, seed=750)
    np.testing.assert_allclose(candidates[:3], warm_starts)


def test_diverse_regions_do_not_all_come_from_one_mount_basin():
    candidates = np.array([
        [0.00, 0, 0, 0, 0, 0],
        [0.01, 0, 0, 0, 0, 0],
        [0.02, 0, 0, 0, 0, 0],
        [0.90, 0, 0, 0, 0, 0],
        [-0.90, 0, 0, 0, 0, 0],
    ])
    ranks = [(10,), (9,), (8,), (7,), (6,)]
    lower = np.array([-1, -1, -1, -1, -1, -1], dtype=float)
    upper = np.array([1, 1, 1, 1, 1, 1], dtype=float)

    selected = diverse_region_indices(
        candidates, ranks, shortlist=5, regions=3,
        lower=lower, upper=upper, minimum_distance=0.2,
    )

    assert selected[0] == 0
    assert set(selected) == {0, 3, 4}


def test_local_candidates_allocate_the_requested_cloud_to_every_center():
    centers = np.array([
        [0.0, -0.2, 0.2, 0.0, 0.0, 0.0],
        [0.3, 0.0, 0.4, 20.0, 30.0, 10.0],
    ])
    lower = np.array([-0.5, -0.55, 0.02, -90.0, -180.0, -180.0])
    upper = np.array([0.5, 0.25, 0.65, 90.0, 180.0, 180.0])
    radii = np.array([0.08, 0.08, 0.06, 15.0, 20.0, 12.0])

    candidates = local_mount_candidates(
        centers, lower, upper, per_center=8, radii=radii, seed=751,
    )

    assert candidates.shape == (16, 6)
    np.testing.assert_allclose(candidates[0], centers[0])
    np.testing.assert_allclose(candidates[8], centers[1])
    assert np.all(candidates >= lower)
    assert np.all(candidates <= upper)
    assert np.all(np.abs(candidates[:8] - centers[0]) <= radii + 1e-12)
    assert np.all(np.abs(candidates[8:] - centers[1]) <= radii + 1e-12)


def test_strict_search_cli_defaults_to_formal_hierarchical_budget():
    args = build_parser().parse_args([
        "--domain", "local", "--robot", "xarm6", "--task", "cap-left",
    ])
    assert args.global_candidates == 4096
    assert args.global_retain == 128
    assert args.local_regions == 16
    assert args.local_per_region == 64
    assert args.medium_retain == 64
    assert args.strict_retain == 16
    assert args.final_centers == 4
    assert args.final_per_center == 128
    assert args.output is None


def test_legacy_candidates_flag_uses_a_compact_smoke_hierarchy():
    args = build_parser().parse_args([
        "--domain", "local", "--robot", "xarm6", "--task", "cap-left",
        "--candidates", "32",
    ])
    resolved = resolved_hierarchy_budget(args)
    assert resolved == {
        "global": 32, "global_retain": 16, "local_regions": 4,
        "local_per_region": 8, "medium_retain": 16, "strict_retain": 6,
        "final_centers": 2, "final_per_center": 16, "final_retain": 4,
    }


def test_strict_search_accepts_explicit_gpu_coarse_incumbent_file():
    args = build_parser().parse_args([
        "--domain", "local", "--robot", "xarm6", "--task", "cap-left",
        "--incumbent-json", "reports/single_arm/dense_search_results.json",
    ])
    assert str(args.incumbent_json).endswith("dense_search_results.json")
