import torch

from design_optimization.mount_sweep import mount_quality_score, sobol_mount_candidates


def test_sobol_mount_candidates_are_reproducible_and_in_bounds():
    first = sobol_mount_candidates(17, 123)
    second = sobol_mount_candidates(17, 123)
    assert torch.equal(first, second)
    assert bool(((first[:, 0] >= .45) & (first[:, 0] <= 1.15)).all())
    assert bool(((first[:, 1] >= -.25) & (first[:, 1] <= .20)).all())
    assert bool(((first[:, 2] >= 0.) & (first[:, 2] <= .80)).all())


def test_mount_quality_prefers_success_and_clearance():
    score = mount_quality_score(
        torch.tensor([1., .9, 1.]), torch.tensor([.002, .002, .002]),
        torch.tensor([.01, .01, .01]), torch.tensor([0., 0., .1]),
        torch.tensor([0., 0., .01]))
    assert int(score.argmin()) == 0
