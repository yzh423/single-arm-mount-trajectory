import pytest
import torch

from design_optimization.robustness import (symmetric_sobol_perturbations,
                                            upper_tail_cvar)


def test_sobol_mount_perturbations_are_reproducible_centered_and_antithetic():
    first = symmetric_sobol_perturbations(33, 4, seed=17, dtype=torch.float64)
    second = symmetric_sobol_perturbations(33, 4, seed=17, dtype=torch.float64)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[0], torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(first[1:17], -first[17:33])
    assert bool((first.abs() <= 1).all())


def test_upper_tail_cvar_uses_worst_largest_values():
    values = torch.tensor([0., 1., 2., 3., 4.])
    assert upper_tail_cvar(values, .4) == pytest.approx(3.5)
