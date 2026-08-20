import torch

from design_optimization.surrogate import (ObjectiveSurrogate, ensemble_predict, gaussian_nll,
                                           propose_ucb_candidates, standardize_targets)


def test_surrogate_shapes_and_finite_loss():
    torch.manual_seed(1); x = torch.randn(12, 10); y = torch.randn(12, 7)
    model = ObjectiveSurrogate(); mean, log_variance = model(x)
    assert mean.shape == log_variance.shape == y.shape
    assert torch.isfinite(gaussian_nll(model, x, y))


def test_ensemble_proposal_is_bounded_and_deterministic():
    torch.manual_seed(2); models = [ObjectiveSurrogate(width=24) for _ in range(3)]
    seeds = torch.randn(8, 10); prediction = ensemble_predict(models, seeds)
    assert prediction.epistemic_std.shape == (8, 7)
    generator = torch.Generator().manual_seed(3)
    candidates = propose_ucb_candidates(models, seeds, count=4, steps=3, generator=generator)
    assert candidates.shape == (4, 10)
    assert float(candidates.abs().max()) <= 4


def test_target_standardization_includes_constraint_violation():
    objective = torch.randn(20, 6); violation = torch.rand(20)
    standardized, center, scale = standardize_targets(objective, violation)
    assert standardized.shape == (20, 7)
    torch.testing.assert_close(standardized.mean(0), torch.zeros(7), atol=1e-6, rtol=0)
