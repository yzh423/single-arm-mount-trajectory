from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


class ObjectiveSurrogate(nn.Module):
    """Heteroscedastic multi-objective regressor for geometry+mount candidates."""

    def __init__(self, input_dim: int = 10, objective_dim: int = 7, width: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(),
                                      nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                      nn.Linear(width, 2 * objective_dim))
        self.objective_dim = objective_dim

    def forward(self, parameters: torch.Tensor):
        mean, raw_log_variance = self.backbone(parameters).chunk(2, dim=-1)
        return mean, raw_log_variance.clamp(-8, 5)


def gaussian_nll(model: ObjectiveSurrogate, parameters: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
    mean, log_variance = model(parameters)
    return .5 * (log_variance + (targets - mean).square() * torch.exp(-log_variance)).mean()


@dataclass
class EnsemblePrediction:
    mean: torch.Tensor
    epistemic_std: torch.Tensor
    aleatoric_std: torch.Tensor


def ensemble_predict(models: list[ObjectiveSurrogate], parameters: torch.Tensor) -> EnsemblePrediction:
    predictions, variances = zip(*(model(parameters) for model in models))
    means = torch.stack(predictions); aleatoric = torch.stack(variances).exp()
    return EnsemblePrediction(means.mean(0), means.std(0, unbiased=False),
                              aleatoric.mean(0).sqrt())


def standardize_targets(objectives: torch.Tensor, violation: torch.Tensor):
    targets = torch.cat((objectives, violation[:, None]), dim=-1)
    center = targets.mean(0); scale = targets.std(0).clamp_min(1e-6)
    return (targets - center) / scale, center, scale


def propose_ucb_candidates(models: list[ObjectiveSurrogate], seeds: torch.Tensor,
                           *, count: int, steps: int = 80, learning_rate: float = .04,
                           exploration: float = .6, generator: torch.Generator | None = None):
    """Gradient-optimize random scalarizations of ensemble lower confidence bounds."""
    if count > seeds.shape[0]: raise ValueError("count cannot exceed number of proposal seeds")
    candidates = seeds[:count].detach().clone().requires_grad_(True)
    objective_dim = models[0].objective_dim
    weights = torch.rand((count, objective_dim), device=seeds.device, dtype=seeds.dtype,
                         generator=generator)
    weights = weights / weights.sum(-1, keepdim=True)
    optimizer = torch.optim.Adam([candidates], lr=learning_rate)
    for _ in range(steps):
        prediction = ensemble_predict(models, candidates)
        lower_confidence = prediction.mean - exploration * prediction.epistemic_std
        loss = (weights * lower_confidence).sum(-1).mean() + 1e-4 * candidates.square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad(): candidates.clamp_(-4, 4)
    return candidates.detach()

