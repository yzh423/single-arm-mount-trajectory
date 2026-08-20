from __future__ import annotations

import torch


def symmetric_sobol_perturbations(
    count: int,
    dimension: int,
    *,
    seed: int = 20260806,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic zero-centered, antithetic Sobol perturbations in [-1, 1]."""
    if count < 1 or dimension < 1:
        raise ValueError("count and dimension must be positive")
    half = count // 2
    if half:
        engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
        positive = 2.0 * engine.draw(half).to(dtype=dtype) - 1.0
        samples = torch.cat((torch.zeros((1, dimension), dtype=dtype),
                             positive, -positive), dim=0)
    else:
        samples = torch.zeros((1, dimension), dtype=dtype)
    if count % 2 == 0:
        samples = samples[:-1]
    return samples.to(device)


def upper_tail_cvar(values: torch.Tensor, fraction: float = .2) -> torch.Tensor:
    """Mean of the largest ``fraction`` along axis 0 (larger means worse)."""
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("values must have a non-empty sample axis")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(torch.ceil(torch.tensor(values.shape[0] * fraction)).item()))
    return torch.topk(values, count, dim=0, largest=True).values.mean(dim=0)
