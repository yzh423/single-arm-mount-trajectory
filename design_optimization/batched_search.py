"""Memory-bounded evaluation helpers for large candidate populations."""
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch


def safe_candidate_batch_size(requested: int, *, frames: int, seeds: int,
                              maximum_batched_systems: int = 30_000) -> int:
    """Bound simultaneous 6x6 IK systems to avoid CUDA solver instability."""
    if requested < 1 or frames < 1 or seeds < 1 or maximum_batched_systems < 1:
        raise ValueError("batch, frames, seeds and system limit must be positive")
    return max(1, min(requested, maximum_batched_systems // (frames * seeds)))


def evaluate_candidate_batches(
    candidates: torch.Tensor,
    *,
    batch_size: int,
    evaluate: Callable[[torch.Tensor, int], Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Evaluate candidates in order and concatenate detached CPU results."""
    if candidates.ndim < 1:
        raise ValueError("candidates must have a leading candidate dimension")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks: dict[str, list[torch.Tensor]] = {}
    expected_keys: tuple[str, ...] | None = None
    for start in range(0, len(candidates), batch_size):
        stop = min(start + batch_size, len(candidates))
        result = dict(evaluate(candidates[start:stop], start))
        keys = tuple(result)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"inconsistent batch result keys: {keys} != {expected_keys}")
        for key, value in result.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != stop - start:
                raise ValueError(f"{key} must be a tensor with one row per candidate")
            chunks.setdefault(key, []).append(value.detach().cpu())
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}
