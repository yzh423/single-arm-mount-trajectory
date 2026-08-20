"""Handbook-aligned provenance validation and aggregate reporting."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


VALID_STATES = frozenset({"VERIFIED", "UNVERIFIED", "NOT_APPLICABLE", "REJECTED"})


def validate_provenance(
    states: Mapping[str, str], *, allow_unverified: bool = False
) -> dict[str, str]:
    normalized = {str(key): str(value).upper() for key, value in states.items()}
    invalid = {key: value for key, value in normalized.items() if value not in VALID_STATES}
    if invalid:
        raise ValueError(f"invalid provenance states: {invalid}")
    rejected = [key for key, value in normalized.items() if value == "REJECTED"]
    if rejected:
        raise ValueError(f"REJECTED provenance dependencies: {', '.join(rejected)}")
    unverified = [key for key, value in normalized.items() if value == "UNVERIFIED"]
    if unverified and not allow_unverified:
        raise ValueError(f"UNVERIFIED provenance dependencies: {', '.join(unverified)}")
    return normalized


def aggregate_scores(scores: Sequence[float], *, threshold: float) -> dict[str, object]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite one-dimensional sequence")
    return {
        "mean": float(np.mean(values)),
        "threshold": float(threshold),
        "threshold_count": int(np.sum(values >= threshold)),
        "threshold_total": int(len(values)),
        "p10": float(np.percentile(values, 10)),
    }
