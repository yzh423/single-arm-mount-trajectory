from __future__ import annotations

import torch


def sobol_mount_candidates(count: int, seed: int, *, device: str | torch.device = "cpu",
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministic [spacing, forward, height] candidates for a fair roll sweep.

    Bounds deliberately cover flat desktop and anthropomorphic shoulder mounts:
    spacing 0.45--1.15 m, forward offset -0.25--0.20 m, height 0--0.80 m.
    The same candidates must be reused at every roll angle.
    """
    if count < 1:
        raise ValueError("count must be positive")
    unit = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(count)
    lower = torch.tensor((.45, -.25, 0.), dtype=dtype)
    upper = torch.tensor((1.15, .20, .80), dtype=dtype)
    return (lower + unit.to(dtype) * (upper - lower)).to(device)


def mount_quality_score(success: torch.Tensor, position_rmse_m: torch.Tensor,
                        orientation_rmse_rad: torch.Tensor,
                        collision_fraction: torch.Tensor,
                        margin_violation_m: torch.Tensor) -> torch.Tensor:
    """Single selector used only within a fixed-geometry, fixed-roll sweep.

    Raw objectives are retained in the report; this score selects one mount
    without pretending that the multi-objective trade-off has disappeared.
    """
    return (12.0 * (1.0 - success) + 80.0 * position_rmse_m +
            3.0 * orientation_rmse_rad + 20.0 * collision_fraction +
            120.0 * margin_violation_m)
