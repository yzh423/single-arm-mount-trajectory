from __future__ import annotations

import torch

from design_optimization.ik import pose_error, solve_multistart
from design_optimization.kinematics import (
    build_designs,
    deterministic_joint_samples,
    fk_tcp,
    geometric_jacobian,
)
from design_optimization.topology import load_templates
from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64


def _design(kind: str = "xarm6"):
    audit = Path(__file__).resolve().parents[1] / "reports" / "parametric_topology_audit.json"
    template = load_templates(audit, device=DEVICE, dtype=DTYPE)[kind]
    calibration = deterministic_joint_samples(template, 8192)
    return template, build_designs(template, torch.zeros((1, 6), device=DEVICE, dtype=DTYPE), calibration)


def test_geometric_jacobian_matches_finite_difference():
    template, design = _design()
    q = (0.37 * template.q_min + 0.63 * template.q_max).view(1, 1, 6)
    analytic = geometric_jacobian(design, q)[0, 0]
    base = fk_tcp(design, q)[0, 0]
    eps = 1e-6
    numeric = []
    for joint in range(6):
        shifted = q.clone()
        shifted[..., joint] += eps
        changed = fk_tcp(design, shifted)[0, 0]
        numeric.append(pose_error(base[None], changed[None])[0] / eps)
    numeric = torch.stack(numeric, dim=-1)
    torch.testing.assert_close(analytic, numeric, atol=2e-5, rtol=2e-5)


def test_ik_recovers_targets_when_true_branch_is_available():
    template, design = _design("doosan")
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    u = torch.rand((24, 6), generator=generator, device=DEVICE, dtype=DTYPE)
    truth = template.q_min + u * (template.q_max - template.q_min)
    targets = fk_tcp(design, truth)[0]
    seeds = truth.view(1, 24, 1, 6)
    result = solve_multistart(design, targets, seeds, iterations=3)
    assert bool(result.success.all())
    assert float(result.position_error_m.max()) < 1e-8
    assert float(result.orientation_error_rad.max()) < 1e-6
