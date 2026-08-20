from pathlib import Path

import torch

from design_optimization.ik import IKResult, solve_trajectory_multistart
from design_optimization.kinematics import build_designs, deterministic_joint_samples, fk_tcp
from design_optimization.topology import load_templates
from design_optimization.trajectory import select_bimanual_collision_aware


def test_warm_started_trajectory_solver_tracks_reachable_smooth_path():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["ur5"]
    design = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                           deterministic_joint_samples(template, 2048))
    q = torch.zeros((8, 6), dtype=torch.float64)
    q[:, 0] = torch.linspace(-.2, .2, 8); q[:, 2] = torch.linspace(.1, .25, 8)
    targets = fk_tcp(design, q)[0]
    result = solve_trajectory_multistart(design, targets, seed_count=8,
                                         initial_iterations=80, tracking_iterations=30)
    assert result.success.any(dim=-1).all()
    best = result.position_error_m.min(dim=-1).values
    assert float(best.max()) < .0025
    selected = result.q[0, :, result.position_error_m[0].argmin(dim=-1)]
    assert torch.isfinite(selected).all()


def test_trajectory_solver_enforces_per_joint_delta_when_requested():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["ur5"]
    design = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                           deterministic_joint_samples(template, 1024))
    q = torch.zeros((5, 6), dtype=torch.float64); q[:, 0] = torch.linspace(0, 1.2, 5)
    targets = fk_tcp(design, q)[0]
    result = solve_trajectory_multistart(design, targets, seed_count=6, initial_iterations=60,
                                         tracking_iterations=20, maximum_joint_delta_rad=.05)
    assert float(torch.diff(result.q[0], dim=0).abs().max()) <= .0500001


def test_bimanual_beam_enforces_per_joint_delta():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["ur5"]
    design = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                           deterministic_joint_samples(template, 1024))
    time_steps, branches = 5, 3
    q = torch.zeros((1, time_steps, branches, 6), dtype=torch.float64)
    q[0, :, 0, 0] = torch.arange(time_steps) * .04
    q[0, :, 1, 0] = 1.0 + torch.arange(time_steps) * .04
    q[0, :, 2, 0] = -1.0 + torch.arange(time_steps) * .04
    scalar = torch.zeros((1, time_steps, branches), dtype=torch.float64)
    result = IKResult(q, scalar, scalar, torch.ones_like(scalar) * .2,
                      torch.ones_like(scalar, dtype=torch.bool))
    path = select_bimanual_collision_aware(
        result, result, design,
        torch.tensor([-.3, 0., .5], dtype=torch.float64),
        torch.tensor([.3, 0., .5], dtype=torch.float64),
        maximum_joint_step_rad=.05, collision_weight=0.0,
    )
    assert float(torch.diff(path.left_q, dim=0).abs().max()) <= .050001
    assert float(torch.diff(path.right_q, dim=0).abs().max()) <= .050001
