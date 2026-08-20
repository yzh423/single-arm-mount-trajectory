from pathlib import Path

import torch

from design_optimization.kinematics import assemble_from_deltas, fk_tcp
from design_optimization.slp_refinement import refine_joint_window_slp
from design_optimization.topology import load_templates


ROOT = Path(__file__).resolve().parents[1]


def test_slp_refinement_reduces_exact_fk_merit_and_preserves_endpoints():
    template = load_templates(ROOT / "reports/parametric_topology_audit.json",
                              device="cpu", dtype=torch.float64)["doosan"]
    design = assemble_from_deltas(template, template.deltas)
    time = torch.linspace(0, 1, 7, dtype=torch.float64)
    goal = torch.zeros((7, 6), dtype=torch.float64)
    goal[:, 0] = .2 * time
    goal[:, 1] = -.15 * time
    targets = fk_tcp(design, goal)[0].detach()
    initial = goal.clone()
    initial[1:-1, 0] += .08
    initial[1:-1, 1] -= .06
    result = refine_joint_window_slp(
        design, targets, initial, iterations=12, trust_region_rad=.12,
        smoothness_weight=.05, fix_endpoints=True)
    assert result.accepted_iterations > 0
    assert result.final_merit < .25 * result.initial_merit
    torch.testing.assert_close(result.q[0], initial[0])
    torch.testing.assert_close(result.q[-1], initial[-1])
    assert float(torch.diff(result.q, dim=0).abs().max()) <= .5 + 1e-9


def test_slp_refinement_calls_exact_feasibility_gate():
    template = load_templates(ROOT / "reports/parametric_topology_audit.json",
                              device="cpu", dtype=torch.float64)["xarm6"]
    design = assemble_from_deltas(template, template.deltas)
    q = torch.zeros((3, 6), dtype=torch.float64)
    targets = fk_tcp(design, q)[0].detach()
    perturbed = q.clone(); perturbed[1, 0] = .1
    calls = []
    result = refine_joint_window_slp(
        design, targets, perturbed, iterations=2,
        feasibility_fn=lambda proposal: calls.append(proposal.clone()) or False)
    assert calls
    assert result.accepted_iterations == 0
    torch.testing.assert_close(result.q, perturbed)

