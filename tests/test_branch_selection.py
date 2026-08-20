from pathlib import Path

import torch

from design_optimization.ik import IKResult, select_continuous_branches
from design_optimization.kinematics import build_designs, deterministic_joint_samples
from design_optimization.topology import load_templates


def test_continuous_selector_rejects_alternating_instantaneous_branch():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["doosan"]
    design = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                           deterministic_joint_samples(template, 512))
    q = torch.zeros((1, 5, 2, 6), dtype=torch.float64)
    q[:, :, 1, 0] = 2.0
    pe = torch.tensor([[[0.001, 0.002], [0.002, 0.001], [0.001, 0.002],
                        [0.002, 0.001], [0.001, 0.002]]], dtype=torch.float64)
    oe = torch.zeros_like(pe)
    sigma = torch.ones_like(pe)
    success = torch.ones_like(pe, dtype=torch.bool)
    selected = select_continuous_branches(IKResult(q, pe, oe, sigma, success), design,
                                          continuity_weight=50.0,
                                          singularity_weight=0.0, joint_margin_weight=0.0)
    assert torch.unique(selected.branch_index).numel() == 1
