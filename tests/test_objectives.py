from pathlib import Path

import torch

from design_optimization.kinematics import build_designs, deterministic_joint_samples
from design_optimization.objectives import motor_clearance_violation, topology_intersection_error
from design_optimization.topology import load_templates


def test_native_uniform_designs_preserve_declared_intersections():
    root = Path(__file__).resolve().parents[1]
    templates = load_templates(root / "reports" / "parametric_topology_audit.json", dtype=torch.float64)
    for template in templates.values():
        designs = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                                deterministic_joint_samples(template, 2048))
        assert float(topology_intersection_error(designs)[0]) < 1e-4


def test_motor_clearance_detects_collapsed_ur_wrist():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["ur5"]
    deltas = template.deltas.clone(); deltas[5] *= 0.1
    from design_optimization.kinematics import assemble_from_deltas
    assert float(motor_clearance_violation(assemble_from_deltas(template, deltas))) > 0.04


def test_kinova_uses_relative_vendor_scale_not_57mm_everywhere():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["kinova"]
    from design_optimization.kinematics import assemble_from_deltas
    compliant = template.deltas.clone(); compliant[2:5] *= 0.5
    collapsed = template.deltas.clone(); collapsed[2:5] *= 0.2
    assert float(motor_clearance_violation(assemble_from_deltas(template, compliant))) == 0.0
    assert float(motor_clearance_violation(assemble_from_deltas(template, collapsed))) > 0.05
