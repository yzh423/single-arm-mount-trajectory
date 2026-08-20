from pathlib import Path

import torch

from design_optimization.collision import (segment_segment_distance, self_capsule_clearance,
                                           self_segment_pair_distances)
from design_optimization.kinematics import build_designs, deterministic_joint_samples
from design_optimization.topology import load_templates


def test_segment_distance_parallel_and_crossing():
    dtype = torch.float64
    p0 = torch.tensor((0., 0., 0.), dtype=dtype); p1 = torch.tensor((1., 0., 0.), dtype=dtype)
    q0 = torch.tensor((0., 2., 0.), dtype=dtype); q1 = torch.tensor((1., 2., 0.), dtype=dtype)
    torch.testing.assert_close(segment_segment_distance(p0, p1, q0, q1), torch.tensor(2., dtype=dtype))
    q0 = torch.tensor((.5, -1., 0.), dtype=dtype); q1 = torch.tensor((.5, 1., 0.), dtype=dtype)
    torch.testing.assert_close(segment_segment_distance(p0, p1, q0, q1), torch.tensor(0., dtype=dtype), atol=1e-10, rtol=0)


def test_self_clearance_is_finite_for_all_topologies():
    root = Path(__file__).resolve().parents[1]
    for template in load_templates(root / "reports" / "parametric_topology_audit.json",
                                   dtype=torch.float64).values():
        design = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                               deterministic_joint_samples(template, 1024))
        q = torch.zeros((1, 3, 6), dtype=torch.float64)
        clearance = self_capsule_clearance(design, q)
        assert clearance.shape == (1, 3)
        assert torch.isfinite(clearance).all()


def test_ur_fixed_wrist_topology_pair_is_excluded():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["ur5"]
    design = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                           deterministic_joint_samples(template, 1024))
    _, pairs = self_segment_pair_distances(design, torch.zeros((1, 1, 6), dtype=torch.float64))
    assert (2, 4) not in pairs
    assert (2, 5) in pairs
