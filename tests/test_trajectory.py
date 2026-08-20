from pathlib import Path

import pytest
import torch

from design_optimization.collision import table_capsule_clearance
from design_optimization.ik import IKResult
from design_optimization.kinematics import build_designs, deterministic_joint_samples
from design_optimization.trajectory import select_bimanual_collision_aware
from design_optimization.topology import load_templates


def test_bimanual_beam_search_returns_one_branch_per_frame():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["xarm6"]
    design = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                           deterministic_joint_samples(template, 1024))
    q = torch.zeros((1, 4, 2, 6), dtype=torch.float64)
    q[:, :, 1, 0] = .3
    error = torch.full((1, 4, 2), .001, dtype=torch.float64)
    result = IKResult(q, error, error, torch.ones_like(error), torch.ones_like(error, dtype=torch.bool))
    progress = []
    path = select_bimanual_collision_aware(
        result, result, design, (-.35, 0., .4), (.35, 0., .4), beam_width=3,
        progress_callback=lambda done, total: progress.append((done, total)))
    assert path.left_q.shape == (4, 6)
    assert path.right_branch.shape == (4,)
    assert torch.isfinite(path.total_cost)
    assert path.metrics()["joint_path_length_rad"] >= 0.0
    assert progress == [(index, 4) for index in range(5)]


def test_bimanual_table_clearance_uses_full_mount_transform():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["xarm6"]
    design = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                           deterministic_joint_samples(template, 1024))
    q = torch.zeros((1, 2, 1, 6), dtype=torch.float64)
    scalar = torch.zeros((1, 2, 1), dtype=torch.float64)
    result = IKResult(q, scalar, scalar, torch.ones_like(scalar),
                      torch.ones_like(scalar, dtype=torch.bool))
    angle = torch.tensor(torch.pi / 2, dtype=torch.float64)
    rotation_y = torch.tensor([[torch.cos(angle), 0., torch.sin(angle), -.35],
                               [0., 1., 0., 0.],
                               [-torch.sin(angle), 0., torch.cos(angle), .55],
                               [0., 0., 0., 1.]], dtype=torch.float64)
    right = rotation_y.clone(); right[0, 3] = .35
    path = select_bimanual_collision_aware(
        result, result, design, rotation_y, right, beam_width=1, collision_weight=0.0)
    expected_left = table_capsule_clearance(design, q[0, :, 0], rotation_y)[0]
    expected_right = table_capsule_clearance(design, q[0, :, 0], right)[0]
    torch.testing.assert_close(path.table_clearance_m,
                               torch.minimum(expected_left, expected_right))


def test_hard_collision_rejects_an_all_colliding_frame():
    root = Path(__file__).resolve().parents[1]
    template = load_templates(root / "reports" / "parametric_topology_audit.json",
                              dtype=torch.float64)["xarm6"]
    design = build_designs(template, torch.zeros((1, 6), dtype=torch.float64),
                           deterministic_joint_samples(template, 1024))
    q = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    scalar = torch.zeros((1, 1, 1), dtype=torch.float64)
    result = IKResult(q, scalar, scalar, torch.ones_like(scalar),
                      torch.ones_like(scalar, dtype=torch.bool))
    with pytest.raises(RuntimeError, match="no feasible bimanual branch"):
        select_bimanual_collision_aware(
            result, result, design, (-.1, 0., -.5), (.1, 0., -.5),
            beam_width=1, hard_collision=True)
