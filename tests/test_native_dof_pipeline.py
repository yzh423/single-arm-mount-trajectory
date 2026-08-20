import torch

from design_optimization.kinematics import (
    assemble_from_deltas,
    build_designs,
    deterministic_joint_samples,
    fk_flange,
    geometric_jacobian,
    joint_world_positions,
)
from design_optimization.topology import TopologyTemplate
from design_optimization.ik import deterministic_seeds, solve_multistart
from design_optimization.collision import self_segment_pair_distances


def _seven_dof_template() -> TopologyTemplate:
    return TopologyTemplate(
        name="seven_test",
        axes=torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0]]
        ),
        deltas=torch.tensor(
            [[0.0, 0.0, 0.1], [0.2, 0.0, 0.0], [0.2, 0.0, 0.0],
             [0.0, 0.0, 0.1], [0.1, 0.0, 0.0], [0.0, 0.0, 0.08],
             [0.0, 0.0, 0.07]]
        ),
        home_rotation=torch.eye(3),
        q_min=torch.full((7,), -1.5),
        q_max=torch.full((7,), 1.5),
        tool_length_m=0.0,
    )


def test_native_seven_dof_fk_jacobian_and_joint_points():
    template = _seven_dof_template()
    designs = build_designs(template, torch.zeros((2, 7)),
                            deterministic_joint_samples(template, 64))
    q = torch.zeros((5, 7))
    assert fk_flange(designs, q).shape == (2, 5, 4, 4)
    assert geometric_jacobian(designs, q, tcp=False).shape == (2, 5, 6, 7)
    assert joint_world_positions(designs, q).shape == (2, 5, 8, 3)


def test_six_dof_template_reports_native_dof_without_behavior_change():
    seven = _seven_dof_template()
    six = TopologyTemplate(
        seven.name,
        seven.axes[:6],
        seven.deltas[:6],
        seven.home_rotation,
        seven.q_min[:6],
        seven.q_max[:6],
        seven.tool_length_m,
    )
    assert six.dof == 6
    assert seven.dof == 7


def test_native_seven_dof_ik_and_collision_shapes():
    template = _seven_dof_template()
    designs = build_designs(template, torch.zeros((1, 7)),
                            deterministic_joint_samples(template, 64))
    seeds = deterministic_seeds(template, 3)
    targets = fk_flange(designs, seeds[:2])[0]
    result = solve_multistart(designs, targets, seeds, iterations=1)
    distances, pairs = self_segment_pair_distances(designs, result.q[:, :1])
    assert result.q.shape == (1, 2, 3, 7)
    assert distances.shape[-1] == len(pairs)


def test_base_to_first_joint_offset_is_preserved_in_fk():
    template = TopologyTemplate(
        name="offset_test",
        axes=torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        deltas=torch.tensor([[0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        home_rotation=torch.eye(3),
        q_min=torch.full((2,), -1.0),
        q_max=torch.full((2,), 1.0),
        tool_length_m=0.0,
        first_joint_origin_m=torch.tensor([0.0, 0.0, 0.25]),
    )
    design = assemble_from_deltas(template, template.deltas)
    flange = fk_flange(design, torch.zeros((1, 2)))[0, 0]
    torch.testing.assert_close(flange[:3, 3], torch.tensor([0.3, 0.0, 0.25]))
