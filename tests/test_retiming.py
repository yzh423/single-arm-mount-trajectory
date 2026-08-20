import torch

from design_optimization.retiming import retime_joint_path


def test_retiming_preserves_path_and_enforces_per_axis_velocity():
    q = torch.tensor([[0., 0.], [.3, .1], [.5, -.3]])
    nominal = torch.tensor([0., .1, .2])
    result = retime_joint_path(q, nominal, torch.tensor([1., 2.]))
    velocity = torch.diff(q, dim=0).abs() / result.segment_dt_s[:, None]
    assert torch.all(velocity <= torch.tensor([1., 2.]) + 1e-6)
    assert torch.all(torch.diff(result.time_s) >= torch.diff(nominal))
    assert torch.all((result.speed_scale > 0) & (result.speed_scale <= 1))


def test_retiming_rejects_nonpositive_limits():
    q = torch.zeros((2, 2)); time = torch.tensor([0., .1])
    try:
        retime_joint_path(q, time, torch.tensor([1., 0.]))
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive velocity limit was accepted")
