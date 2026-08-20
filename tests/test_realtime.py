import torch

from design_optimization.realtime import (BranchProposal, BranchProposalMailbox,
                                          RealtimeBudget, ServoMode, ServoObservation,
                                          govern_cartesian_target, schedule_bounded_solve)


def test_nominal_schedule_is_50hz_deadline_bounded():
    schedule = schedule_bounded_solve(ServoObservation(.01, .08, .08, .004))
    assert schedule.mode is ServoMode.TRACK
    assert schedule.deadline_s < 0.02
    assert schedule.local_iterations == 3


def test_difficult_frame_requests_async_search_without_blocking_servo():
    schedule = schedule_bounded_solve(ServoObservation(.01, .01, .02, .004))
    assert schedule.mode is ServoMode.CAUTION
    assert schedule.request_async_branch_search
    assert schedule.target_velocity_scale < 1


def test_collision_or_stale_target_holds_certified_command():
    assert schedule_bounded_solve(ServoObservation(.01, .1, .001, .001)).mode is ServoMode.HOLD
    assert schedule_bounded_solve(ServoObservation(.2, .1, .1, .001)).mode is ServoMode.HOLD


def test_cartesian_target_governor_bounds_motion_and_holds():
    current = torch.eye(4); requested = torch.eye(4)
    requested[0, 3] = 1.0
    angle = torch.tensor(torch.pi / 2)
    requested[:3, :3] = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.],
                                      [torch.sin(angle), torch.cos(angle), 0.], [0., 0., 1.]])
    track = schedule_bounded_solve(ServoObservation(.01, .08, .08, .004))
    governed = govern_cartesian_target(current, requested, .02, track,
                                       maximum_linear_m_s=.5, maximum_angular_rad_s=2.)
    assert torch.allclose(governed[0, 3], torch.tensor(.01), atol=1e-6)
    angular_step = torch.acos(((torch.trace(governed[:3, :3]) - 1) / 2).clamp(-1, 1))
    assert torch.allclose(angular_step, torch.tensor(.04), atol=1e-5)
    hold = schedule_bounded_solve(ServoObservation(.2, .1, .1, .001))
    assert torch.equal(govern_cartesian_target(current, requested, .02, hold), current)


def test_branch_mailbox_is_monotonic_and_rejects_stale_result():
    mailbox = BranchProposalMailbox()
    first = BranchProposal(2, 10., torch.ones(6), .04, .08)
    assert mailbox.publish(first)
    assert not mailbox.publish(BranchProposal(1, 11., torch.zeros(6), .1, .1))
    received = mailbox.latest(10.05, .1)
    assert received is not None and received.generation == 2
    assert received.q.data_ptr() != first.q.data_ptr()
    assert mailbox.latest(10.2, .1) is None
