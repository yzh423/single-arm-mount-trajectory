import torch

from design_optimization.branch_policy import MultiBranchIKPolicy, branch_imitation_loss


def test_multibranch_policy_shapes_and_backward():
    model = MultiBranchIKPolicy(modes=6, width=48)
    q = torch.zeros((7, 6)); target = torch.eye(4).repeat(7, 1, 1)
    target[:, 0, 3] = torch.linspace(-.2, .2, 7)
    topology = torch.nn.functional.one_hot(torch.arange(7) % 4, 4).float()
    output = model(q, target, topology)
    assert output.candidates_q.shape == (7, 6, 6)
    assert output.logits.shape == (7, 6)
    loss = branch_imitation_loss(output, torch.ones((7, 6)) * .2)
    assert torch.isfinite(loss); loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_policy_rejects_bad_target_shape():
    model = MultiBranchIKPolicy()
    try:
        model(torch.zeros(6), torch.eye(3), torch.tensor((1., 0., 0., 0.)))
    except ValueError:
        return
    raise AssertionError("invalid transform shape was accepted")
