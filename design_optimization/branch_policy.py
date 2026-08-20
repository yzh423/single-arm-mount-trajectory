from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BranchPolicyOutput:
    candidates_q: torch.Tensor
    logits: torch.Tensor


class MultiBranchIKPolicy(nn.Module):
    """Small conditional proposal network for the asynchronous branch worker.

    It does not replace constrained IK/MPC.  It proposes several joint-space
    modes; the exact solver and collision checker certify them before the 50 Hz
    servo may consume a proposal.
    """

    def __init__(self, topology_count: int = 4, modes: int = 8, width: int = 192):
        super().__init__(); self.modes = modes
        # current q(6), target translation(3), target rotation first 2 columns(6), topology one-hot
        input_dim = 15 + topology_count
        self.network = nn.Sequential(
            nn.Linear(input_dim, width), nn.SiLU(), nn.LayerNorm(width),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, modes * 7))

    def forward(self, current_q: torch.Tensor, target: torch.Tensor,
                topology_one_hot: torch.Tensor) -> BranchPolicyOutput:
        if current_q.shape[-1] != 6 or target.shape[-2:] != (4, 4):
            raise ValueError("expected current_q [...,6] and target [...,4,4]")
        rotation_6d = target[..., :3, :2].transpose(-1, -2).reshape(target.shape[:-2] + (6,))
        features = torch.cat((current_q, target[..., :3, 3], rotation_6d,
                              topology_one_hot), dim=-1)
        raw = self.network(features).reshape(features.shape[:-1] + (self.modes, 7))
        # A mode may move at most pi from the current wrapped branch.  The
        # downstream exact solver can still cross a wrap using its joint limits.
        candidates = current_q[..., None, :] + torch.pi * torch.tanh(raw[..., :6])
        return BranchPolicyOutput(candidates, raw[..., 6])


def wrapped_joint_error(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = candidate - target
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def branch_imitation_loss(output: BranchPolicyOutput, target_q: torch.Tensor,
                          *, diversity_weight: float = .02) -> torch.Tensor:
    """Winner-take-all imitation with mode classification and anti-collapse."""
    error = wrapped_joint_error(output.candidates_q, target_q[..., None, :])
    per_mode = error.square().mean(dim=-1)
    best = per_mode.detach().argmin(dim=-1)
    regression = per_mode.gather(-1, best[..., None]).mean()
    classification = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]),
                                     best.reshape(-1))
    if output.candidates_q.shape[-2] > 1:
        pair = wrapped_joint_error(output.candidates_q[..., :, None, :],
                                   output.candidates_q[..., None, :, :])
        distance = pair.square().mean(dim=-1)
        mask = ~torch.eye(distance.shape[-1], dtype=torch.bool,
                          device=distance.device)
        diversity = torch.exp(-distance[..., mask]).mean()
    else:
        diversity = regression.new_zeros(())
    return regression + .1 * classification + diversity_weight * diversity
