from __future__ import annotations

import torch
from torch import nn


def periodic_joint_features(q: torch.Tensor) -> torch.Tensor:
    """Continuous representation across +/-pi and multi-turn joint wrap."""
    return torch.cat((torch.sin(q), torch.cos(q)), dim=-1)


class ConfigurationCollisionClassifier(nn.Module):
    """Differentiable narrow phase for one fixed, mesh-certified morphology."""

    def __init__(self, width: int = 128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(12, width), nn.SiLU(), nn.LayerNorm(width),
                                     nn.Linear(width, width), nn.SiLU(),
                                     nn.Linear(width, width // 2), nn.SiLU(),
                                     nn.Linear(width // 2, 1))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.network(periodic_joint_features(q)).squeeze(-1)


def binary_metrics(label: torch.Tensor, prediction: torch.Tensor) -> dict[str, float | int]:
    label, prediction = label.bool(), prediction.bool()
    tp = (label & prediction).sum().float(); tn = (~label & ~prediction).sum().float()
    fp = (~label & prediction).sum().float(); fn = (label & ~prediction).sum().float()
    precision = tp / (tp + fp).clamp_min(1); recall = tp / (tp + fn).clamp_min(1)
    specificity = tn / (tn + fp).clamp_min(1)
    return {"accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "balanced_accuracy": float(.5 * (recall + specificity)),
            "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall).clamp_min(1e-8)),
            "false_positive_rate": float(fp / (fp + tn).clamp_min(1)),
            "false_negative_rate": float(fn / (fn + tp).clamp_min(1)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

