import torch
from design_optimization.collision_classifier import (ConfigurationCollisionClassifier,
                                                       binary_metrics, periodic_joint_features)


def test_periodic_features_are_wrap_invariant():
    q = torch.randn(5, 6)
    torch.testing.assert_close(periodic_joint_features(q),
                               periodic_joint_features(q + 2*torch.pi), atol=1e-6, rtol=1e-6)


def test_collision_classifier_and_metrics():
    model = ConfigurationCollisionClassifier(width=32); logits = model(torch.randn(7, 6))
    assert logits.shape == (7,)
    metrics = binary_metrics(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1, 1]))
    assert metrics["tp"] == 2 and metrics["fp"] == 1 and metrics["fn"] == 0
