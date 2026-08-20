from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.collision_classifier import ConfigurationCollisionClassifier, binary_metrics
from design_optimization.experiment import save_checkpoint_atomic, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("doosan", "xarm6", "ur5", "kinova"), required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "reports" / "learning" / "collision_calibration")
    args = parser.parse_args(); seed_everything(20260806); device = torch.device(args.device)
    data = np.load(args.input_dir / f"{args.robot}_vendor_self_collision.npz")
    q = torch.as_tensor(data["q"], dtype=torch.float32, device=device)
    label = torch.as_tensor(data["collision"], dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(20260806)
    permutation = torch.randperm(len(q), generator=generator, device=device)
    n_train, n_validation = int(.6*len(q)), int(.2*len(q))
    train = permutation[:n_train]; validation = permutation[n_train:n_train+n_validation]
    test = permutation[n_train+n_validation:]
    model = ConfigurationCollisionClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    positive = label[train].sum().clamp_min(1); negative = len(train)-positive
    for _ in range(args.epochs):
        logits = model(q[train]); loss = F.binary_cross_entropy_with_logits(
            logits, label[train], pos_weight=negative/positive)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        validation_probability = model(q[validation]).sigmoid()
        positive_probability = validation_probability[label[validation].bool()]
        # Threshold chosen on validation for 99% positive recall, then frozen.
        threshold = (torch.quantile(positive_probability, .01) if positive_probability.numel()
                     else torch.tensor(.5, device=device))
        test_probability = model(q[test]).sigmoid()
        metrics = binary_metrics(label[test], test_probability >= threshold)
        default_metrics = binary_metrics(label[test], test_probability >= .5)
    output = args.input_dir / f"{args.robot}_collision_classifier.pt"
    save_checkpoint_atomic(output, {"state_dict": model.state_dict(), "threshold": threshold,
                                    "seed": 20260806, "robot": args.robot})
    report = {"robot": args.robot, "samples": len(q), "train_samples": len(train),
              "validation_samples": len(validation), "test_samples": len(test),
              "epochs": args.epochs, "validation_selected_threshold": float(threshold),
              "test_metrics": metrics, "default_threshold_test_metrics": default_metrics,
              "scope": "fixed vendor morphology only; not valid across changed link geometry"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()

