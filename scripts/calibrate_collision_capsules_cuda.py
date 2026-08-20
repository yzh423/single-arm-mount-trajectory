from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.collision import self_capsule_clearance, self_segment_pair_distances
from design_optimization.kinematics import assemble_from_deltas, fk_tcp
from design_optimization.topology import load_templates


def rigid_alignment_rmse(source: torch.Tensor, target: torch.Tensor) -> float:
    source_center = source.mean(0); target_center = target.mean(0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vh = torch.linalg.svd(covariance); rotation = vh.T @ u.T
    if torch.linalg.det(rotation) < 0:
        vh = vh.clone(); vh[-1] *= -1; rotation = vh.T @ u.T
    aligned = (source - source_center) @ rotation.T + target_center
    return float(torch.sqrt(torch.mean(torch.sum((aligned - target) ** 2, dim=1))))


def classification_metrics(label: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    tp = ((label == 1) & (prediction == 1)).sum().float()
    tn = ((label == 0) & (prediction == 0)).sum().float()
    fp = ((label == 0) & (prediction == 1)).sum().float()
    fn = ((label == 1) & (prediction == 0)).sum().float()
    precision = tp / (tp + fp).clamp_min(1); recall = tp / (tp + fn).clamp_min(1)
    specificity = tn / (tn + fp).clamp_min(1)
    return {"accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "balanced_accuracy": float(.5 * (recall + specificity)),
            "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall).clamp_min(1e-8)),
            "false_positive_rate": float(fp / (fp + tn).clamp_min(1)),
            "false_negative_rate": float(fn / (fn + tp).clamp_min(1)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit topology-specific capsule radii to MuJoCo mesh labels")
    parser.add_argument("--robot", choices=("doosan", "xarm6", "ur5", "kinova"), required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "learning" / "collision_calibration")
    args = parser.parse_args(); reference = np.load(args.output_dir /
                                                     f"{args.robot}_vendor_self_collision.npz")
    device = torch.device(args.device); template = load_templates(
        ROOT / "reports" / "parametric_topology_audit.json", device=device)[args.robot]
    design = assemble_from_deltas(template, template.deltas)
    q = torch.as_tensor(reference["q"], dtype=torch.float32, device=device).unsqueeze(0)
    label = torch.as_tensor(reference["collision"], dtype=torch.float32, device=device)
    predicted_tcp = fk_tcp(design, q[0])[0, :, :3, 3].detach()
    target_tcp = torch.as_tensor(reference["tcp_local_xyz"], dtype=torch.float32, device=device)
    fk_rmse = rigid_alignment_rmse(predicted_tcp, target_tcp)
    generator = torch.Generator(device=device).manual_seed(20260806)
    permutation = torch.randperm(label.numel(), generator=generator, device=device)
    train_count = int(.6 * label.numel()); validation_count = int(.2 * label.numel())
    train = permutation[:train_count]
    validation = permutation[train_count:train_count + validation_count]
    test = permutation[train_count + validation_count:]
    all_distances, pairs = self_segment_pair_distances(design, q)
    raw = torch.zeros(len(pairs), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=.04)
    for _ in range(args.iterations):
        thresholds = .015 + .185 * torch.sigmoid(raw)
        clearance = (all_distances[0, train] - thresholds).amin(dim=-1)
        positive = label[train].sum().clamp_min(1); negative = len(train) - positive
        loss = F.binary_cross_entropy_with_logits(-clearance / .004, label[train],
                                                   pos_weight=negative / positive)
        # Weak size prior prevents classification from inflating every link to
        # explain a small number of mesh contacts.
        loss = loss + .02 * ((thresholds - .08) / .08).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    thresholds = .015 + .185 * torch.sigmoid(raw.detach())
    with torch.no_grad():
        validation_clearance = (all_distances[0, validation] - thresholds).amin(dim=-1)
        positive_validation = validation_clearance[label[validation].bool()]
        # A safety proxy is tuned for recall, not raw accuracy. The threshold is
        # chosen on validation only and then frozen for the held-out test set.
        safety_margin = (torch.quantile(positive_validation, .95) if positive_validation.numel()
                         else torch.tensor(0., device=device))
        clearance = (all_distances[0, test] - thresholds).amin(dim=-1)
        raw_metrics = classification_metrics(label[test].bool(), clearance < 0)
        safety_metrics = classification_metrics(label[test].bool(), clearance < safety_margin)
    result = {"robot": args.robot, "samples": int(label.numel()), "train_samples": len(train),
              "validation_samples": len(validation), "test_samples": len(test),
              "seed": 20260806, "iterations": args.iterations,
              "fitted_pair_thresholds_m": {f"{a}-{b}": float(value) for (a, b), value in
                                             zip(pairs, thresholds.cpu())},
              "validation_selected_safety_margin_m": float(safety_margin),
              "raw_zero_margin_test_metrics": raw_metrics,
              "test_metrics": safety_metrics,
              "fk_rigid_alignment_rmse_mm": 1000 * fk_rmse,
              "warning": ("kinematic frames do not agree; do not use calibration"
                          if fk_rmse > .01 else None)}
    path = args.output_dir / f"{args.robot}_capsule_calibration.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
