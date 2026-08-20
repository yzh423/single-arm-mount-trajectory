from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.experiment import load_checkpoint, save_checkpoint_atomic, seed_everything
from design_optimization.surrogate import (ObjectiveSurrogate, ensemble_predict, gaussian_nll,
                                           propose_ucb_candidates, standardize_targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ensemble surrogate from a Pareto evaluation archive")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--proposals", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); seed_everything(20260806); device = torch.device(args.device)
    archive = load_checkpoint(args.archive, device)
    parameters = archive["parameters"].float(); objectives = archive["objectives"].float()
    violation = archive["violation"].float(); targets, target_center, target_scale = standardize_targets(
        objectives, violation)
    parameter_center = parameters.mean(0); parameter_scale = parameters.std(0).clamp_min(1e-6)
    normalized = (parameters - parameter_center) / parameter_scale
    generator = torch.Generator(device=device).manual_seed(20260806)
    permutation = torch.randperm(len(parameters), generator=generator, device=device)
    split = max(1, int(.8 * len(parameters))); train, validation = permutation[:split], permutation[split:]
    models = []
    for member in range(args.ensemble):
        model = ObjectiveSurrogate(objective_dim=targets.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        bootstrap = train[torch.randint(len(train), (len(train),), generator=generator, device=device)]
        for _ in range(args.epochs):
            loss = gaussian_nll(model, normalized[bootstrap], targets[bootstrap])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        models.append(model.eval())
    with torch.no_grad():
        prediction = ensemble_predict(models, normalized[validation])
        physical_prediction = prediction.mean * target_scale + target_center
        physical_target = targets[validation] * target_scale + target_center
        validation_rmse = torch.sqrt((physical_prediction - physical_target).square().mean(0))
    seed_pool = torch.randn((max(args.proposals, 64), parameters.shape[1]), generator=generator,
                            device=device)
    normalized_proposals = propose_ucb_candidates(models, seed_pool, count=args.proposals,
                                                   generator=generator)
    proposals = normalized_proposals * parameter_scale + parameter_center
    output = args.output or args.archive.parent / "surrogate_ensemble.pt"
    save_checkpoint_atomic(output, {"state_dicts": [model.state_dict() for model in models],
                                    "parameter_center": parameter_center, "parameter_scale": parameter_scale,
                                    "target_center": target_center, "target_scale": target_scale,
                                    "proposals": proposals, "source_archive": str(args.archive),
                                    "seed": 20260806})
    report = {"source_archive": str(args.archive), "samples": len(parameters),
              "train_samples": len(train), "validation_samples": len(validation),
              "ensemble": args.ensemble, "epochs": args.epochs,
              "validation_rmse": validation_rmse.cpu().tolist(), "output": str(output)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
