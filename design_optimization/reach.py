from __future__ import annotations

import numpy as np
import torch

from .kinematics import fk_flange
from .topology import DesignBatch


def _single_design(designs: DesignBatch, index: int) -> DesignBatch:
    return DesignBatch(
        designs.template,
        designs.deltas[index:index + 1],
        designs.points[index:index + 1],
        designs.home[index:index + 1],
        designs.reach_scale[index:index + 1],
    )


def numerical_max_flange_reach(
    designs: DesignBatch,
    index: int = 0,
    *,
    seed: int = 20260814,
    population: int = 256,
    generations: int = 300,
    sobol_samples: int = 131072,
) -> tuple[float, np.ndarray]:
    """CUDA Sobol global exploration followed by gradient multi-start polish.

    This deliberately stays inside PyTorch: loading SciPy and CUDA PyTorch in
    one Windows process can load two incompatible Intel OpenMP runtimes.
    Final candidates are independently rechecked with a different scrambled
    Sobol sequence after uniform correction.
    """
    design = _single_design(designs, index)
    template = design.template
    device, dtype = design.deltas.device, design.deltas.dtype
    lo, hi = template.q_min, template.q_max
    engine = torch.quasirandom.SobolEngine(6, scramble=True, seed=seed)
    best_value = torch.tensor(-1.0, dtype=dtype, device=device)
    best_q = torch.zeros(6, dtype=dtype, device=device)
    top_values, top_q = [], []
    remaining = sobol_samples
    with torch.no_grad():
        while remaining:
            count = min(32768, remaining)
            unit = engine.draw(count).to(device=device, dtype=dtype)
            q = lo + unit * (hi - lo)
            xyz = fk_flange(design, q)[0, :, :3, 3]
            squared = (xyz * xyz).sum(dim=-1)
            keep = min(population, count)
            values, indices = torch.topk(squared, keep)
            top_values.append(values); top_q.append(q[indices])
            remaining -= count
        values = torch.cat(top_values)
        q_pool = torch.cat(top_q)
        _, indices = torch.topk(values, min(population, values.numel()))
        q_initial = q_pool[indices]

    # Bounded variables through a sigmoid avoid clipping-induced false optima.
    fraction = ((q_initial - lo) / (hi - lo)).clamp(1e-7, 1.0 - 1e-7)
    raw = torch.logit(fraction).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=0.055)
    for iteration in range(generations):
        q = lo + torch.sigmoid(raw) * (hi - lo)
        xyz = fk_flange(design, q)[0, :, :3, 3]
        squared = (xyz * xyz).sum(dim=-1)
        # Each branch ascends independently; sum only batches autograd work.
        loss = -squared.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration in (generations // 2, (3 * generations) // 4):
            for group in optimizer.param_groups:
                group["lr"] *= 0.25
    with torch.no_grad():
        q = lo + torch.sigmoid(raw) * (hi - lo)
        xyz = fk_flange(design, q)[0, :, :3, 3]
        squared = (xyz * xyz).sum(dim=-1)
        best = torch.argmax(squared)
        best_value = squared[best]
        best_q = q[best]
    return float(torch.sqrt(best_value).cpu()), best_q.detach().cpu().numpy()

