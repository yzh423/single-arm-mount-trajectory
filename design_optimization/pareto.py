from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ParetoRanking:
    rank: torch.Tensor
    crowding: torch.Tensor
    fronts: list[torch.Tensor]


def constraint_dominates(a: int, b: int, objectives: torch.Tensor,
                         violation: torch.Tensor) -> bool:
    va, vb = float(violation[a]), float(violation[b])
    if va <= 0 < vb: return True
    if vb <= 0 < va: return False
    if va > 0 and vb > 0: return va < vb
    return bool(torch.all(objectives[a] <= objectives[b]) and torch.any(objectives[a] < objectives[b]))


def nondominated_sort(objectives: torch.Tensor,
                      violation: torch.Tensor | None = None) -> list[torch.Tensor]:
    """Constraint-aware NSGA-II nondominated sorting."""
    count = objectives.shape[0]
    if violation is None: violation = torch.zeros(count, device=objectives.device)
    dominates_set = [[] for _ in range(count)]
    dominated_count = [0] * count
    first = []
    for p in range(count):
        for q in range(count):
            if p == q: continue
            if constraint_dominates(p, q, objectives, violation): dominates_set[p].append(q)
            elif constraint_dominates(q, p, objectives, violation): dominated_count[p] += 1
        if dominated_count[p] == 0: first.append(p)
    fronts, current = [], first
    while current:
        fronts.append(torch.tensor(current, dtype=torch.long, device=objectives.device))
        following = []
        for p in current:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0: following.append(q)
        current = following
    return fronts


def crowding_distance(objectives: torch.Tensor, front: torch.Tensor) -> torch.Tensor:
    size = front.numel(); distance = torch.zeros(size, dtype=objectives.dtype, device=objectives.device)
    if size <= 2:
        distance[:] = torch.inf
        return distance
    values = objectives[front]
    for column in range(values.shape[1]):
        order = torch.argsort(values[:, column])
        distance[order[0]] = torch.inf; distance[order[-1]] = torch.inf
        span = (values[order[-1], column] - values[order[0], column]).clamp_min(1e-12)
        interior = (values[order[2:], column] - values[order[:-2], column]) / span
        finite = ~torch.isinf(distance[order[1:-1]])
        distance[order[1:-1][finite]] += interior[finite]
    return distance


def rank_population(objectives: torch.Tensor, violation: torch.Tensor | None = None) -> ParetoRanking:
    fronts = nondominated_sort(objectives, violation)
    rank = torch.empty(objectives.shape[0], dtype=torch.long, device=objectives.device)
    crowding = torch.zeros(objectives.shape[0], dtype=objectives.dtype, device=objectives.device)
    for index, front in enumerate(fronts):
        rank[front] = index
        crowding[front] = crowding_distance(objectives, front)
    return ParetoRanking(rank, crowding, fronts)


def nsga2_select(objectives: torch.Tensor, count: int,
                 violation: torch.Tensor | None = None) -> tuple[torch.Tensor, ParetoRanking]:
    ranking = rank_population(objectives, violation)
    selected = []
    for front in ranking.fronts:
        remaining = count - len(selected)
        if front.numel() <= remaining:
            selected.extend(front.tolist())
        else:
            order = torch.argsort(ranking.crowding[front], descending=True)
            selected.extend(front[order[:remaining]].tolist())
            break
    return torch.tensor(selected, dtype=torch.long, device=objectives.device), ranking


def tournament_parents(ranking: ParetoRanking, count: int, generator: torch.Generator) -> torch.Tensor:
    population = ranking.rank.numel()
    pair = torch.randint(population, (count, 2), generator=generator, device=ranking.rank.device)
    a, b = pair[:, 0], pair[:, 1]
    a_better = ((ranking.rank[a] < ranking.rank[b]) |
                ((ranking.rank[a] == ranking.rank[b]) & (ranking.crowding[a] > ranking.crowding[b])))
    return torch.where(a_better, a, b)


def make_offspring(parameters: torch.Tensor, ranking: ParetoRanking,
                   generator: torch.Generator, mutation_sigma: float = 0.12) -> torch.Tensor:
    """Blend crossover plus annealable Gaussian mutation."""
    count = parameters.shape[0]
    parents = tournament_parents(ranking, 2 * count, generator).reshape(count, 2)
    alpha = torch.rand((count, 1), generator=generator, device=parameters.device,
                       dtype=parameters.dtype)
    children = alpha * parameters[parents[:, 0]] + (1 - alpha) * parameters[parents[:, 1]]
    noise = torch.randn(children.shape, generator=generator, device=parameters.device,
                        dtype=parameters.dtype)
    mutate = torch.rand(children.shape, generator=generator, device=parameters.device) < (1 / children.shape[1])
    return (children + mutation_sigma * noise * mutate).clamp(-2.5, 2.5)
