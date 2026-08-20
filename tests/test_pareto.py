import torch

from design_optimization.pareto import nsga2_select, rank_population


def test_nondominated_sort_and_constraint_priority():
    objectives = torch.tensor(((1., 4.), (2., 2.), (4., 1.), (3., 3.), (.1, .1)))
    violation = torch.tensor((0., 0., 0., 0., .5))
    ranking = rank_population(objectives, violation)
    assert set(ranking.fronts[0].tolist()) == {0, 1, 2}
    assert ranking.rank[3] > 0
    assert ranking.rank[4] > 0  # infeasible cannot dominate feasible points
    selected, _ = nsga2_select(objectives, 3, violation)
    assert set(selected.tolist()) == {0, 1, 2}
