import numpy as np

from factory_bimanual.branch_equivalence import compare_initial_branches


def arms(left, right):
    return (np.asarray(left, dtype=float), np.asarray(right, dtype=float))


def test_periodic_wrapping_and_both_arms_below_five_collapses_to_a():
    a = arms([np.pi - .01], [0.0])
    b = arms([-np.pi + .01], [np.deg2rad(4.999)])
    decision = compare_initial_branches(a, b, periodic=arms([True], [True]))
    assert decision.equivalent
    assert decision.required_mpc_modes == ("canonical_a",)


def test_exact_five_degrees_splits_runs():
    decision = compare_initial_branches(
        arms([0], [0]), arms([np.deg2rad(5)], [0]), periodic=arms([True], [True])
    )
    assert not decision.equivalent
    assert decision.required_mpc_modes == ("mpc_a", "mpc_easyik")


def test_initializer_failure_policy():
    easy = arms([0], [0])
    assert compare_initial_branches(None, easy).required_mpc_modes == ("mpc_easyik",)
    assert compare_initial_branches(easy, None).required_mpc_modes == ("mpc_a",)
    assert compare_initial_branches(None, None).required_mpc_modes == ()
