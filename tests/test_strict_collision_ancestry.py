import numpy as np

from scripts.solve_strict_urdf_task_cache import active_chain_distance, active_ancestor_map


def test_fixed_intermediate_bodies_map_to_nearest_actuated_link():
    # world -> joint body 1 -> fixed body 2 -> joint body 3 -> fixed gripper 4
    parents = np.asarray([0, 0, 1, 2, 3])

    mapped = active_ancestor_map(parents, {1, 3})

    assert mapped == {0: 0, 1: 1, 2: 1, 3: 3, 4: 3}


def test_active_chain_distance_counts_actuated_links_through_fixed_bodies():
    parents = np.asarray([0, 0, 1, 2, 3, 4])
    mapped = active_ancestor_map(parents, {1, 3, 5})

    assert active_chain_distance(1, 5, parents, mapped) == 2
