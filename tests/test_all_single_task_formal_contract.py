from scripts.run_twelve_arm_all_single_tasks import TASKS
from scripts.run_twelve_arm_two_single_tasks import ROBOTS


def test_formal_matrix_contains_twelve_arms_and_only_pure_single_hand_tasks():
    assert len(ROBOTS) == 12
    assert len(TASKS) == 12
    assert len(ROBOTS) * len(TASKS) == 144
    assert not any("dual" in task for task in TASKS)

