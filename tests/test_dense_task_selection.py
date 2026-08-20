import pytest

from scripts.run_thirteen_arm_dense_search import select_task_samples


def test_select_task_samples_keeps_both_hands_for_dual_task():
    rows = [
        {"task": "cap-left", "hand": "left"},
        {"task": "fold-towel-dual", "hand": "left"},
        {"task": "fold-towel-dual", "hand": "right"},
        {"task": "other", "hand": "right"},
    ]

    selected = select_task_samples(rows, ["cap-left", "fold-towel-dual"])

    assert selected == rows[:3]


def test_select_task_samples_rejects_unknown_task():
    with pytest.raises(ValueError, match="missing tasks: unknown"):
        select_task_samples([{"task": "cap-left"}], ["unknown"])
