import pytest

from scripts.run_thirteen_arm_dense_search import select_robots


def test_dense_search_can_limit_formal_run_to_requested_robots():
    templates = {"xarm6": object(), "i2rt_yam": object(), "ur5": object()}
    selected = select_robots(templates, ["xarm6", "i2rt_yam"])
    assert list(selected) == ["xarm6", "i2rt_yam"]


def test_dense_search_rejects_unknown_robot():
    with pytest.raises(ValueError, match="missing robots"):
        select_robots({"xarm6": object()}, ["missing"])
