import pytest

from scripts.execute_piperx_seal_bag_orientation_shortlists import (
    CAMERA, attempt_stem, select_collision_free_attempt,
)


def test_comparison_video_uses_requested_front_camera():
    assert CAMERA[0] == 180.0


def _attempt(collisions, coverage):
    return {"summary": {"collision_frame_count": collisions,
                        "synchronous_strict_coverage": coverage}}


def test_execution_selection_never_promotes_colliding_diagnostic():
    with pytest.raises(RuntimeError, match="no collision-free finalist"):
        select_collision_free_attempt([_attempt(3, .9), _attempt(1, .8)])


def test_execution_selection_chooses_best_collision_free_coverage():
    safe_low = _attempt(0, .7)
    safe_high = _attempt(0, .8)
    assert select_collision_free_attempt([
        _attempt(1, .99), safe_low, safe_high]) is safe_high


def test_attempt_cache_path_is_bound_to_mount_fingerprint(tmp_path):
    first = {"xy": {"left": [0, 0], "right": [.6, 0]}}
    second = {"xy": {"left": [0, 0], "right": [.7, 0]}}
    assert attempt_stem(tmp_path, "upright_table", 1, first) != \
        attempt_stem(tmp_path, "upright_table", 1, second)
