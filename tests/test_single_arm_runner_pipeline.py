from scripts.run_twelve_arm_two_single_tasks import _downstream_stages


def test_successful_search_renders_without_redundant_solve():
    assert _downstream_stages(
        search_returncode=0, solution_current=False, cache_exists=True,
        solution_passed=True, render_current=False) == (False, True)


def test_failed_search_cannot_solve_or_render_stale_cache():
    assert _downstream_stages(
        search_returncode=1, solution_current=False, cache_exists=True,
        solution_passed=True, render_current=False) == (False, False)


def test_current_solution_can_render_without_new_search():
    assert _downstream_stages(
        search_returncode=0, solution_current=True, cache_exists=True,
        solution_passed=True, render_current=False) == (False, True)


def test_failed_episode_is_rendered_for_diagnostics_when_cache_is_current():
    assert _downstream_stages(
        search_returncode=2, solution_current=False, cache_exists=True,
        solution_passed=False, render_current=False) == (False, True)
