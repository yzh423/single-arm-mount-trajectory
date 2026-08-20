from factory_bimanual.spacing_scan import (
    SpacingCandidateRow,
    generate_spacing_candidates,
    rank_spacing_candidates,
)


def row(spacing, task, coverage, failure=0.0, cross=0, table=0, error=0.0):
    return SpacingCandidateRow(
        spacing_m=spacing,
        task_name=task,
        synchronous_coverage=coverage,
        longest_failure_s=failure,
        cross_arm_collision_frames=cross,
        table_base_collision_frames=table,
        aggregate_tcp_error=error,
    )


def test_worse_task_dominates_mean_and_selection_is_shared():
    rows = [
        row(.70, "screw", .99), row(.70, "pour", .50),
        row(.80, "screw", .70), row(.80, "pour", .70),
    ]
    selected = rank_spacing_candidates(rows)
    assert selected.spacing_m == .80
    assert selected.task_names == ("pour", "screw")


def test_collision_precedes_error_and_smaller_spacing_breaks_exact_tie():
    rows = [
        row(.70, "a", .9, cross=1, error=.001), row(.70, "b", .9, error=.001),
        row(.80, "a", .9, error=100), row(.80, "b", .9, error=100),
    ]
    assert rank_spacing_candidates(rows).spacing_m == .80

    tied = [row(.8, "a", .9), row(.8, "b", .9), row(.7, "a", .9), row(.7, "b", .9)]
    decision = rank_spacing_candidates(tied)
    assert decision.spacing_m == .7
    assert decision.tie_count == 2


def test_deterministic_coarse_to_local_candidates():
    assert generate_spacing_candidates(.6, 1.0, .2) == (.6, .8, 1.0)
    assert generate_spacing_candidates(.6, 1.0, .2, finalists=(.8,), local_step_m=.05) == (
        .6, .75, .8, .85, 1.0
    )
