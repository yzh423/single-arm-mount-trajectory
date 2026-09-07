import copy

import pytest

from scripts.build_piperx_controller_event_v4_report import (
    MODES,
    TASKS,
    validate_report_scope,
)


def _scopes():
    totals = {TASKS[0]: 1872, TASKS[1]: 755}
    prefix = {}
    full = {}
    for task, total in totals.items():
        for mode in MODES:
            prefix[(task, mode)] = {
                "controller_event_rows": 300,
                "controller_event_rows_total": total,
                "controller_events_not_solved": total - 300,
            }
        for mode in ("baseline", "upright_table"):
            full[(task, mode)] = {
                "controller_event_rows": total,
                "controller_event_rows_total": total,
                "controller_events_not_solved": 0,
            }
    return prefix, full


def test_report_scope_accepts_full_runs_and_exact_300_event_comparisons():
    prefix, full = _scopes()

    assert validate_report_scope(prefix, full)


def test_report_scope_rejects_prefix_presented_as_complete():
    prefix, full = _scopes()
    bad = copy.deepcopy(full)
    bad[(TASKS[1], "baseline")]["controller_event_rows"] = 300
    bad[(TASKS[1], "baseline")]["controller_events_not_solved"] = 455

    with pytest.raises(ValueError, match="complete-run scope"):
        validate_report_scope(prefix, bad)
