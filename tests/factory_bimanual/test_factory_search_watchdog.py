from scripts.watch_piperx_factory_per_task_mount_search import (
    should_restart, status_payload,
)


def test_watchdog_restarts_unfinished_crash_within_retry_budget():
    assert should_restart(exit_code=-9, batch_status="planned",
                          restart_count=2, max_restarts=8)


def test_watchdog_stops_after_complete_or_retry_budget():
    assert not should_restart(exit_code=0, batch_status="complete",
                              restart_count=0, max_restarts=8)
    assert not should_restart(exit_code=-9, batch_status="planned",
                              restart_count=8, max_restarts=8)


def test_status_payload_is_machine_readable():
    payload = status_payload(3, 8, -9, "planned")
    assert payload["restart_count"] == 3
    assert payload["last_exit_code"] == -9
    assert payload["batch_status"] == "planned"
