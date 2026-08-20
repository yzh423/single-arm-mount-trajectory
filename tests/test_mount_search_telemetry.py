from scripts.mount_search_telemetry import SearchTelemetry


def test_telemetry_aggregates_candidates_frames_restarts_and_iterations():
    telemetry = SearchTelemetry()
    telemetry.record_candidate("rank", outcome="evaluated", new_frames=16, reused_frames=0)
    telemetry.record_candidate("medium", outcome="promoted", new_frames=48, reused_frames=16)
    telemetry.record_ik("medium", first_frame=True, restarts=2, iterations=73, success=True)

    payload = telemetry.to_dict()
    assert payload["stages"]["rank"]["evaluated_candidates"] == 1
    assert payload["stages"]["medium"]["promoted_candidates"] == 1
    assert payload["stages"]["medium"]["new_frames"] == 48
    assert payload["stages"]["medium"]["reused_frames"] == 16
    assert payload["stages"]["medium"]["first_frame_restart_histogram"]["1-2"] == 1
    assert payload["stages"]["medium"]["dls_iterations"] == 73


def test_stage_context_records_monotonic_elapsed_time(monkeypatch):
    ticks = iter((10.0, 12.5))
    monkeypatch.setattr("scripts.mount_search_telemetry.time.perf_counter", lambda: next(ticks))
    telemetry = SearchTelemetry()
    with telemetry.stage("selection"):
        pass
    assert telemetry.to_dict()["stages"]["selection"]["elapsed_s"] == 2.5


def test_telemetry_rejects_negative_counters():
    telemetry = SearchTelemetry()
    try:
        telemetry.record_candidate("rank", outcome="evaluated", new_frames=-1, reused_frames=0)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative counters must be rejected")
