from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from factory_bimanual.defaults import TASK_SPECS, derive_initial_registration
from factory_bimanual.preflight import run_preflight
from factory_bimanual.preflight import validate_spacing_evidence
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.source_data import load_factory_task


ROOT = Path(__file__).resolve().parents[2]


def test_default_tasks_are_exact_full_recordings_and_registration_is_shared():
    assert {spec.name for spec in TASK_SPECS} == {"screw_cap", "pour_raw_material"}
    for spec in TASK_SPECS:
        task = load_factory_task(spec.csv_path, spec.name)
        registration = derive_initial_registration(task)
        combined = np.vstack((task.left_position_m, task.right_position_m))
        transformed = combined @ registration.rotation_world_from_vr.T + registration.translation_world_m
        np.testing.assert_allclose(transformed[:, :2].mean(axis=0), 0.0, atol=1e-12)
        assert transformed[:, 2].min() == 0.90
        assert spec.registration(task).matrix.tolist() == registration.matrix.tolist()


def test_preflight_writes_truthful_not_ready_report_when_spacing_is_provisional():
    output = ROOT / "reports/factory_bimanual/preflight.json"
    report = run_preflight(ROOT, output_path=output)
    assert report["ready"] is False
    assert report["checks"]["full_source_rows"] is True
    assert report["checks"]["assets"] is True
    assert report["checks"]["scenes_compile"] is True
    assert report["checks"]["controllers_construct"] is True
    assert report["checks"]["workspace_feasible"] is False
    assert any("mathematically impossible" in item for item in report["blockers"])
    assert {row["task"] for row in report["workspace_audits"]} == {
        "screw_cap", "pour_raw_material"
    }
    assert report["checks"]["spacing_finalized"] is False
    assert "spacing" in " ".join(report["blockers"]).lower()
    assert json.loads(output.read_text(encoding="utf-8"))["ready"] is False


def test_preflight_output_must_stay_in_factory_report_tree(tmp_path):
    try:
        run_preflight(ROOT, output_path=ROOT / "reports" / "single_arm" / "bad.json")
    except ValueError as exc:
        assert "factory_bimanual" in str(exc)
    else:
        raise AssertionError("preflight accepted a single-arm output path")


def test_spacing_evidence_requires_valid_nonzero_exact_tasks_and_metrics(tmp_path):
    spacing = {name: .8 for name in ROBOT_CONTRACTS}
    evidence = tmp_path / "spacing"; evidence.mkdir()
    row = {"spacing_m": .8, "task": "screw_cap", "synchronous_coverage": .5,
           "longest_failure_s": 0., "cross_arm_collision_frames": 0,
           "table_base_collision_frames": 0, "aggregate_tcp_error": .1}
    for robot in spacing:
        payload = {"status": "valid_selection", "robot": robot, "selected_spacing_m": .8,
                   "candidates": [row, row | {"task": "pour_raw_material"}]}
        (evidence / f"{robot}.json").write_text(json.dumps(payload))
    ok, errors = validate_spacing_evidence(evidence, spacing)
    assert ok and not errors
    payload = json.loads((evidence / "xarm6.json").read_text()); payload["status"] = "invalid_all_zero"
    (evidence / "xarm6.json").write_text(json.dumps(payload))
    ok, errors = validate_spacing_evidence(evidence, spacing)
    assert not ok and any("invalid_all_zero" in error for error in errors)
