import json
from dataclasses import replace
from pathlib import Path

import pytest

from factory_bimanual.run_experiment import ExperimentConfig, FactoryBimanualExperiment
import factory_bimanual.run_experiment as run_experiment_module
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS


def test_dry_run_has_exact_mandatory_matrix_and_no_execution(tmp_path: Path):
    called = []
    config = ExperimentConfig(output_root=tmp_path / "reports/factory_bimanual", dry_run=True)
    manifest = FactoryBimanualExperiment(config, ik_executor=lambda job: called.append(job)).run()
    ik = [j for j in manifest["jobs"] if j["stage"] == "ik"]
    assert len(ik) == 16
    assert {j["robot"] for j in ik} == {"xarm6", "franka_panda", "i2rt_yam", "ur5"}
    assert {j["task"] for j in ik} == {"screw_cap", "pour_raw_material"}
    assert {j["mode"] for j in ik} == {"strict_a", "easyik"}
    assert called == []
    assert all(j["status"] == "planned" for j in ik)


def test_conditional_mpc_schedule_and_failure_continuation(tmp_path: Path):
    def ik(job):
        if job.robot == "ur5" and job.mode == "easyik":
            raise RuntimeError("synthetic")
        delta = 1.0 if job.robot == "xarm6" else 7.0
        return {"initializer_valid": True, "left_delta_deg": delta, "right_delta_deg": delta}

    config = ExperimentConfig(output_root=tmp_path / "reports/factory_bimanual")
    manifest = FactoryBimanualExperiment(config, ik_executor=ik, mpc_executor=lambda job: {}).run()
    assert any(j["status"] == "failed" for j in manifest["jobs"])
    mpc = [j for j in manifest["jobs"] if j["stage"] == "mpc"]
    assert {j["mode"] for j in mpc if j["robot"] == "xarm6"} == {"canonical_a"}
    assert {j["mode"] for j in mpc if j["robot"] == "franka_panda"} == {"mpc_a", "mpc_easyik"}
    assert {j["mode"] for j in mpc if j["robot"] == "ur5"} == {"mpc_a"}


def test_resume_fingerprint_skips_completed_and_short_prefix_is_propagated(tmp_path: Path):
    calls = []
    root = tmp_path / "reports/factory_bimanual"
    config = ExperimentConfig(output_root=root, short_prefix_rows=12)
    executor = lambda job: calls.append(job) or {"initializer_valid": True, "left_delta_deg": 0, "right_delta_deg": 0}
    FactoryBimanualExperiment(config, ik_executor=executor, mpc_executor=lambda job: {}).run()
    first = len(calls)
    FactoryBimanualExperiment(config, ik_executor=executor, mpc_executor=lambda job: {}).run()
    assert len(calls) == first
    assert all(job.short_prefix_rows == 12 for job in calls)
    assert (root / "experiment_manifest.json").exists()


def test_output_root_and_prefix_validation(tmp_path: Path):
    with pytest.raises(ValueError):
        ExperimentConfig(output_root=tmp_path / "reports/single_arm")
    with pytest.raises(ValueError):
        ExperimentConfig(output_root=tmp_path / "reports/factory_bimanual", short_prefix_rows=0)


def test_completed_compute_jobs_schedule_resumable_render_post_jobs(tmp_path: Path):
    rendered = []
    root = tmp_path / "reports/factory_bimanual"
    config = ExperimentConfig(output_root=root, render_videos=True)
    ik = lambda job: {"initializer_valid": True, "left_delta_deg": 0, "right_delta_deg": 0,
                      "artifact_path": f"runs/{job.fingerprint}/arrays.npz"}
    mpc = lambda job: {"artifact_path": f"runs/{job.fingerprint}/arrays.npz"}
    video = lambda job: rendered.append(job) or {"video_path": f"videos/{job.fingerprint}.mp4",
                                                 "decode_verified": True}
    first = FactoryBimanualExperiment(config, ik_executor=ik, mpc_executor=mpc,
                                      video_executor=video).run()
    compute = [j for j in first["jobs"] if j["stage"] in {"ik", "mpc"}]
    render = [j for j in first["jobs"] if j["stage"] == "render"]
    assert len(render) == len(compute)
    assert all(j["status"] == "completed" and j["result"]["decode_verified"] for j in render)
    assert {j["source_fingerprint"] for j in render} == {j["fingerprint"] for j in compute}
    count = len(rendered)
    FactoryBimanualExperiment(config, ik_executor=ik, mpc_executor=mpc,
                              video_executor=video).run()
    assert len(rendered) == count


def test_failed_compute_job_does_not_schedule_render(tmp_path: Path):
    root = tmp_path / "reports/factory_bimanual"
    config = ExperimentConfig(output_root=root, render_videos=True)
    manifest = FactoryBimanualExperiment(
        config, ik_executor=lambda job: (_ for _ in ()).throw(RuntimeError("bad")),
        video_executor=lambda job: pytest.fail("render should not run"),
    ).run()
    assert not [j for j in manifest["jobs"] if j["stage"] == "render"]


def test_fingerprint_changes_with_same_named_robot_model_identity(
        tmp_path: Path, monkeypatch):
    config = ExperimentConfig(output_root=tmp_path / "reports/factory_bimanual")
    experiment = FactoryBimanualExperiment(config)
    original = experiment._fingerprint("ik", "piperx", "screw_cap", "strict_a")
    contracts = dict(ROBOT_CONTRACTS)
    contracts["piperx"] = replace(
        contracts["piperx"], tcp_offset_m=(0.0, 0.0, 0.13))
    monkeypatch.setattr(run_experiment_module, "ROBOT_CONTRACTS", contracts)

    changed = experiment._fingerprint("ik", "piperx", "screw_cap", "strict_a")
    assert changed != original
