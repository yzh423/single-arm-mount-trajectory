from pathlib import Path

from scripts.strict_urdf_model_audit import MODELS
from scripts.official_model_manifest import OFFICIAL_MODELS
from scripts.build_official_model_provenance_gate import build_gate
from scripts.run_ten_arm_three_pick_episodes import (
    OUT, ROBOTS, TASKS, coarse_fingerprint, coarse_result_complete, job_fingerprint,
    selected_episodes,
)
from scripts.run_thirteen_arm_dense_search import load_official_search_templates
from scripts.derive_openarm_single_arm import derive as derive_openarm_single


ROOT = Path(__file__).resolve().parents[1]
TEN_ARM_SET = {
    "doosan",
    "xarm6",
    "ur5",
    "kinova_gen3_lite",
    "arx_x5",
    "franka_panda",
    "franka_panda_locked_j3",
    "i2rt_yam",
    "openarm",
    "piperx",
}


def test_ten_arm_experiment_uses_only_pinned_official_models():
    official_root = (ROOT / "third_party" / "official_robot_models").resolve()
    for name in TEN_ARM_SET:
        entry = MODELS[name]
        assert entry.path.is_file(), (name, entry.path)
        assert entry.path.resolve().is_relative_to(official_root), (name, entry.path)
        assert "canonical" not in entry.path.parts


def test_tcp_policy_is_explicit_and_uses_130mm_only_as_fallback():
    official_tcp = {
        "xarm6", "ur5", "kinova_gen3_lite",
        "franka_panda", "franka_panda_locked_j3",
    }
    fallback_tcp = TEN_ARM_SET - official_tcp - {"doosan", "openarm", "piperx"}
    for name in official_tcp:
        assert MODELS[name].tcp_authority == "official_frame"
        assert MODELS[name].tool_offset_m == 0.0
    for name in fallback_tcp:
        assert MODELS[name].tcp_authority == "fallback_130mm"
        assert MODELS[name].tool_offset_m == 0.130
    assert MODELS["doosan"].tcp_authority == "specified_139mm"
    assert MODELS["doosan"].tool_offset_m == 0.139
    assert MODELS["openarm"].tcp_authority == "official_xacro_default_83p5mm"
    assert MODELS["openarm"].tool_offset_m == 0.0
    assert MODELS["piperx"].tcp_authority == "model_ee_frame_115mm"
    assert MODELS["piperx"].tool_offset_m == 0.0


def test_openarm_is_a_single_arm_official_derivative():
    entry = MODELS["openarm"]
    assert entry.path.resolve().is_relative_to(
        (ROOT / "third_party/official_robot_models/official_derived").resolve())
    assert entry.root_body == "openarm_left_link0"
    text = entry.path.read_text(encoding="utf-8")
    assert "openarm_left_hand_tcp" in text
    assert "openarm_right_joint1" not in text
    assert "openarm_body_link0" not in text
    assert 'xyz="0 0 0.0835"' in text
    assert entry.tcp_authority == "official_xacro_default_83p5mm"
    assert entry.tool_offset_m == 0.0


def test_openarm_single_arm_regeneration_preserves_official_tcp_default(tmp_path):
    output = derive_openarm_single(output=tmp_path / "openarm.urdf")
    text = output.read_text(encoding="utf-8")

    assert 'name="openarm_left_hand_tcp_joint"' in text
    assert 'xyz="0 0 0.0835"' in text


def test_gpu_coarse_templates_are_built_from_the_same_official_entries():
    templates = load_official_search_templates(device="cpu")
    assert set(templates) == set(MODELS)
    for name, template in templates.items():
        assert template.source_model.resolve() == MODELS[name].path.resolve()
        assert template.tcp_authority == MODELS[name].tcp_authority


def test_every_official_model_has_a_full_pinned_revision():
    assert set(OFFICIAL_MODELS) == TEN_ARM_SET
    for name, row in OFFICIAL_MODELS.items():
        if name == "piperx":
            assert len(row["source_artifact_sha256"]) == 64
            int(row["source_artifact_sha256"], 16)
            assert row["revision"] == "f6642ce0d7872c686f29c99e9e10cd23d1d49313"
            continue
        assert row["repository"].startswith("https://github.com/")
        assert len(row["revision"]) == 40
        int(row["revision"], 16)


def test_provenance_gate_separates_execution_from_scientific_ranking():
    gate = build_gate()
    assert gate["status"] == "pass_with_limit_warning"
    assert gate["experiment_execution_allowed"] is True
    assert gate["unqualified_ranking_robots"] == ["arx_x5"]
    arx = next(row for row in gate["robots"] if row["robot"] == "arx_x5")
    assert arx["joint_limit_authority"] == "official_urdf_unverified_placeholder"
    assert arx["ranking_eligible"] is False


def test_locked_panda_reuses_the_exact_official_panda_geometry():
    normal = MODELS["franka_panda"]
    locked = MODELS["franka_panda_locked_j3"]
    assert locked.path == normal.path
    assert locked.mesh_dir == normal.mesh_dir
    assert locked.locked_joint_ranges == {"panda_joint3": (-0.0001, 0.0001)}


def test_three_episode_rerun_is_isolated_and_fingerprints_official_geometry():
    assert OUT.name == "ten_arm_three_tasks_official_models_4096"
    episode = {"episode_id": "episode-a", "content_sha256": "trajectory-sha"}
    before = job_fingerprint("xarm6", episode)
    assert MODELS["xarm6"].path.read_bytes()
    assert before == job_fingerprint("xarm6", episode)


def test_three_different_validation_tasks_each_select_one_episode():
    episodes = selected_episodes()
    assert tuple(row["task"] for row in episodes) == TASKS
    assert len({row["episode_id"] for row in episodes}) == 3
    assert all(row["split"] == "validation" for row in episodes)


def test_strict_fingerprint_changes_with_tcp_or_limit_contract(monkeypatch):
    import scripts.run_ten_arm_three_pick_episodes as runner
    import scripts.strict_urdf_model_audit as audit
    from dataclasses import replace

    episode = {"episode_id": "episode-a", "content_sha256": "trajectory-sha"}
    before = runner.job_fingerprint("i2rt_yam", episode)
    monkeypatch.setitem(audit.MODELS, "i2rt_yam", replace(
        audit.MODELS["i2rt_yam"], tool_offset_m=0.129))
    assert runner.job_fingerprint("i2rt_yam", episode) != before


def test_coarse_resume_only_accepts_all_ten_robot_rows(tmp_path):
    result = tmp_path / "coarse.json"
    result.write_text('{"robots": {}}', encoding="utf-8")
    expected = coarse_fingerprint({"episode_id": "episode-a", "content_sha256": "sha"})
    assert not coarse_result_complete(result, expected)
    result.write_text(__import__("json").dumps({
        "input_fingerprint": "stale",
        "robots": {name: {} for name in ROBOTS},
    }), encoding="utf-8")
    assert not coarse_result_complete(result, expected)
    result.write_text(__import__("json").dumps({
        "input_fingerprint": expected,
        "robots": {name: {} for name in ROBOTS},
    }), encoding="utf-8")
    assert coarse_result_complete(result, expected)


def test_coarse_fingerprint_covers_pipeline_and_collision_proxy(monkeypatch, tmp_path):
    import scripts.run_ten_arm_three_pick_episodes as runner
    source_root = runner.ROOT
    sandbox = tmp_path / "project"
    (sandbox / "scripts").mkdir(parents=True)
    (sandbox / "reports/single_arm").mkdir(parents=True)
    (sandbox / "scripts/run_thirteen_arm_dense_search.py").write_text("pipeline-v1")
    collision = sandbox / "reports/single_arm/collision_proxy_profiles.json"
    collision.write_text("collision-v1")
    monkeypatch.setattr(runner, "ROOT", sandbox)
    episode = {"episode_id": "episode-a", "content_sha256": "sha"}
    before = runner.coarse_fingerprint(episode)
    collision.write_text("collision-v2")
    after = runner.coarse_fingerprint(episode)
    monkeypatch.setattr(runner, "ROOT", source_root)
    assert before != after
