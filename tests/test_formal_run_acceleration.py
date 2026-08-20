from pathlib import Path

from scripts.formal_run_acceleration import (
    artifact_is_current,
    atomic_write_json,
    input_fingerprint,
    json_fingerprint_matches,
    render_fingerprint,
    search_fingerprint,
    solve_fingerprint,
)
from scripts.run_twelve_arm_two_single_tasks import _task_fingerprint, build_parser


def test_fingerprint_changes_when_an_input_changes(tmp_path: Path):
    model = tmp_path / "robot.urdf"
    trajectory = tmp_path / "trajectory.json"
    model.write_text("model-v1", encoding="utf-8")
    trajectory.write_text("trajectory-v1", encoding="utf-8")

    before = input_fingerprint([model, trajectory], {"candidates": 128})
    trajectory.write_text("trajectory-v2", encoding="utf-8")
    after = input_fingerprint([model, trajectory], {"candidates": 128})

    assert before != after


def test_current_artifact_requires_matching_fingerprint_and_files(tmp_path: Path):
    cache = tmp_path / "task.npz"
    audit = tmp_path / "task.json"
    cache.write_bytes(b"cache")
    fingerprint = "expected"
    audit.write_text('{"input_fingerprint": "expected"}', encoding="utf-8")

    assert artifact_is_current(cache, audit, fingerprint)
    assert not artifact_is_current(cache, audit, "different")
    cache.unlink()
    assert not artifact_is_current(cache, audit, fingerprint)


def test_json_fingerprint_match_rejects_stale_or_broken_results(tmp_path: Path):
    result = tmp_path / "dense.json"
    result.write_text('{"input_fingerprint": "fresh"}', encoding="utf-8")
    assert json_fingerprint_matches(result, "fresh")
    assert not json_fingerprint_matches(result, "stale")
    result.write_text("broken", encoding="utf-8")
    assert not json_fingerprint_matches(result, "fresh")


def test_fingerprint_is_independent_of_input_order(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    assert input_fingerprint([first, second], {"mode": "formal"}) == input_fingerprint(
        [second, first], {"mode": "formal"}
    )


def test_formal_runner_spends_saved_time_on_more_gpu_screening():
    args = build_parser().parse_args([])

    assert args.gpu_candidates == 8192
    assert args.candidate_batch == 64
    assert args.workers >= 2
    assert args.resume is True
    assert args.search_policy == "legacy"


def test_fingerprint_namespaces_are_independent(tmp_path: Path):
    source = tmp_path / "source"
    source.write_text("same", encoding="utf-8")
    parameters = {"mode": "formal"}
    values = {
        search_fingerprint([source], parameters),
        solve_fingerprint([source], parameters),
        render_fingerprint([source], parameters),
    }
    assert len(values) == 3


def test_render_change_does_not_invalidate_search_or_solve_fingerprint(tmp_path: Path):
    model = tmp_path / "robot.urdf"
    trajectory = tmp_path / "trajectory.npz"
    renderer = tmp_path / "renderer.py"
    model.write_bytes(b"model")
    trajectory.write_bytes(b"trajectory")
    renderer.write_bytes(b"renderer-v1")
    search_before = search_fingerprint([model, trajectory], {"policy": "best-first"})
    solve_before = solve_fingerprint([model, trajectory], {"mount": [0, 0, 0, 0, 0, 0]})
    render_before = render_fingerprint([renderer], {"width": 960})
    renderer.write_bytes(b"renderer-v2")
    assert search_fingerprint([model, trajectory], {"policy": "best-first"}) == search_before
    assert solve_fingerprint([model, trajectory], {"mount": [0, 0, 0, 0, 0, 0]}) == solve_before
    assert render_fingerprint([renderer], {"width": 960}) != render_before


def test_atomic_write_json_leaves_valid_main_file(tmp_path: Path):
    destination = tmp_path / "state.json"
    atomic_write_json(destination, {"frontier": [3, 1, 2]})
    assert destination.read_text(encoding="utf-8").startswith("{")
    assert json_fingerprint_matches(destination, "missing") is False
    assert not destination.with_suffix(".json.tmp").exists()


def test_gpu_coarse_budget_does_not_invalidate_strict_solution_fingerprint():
    small = build_parser().parse_args(["--gpu-candidates", "4096"])
    large = build_parser().parse_args(["--gpu-candidates", "8192"])
    assert _task_fingerprint("xarm6", "cap-left", small) == _task_fingerprint(
        "xarm6", "cap-left", large
    )
