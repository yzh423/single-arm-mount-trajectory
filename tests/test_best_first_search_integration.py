import mujoco
import numpy as np
import scripts.search_strict_urdf_mount as strict_search

from scripts.search_strict_urdf_mount import (
    _best_first_mount_candidate,
    _evaluate_layered_model_path,
    _layered_record_metrics,
    _merge_evaluation_records,
    _should_use_incumbent_replay,
    build_parser,
    resolved_best_first_budget,
)


def test_layered_window_record_cannot_claim_episode_success() -> None:
    record = _layered_record_metrics(
        scope="window", candidate_id=73,
        pose_success=np.ones(4, dtype=bool),
        position_error_m=np.zeros(4), orientation_error_rad=np.zeros(4),
        state_collision=np.zeros(4, dtype=bool),
        edge_collision=np.zeros(4, dtype=bool),
    )

    assert record["candidate_id"] == 73
    assert record["episode_success"] is False
    assert record["frame_coverage"] == 1.0
    assert record["rank"][0] == 0.0


def test_layered_full_record_hard_rejects_swept_collision() -> None:
    record = _layered_record_metrics(
        scope="full_episode", candidate_id=9,
        pose_success=np.ones(3, dtype=bool),
        position_error_m=np.zeros(3), orientation_error_rad=np.zeros(3),
        state_collision=np.zeros(3, dtype=bool),
        edge_collision=np.asarray((False, True, False)),
    )

    assert record["feasible"] is False
    assert record["episode_success"] is False
    assert record["edge_collision_frames"] == 1


def test_layered_model_evaluator_preserves_candidate_id_and_window_scope() -> None:
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/><worldbody><body>
      <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="sphere" size=".01" mass=".1"/><site name="strict_tracking_tcp" pos=".2 0 0"/>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "strict_tracking_tcp")
    positions, quaternions = [], []
    for angle in (0.0, 0.05):
        data.qpos[0] = angle
        mujoco.mj_forward(model, data)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
        positions.append(data.site_xpos[site].copy())
        quaternions.append(quaternion)

    record = _evaluate_layered_model_path(
        model=model, joint_names=("j1",), candidate_id=44,
        targets=np.asarray(positions), target_quaternions=np.asarray(quaternions),
        time_s=np.asarray((0.0, 0.1)), scope="window",
        candidates_per_frame=3, global_seed_count=4, iterations=60,
        velocity_limit_rad_s=2.0, maximum_frame_jump_rad=.3,
    )

    assert record["candidate_id"] == 44
    assert record["episode_success"] is False
    assert record["frame_coverage"] == 1.0
    assert record["q"].shape == (2, 1)


def parse(*extra: str):
    return build_parser().parse_args([
        "--domain", "local", "--robot", "xarm6", "--task", "cap-left", *extra,
    ])


def test_best_first_cli_budget_is_explicit_and_has_64_local_candidates():
    args = parse("--search-policy", "best-first")
    budget = resolved_best_first_budget(args)
    assert args.search_policy == "best-first"
    assert budget.global_candidates == 128
    assert budget.retained_regions * budget.local_per_region == 64
    assert budget.post_success_expansions == 8


def test_legacy_remains_default_until_ablation_passes():
    assert parse().search_policy == "legacy"


def test_best_first_rejects_non_64_local_budget():
    args = parse(
        "--search-policy", "best-first", "--best-first-regions", "7",
        "--best-first-local-per-region", "8",
    )
    try:
        resolved_best_first_budget(args)
    except ValueError as exc:
        assert "64" in str(exc)
    else:
        raise AssertionError("best-first must require exactly 64 local candidates")


def test_legacy_compact_candidates_alias_still_parses():
    args = parse("--search-policy", "legacy", "--candidates", "2")
    assert args.candidates == 2


def test_top_records_hard_rejects_any_mount_with_collision_frames():
    colliding = {
        "feasible": True, "rank": (1.0,), "collision": np.asarray((False, True)),
        "candidate": np.zeros(6),
    }
    clean = {
        "feasible": True, "rank": (0.0,), "collision": np.asarray((False, False)),
        "candidate": np.ones(6),
    }

    selected = strict_search._top_records([colliding, clean], 2)

    assert selected == [clean]


def test_best_first_policy_evaluates_exactly_64_local_and_bounds_dense_work():
    args = parse(
        "--search-policy", "best-first",
        "--best-first-global-candidates", "8",
        "--best-first-regions", "2",
        "--best-first-local-per-region", "32",
        "--best-first-dense-before-success", "1",
        "--best-first-dense-fallback", "1",
        "--best-first-post-success-expansions", "1",
    )
    calls = []

    def fake_evaluate(**kwargs):
        stage = kwargs["label"]
        candidates = np.asarray(kwargs["candidates"])
        calls.append((stage, len(candidates)))
        records = []
        for item in candidates:
            value = float(item[0])
            records.append({
                "rank": (1.0 if stage == "dense" else 0.0, value),
                "candidate": item.copy(),
                "feasible": True,
                "episode_success": stage == "dense",
            })
        return records

    candidate, audit = _best_first_mount_candidate(
        args=args,
        targets=np.zeros((10, 3)),
        target_quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
        robot="xarm6",
        entry=object(),
        lower=np.array([-1.0] * 6),
        upper=np.array([1.0] * 6),
        warm_starts=np.zeros((1, 6)),
        evaluate_fn=fake_evaluate,
    )
    assert candidate.shape == (6,)
    assert ("local", 64) in calls
    assert audit["local_candidate_count"] == 64
    assert audit["dense_candidate_count"] <= 2
    assert audit["search_policy"] == "best-first"
    assert audit["_authoritative_dense_record"]["episode_success"]


def test_dense_promotion_always_evaluates_full_chronological_episode():
    assert hasattr(strict_search, "_dense_evaluation_indices")
    np.testing.assert_array_equal(
        strict_search._dense_evaluation_indices(
            10, previously_ranked=np.array([0, 3, 9])), np.arange(10))
    args = parse(
        "--search-policy", "best-first", "--best-first-global-candidates", "8",
        "--best-first-regions", "2", "--best-first-local-per-region", "32",
        "--best-first-dense-before-success", "1",
        "--best-first-post-success-expansions", "1",
    )
    dense_indices = []

    def fake_evaluate(**kwargs):
        stage = kwargs["label"]
        if stage == "dense":
            dense_indices.append(np.asarray(kwargs["indices"]).copy())
        return [{"rank": (float(stage == "dense"), float(item[0])),
                 "candidate": item.copy(), "feasible": True,
                 "episode_success": stage == "dense"}
                for item in np.asarray(kwargs["candidates"])]

    _best_first_mount_candidate(
        args=args, targets=np.zeros((10, 3)),
        target_quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
        robot="xarm6", entry=object(), lower=np.full(6, -1.0),
        upper=np.full(6, 1.0), warm_starts=np.zeros((1, 6)),
        evaluate_fn=fake_evaluate)
    assert dense_indices
    assert all(np.array_equal(indices, np.arange(10)) for indices in dense_indices)


def test_dense_promotion_uses_authoritative_solver_with_real_timeline():
    args = parse(
        "--search-policy", "best-first", "--best-first-global-candidates", "8",
        "--best-first-regions", "2", "--best-first-local-per-region", "32",
        "--best-first-dense-before-success", "1",
        "--best-first-post-success-expansions", "1")
    timeline = np.linspace(0.0, 0.09, 10)
    strict_calls = []

    def cheap_evaluate(**kwargs):
        return [{"rank": (0.0, float(item[0])), "candidate": item.copy(),
                 "feasible": True, "episode_success": False}
                for item in np.asarray(kwargs["candidates"])]

    def authoritative_evaluate(**kwargs):
        strict_calls.append(kwargs)
        item = np.asarray(kwargs["candidate"])
        return {"rank": (1.0, 1.0), "candidate": item.copy(), "feasible": True,
                "episode_success": True, "planner_type": "rolling_multibranch"}

    _, audit = _best_first_mount_candidate(
        args=args, targets=np.zeros((10, 3)),
        target_quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
        time_s=timeline, robot="xarm6", entry=object(),
        lower=np.full(6, -1.0), upper=np.full(6, 1.0),
        warm_starts=np.zeros((1, 6)), evaluate_fn=cheap_evaluate,
        authoritative_evaluate_fn=authoritative_evaluate)
    assert strict_calls
    assert all(np.array_equal(call["time_s"], timeline) for call in strict_calls)
    assert all(np.array_equal(call["indices"], np.arange(10)) for call in strict_calls)
    assert audit["_authoritative_dense_record"]["planner_type"] == "rolling_multibranch"


def test_rank_stage_passes_separate_contiguous_windows():
    args = parse(
        "--search-policy", "best-first", "--best-first-global-candidates", "8",
        "--best-first-regions", "2", "--best-first-local-per-region", "32",
        "--best-first-dense-before-success", "1",
        "--best-first-post-success-expansions", "1")
    observed = []

    def fake_evaluate(**kwargs):
        if kwargs["label"] in {"global-rank", "local"}:
            observed.append(kwargs.get("index_windows"))
        return [{"rank": (float(kwargs["label"] == "dense"), float(item[0])),
                 "candidate": item.copy(), "feasible": True,
                 "episode_success": kwargs["label"] == "dense"}
                for item in np.asarray(kwargs["candidates"])]

    _best_first_mount_candidate(
        args=args, targets=np.zeros((40, 3)),
        target_quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (40, 1)),
        robot="xarm6", entry=object(), lower=np.full(6, -1.0),
        upper=np.full(6, 1.0), warm_starts=np.zeros((1, 6)),
        evaluate_fn=fake_evaluate)
    assert observed and all(windows is not None for windows in observed)
    assert all(all(np.all(np.diff(window) == 1) for window in windows)
               for windows in observed)


def test_promotion_merge_reuses_old_frames_without_duplicates():
    prior = {
        "candidate": np.zeros(6), "feasible": True,
        "frame_indices": np.array([0, 3]),
        "q": np.array([[0.0], [3.0]]),
        "success": np.array([True, True]),
        "position_error_m": np.array([0.0, 0.0]),
        "orientation_error_rad": np.array([0.0, 0.0]),
        "collision": np.array([False, False]),
    }
    added = {
        "candidate": np.zeros(6), "feasible": True,
        "frame_indices": np.array([1, 2]),
        "q": np.array([[1.0], [2.0]]),
        "success": np.array([True, True]),
        "position_error_m": np.array([0.0, 0.0]),
        "orientation_error_rad": np.array([0.0, 0.0]),
        "collision": np.array([False, False]),
    }
    merged = _merge_evaluation_records(prior, added)
    assert merged["frame_indices"].tolist() == [0, 1, 2, 3]
    assert merged["q"].ravel().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert merged["new_frames"] == 2
    assert merged["reused_frames"] == 2


def test_promotion_merge_rejects_recomputed_frame():
    record = {
        "candidate": np.zeros(6), "feasible": True,
        "frame_indices": np.array([0]), "q": np.array([[0.0]]),
        "success": np.array([True]), "position_error_m": np.array([0.0]),
        "orientation_error_rad": np.array([0.0]), "collision": np.array([False]),
    }
    try:
        _merge_evaluation_records(record, record)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("promotion must not recompute an existing frame")


def test_post_success_expands_medium_frontier_but_only_two_dense_challengers():
    args = parse(
        "--search-policy", "best-first",
        "--best-first-global-candidates", "8",
        "--best-first-regions", "2",
        "--best-first-local-per-region", "32",
        "--best-first-dense-before-success", "1",
        "--best-first-dense-fallback", "1",
        "--best-first-post-success-expansions", "4",
        "--best-first-post-success-dense", "2",
    )

    def fake_evaluate(**kwargs):
        stage = kwargs["label"]
        return [{
            "rank": (1.0 if stage == "dense" else 0.0, float(item[0])),
            "candidate": item.copy(), "feasible": True,
            "episode_success": stage == "dense",
        } for item in np.asarray(kwargs["candidates"])]

    _, audit = _best_first_mount_candidate(
        args=args, targets=np.zeros((10, 3)),
        target_quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
        robot="xarm6", entry=object(), lower=np.full(6, -1.0), upper=np.full(6, 1.0),
        warm_starts=np.zeros((1, 6)), evaluate_fn=fake_evaluate,
    )
    assert audit["post_success_expansions"] == 4
    assert audit["post_success_dense_candidates"] == 2
    assert audit["dense_candidate_count"] == 3


def test_failed_new_finalist_falls_back_to_known_pass_incumbent():
    assert _should_use_incumbent_replay(
        primary_episode_success=False,
        primary_joint_limits=True,
        incumbent_known_pass=True,
    )
    assert not _should_use_incumbent_replay(
        primary_episode_success=True,
        primary_joint_limits=True,
        incumbent_known_pass=True,
    )
    assert not _should_use_incumbent_replay(
        primary_episode_success=False,
        primary_joint_limits=True,
        incumbent_known_pass=False,
    )
