import numpy as np
import pytest
from types import SimpleNamespace
import scripts.build_piperx_multitask_fixed_time_bundle as bundle

from scripts.build_piperx_multitask_fixed_time_bundle import (
    _piperx_collision_checker_kwargs,
    _select_collision_safe_pair,
    aggregate_shard,
    build_shard_arrays,
    validate_manifest,
)


def test_fixed_time_solver_uses_same_clearance_contract_as_final_audit():
    assert _piperx_collision_checker_kwargs() == {
        "transition_steps": 5,
        "clearance_margin_m": pytest.approx(0.015),
    }


def _manifest(count=108):
    return {
        "schema": "piperx-multitask-fixed-time-bundle-v1",
        "trajectory_count": 27,
        "family_count": 12,
        "mode_count": 4,
        "shards": [
            {"trajectory": f"trajectory-{index // 4}",
             "mode": ("baseline", "upright_table", "horizontal_wall", "inverted")[index % 4]}
            for index in range(count)
        ],
    }


def test_bundle_requires_all_108_shards():
    validate_manifest(_manifest())

    with pytest.raises(ValueError, match="108"):
        validate_manifest(_manifest(107))


def test_collision_safe_pair_rejects_state_and_swept_edge_collisions():
    candidate = lambda value, cost=0.0: SimpleNamespace(
        q=np.asarray([value]), pose_cost=cost,
        joint_limit_margin_rad=1.0, singularity_margin=1.0)

    class Checker:
        @staticmethod
        def state(left, right):
            return SimpleNamespace(valid=not (
                left[0] == 1.0 and right[0] == 1.0))

        @staticmethod
        def transition(previous, current):
            return SimpleNamespace(valid=not (
                current[0][0] == 2.0 and current[1][0] == 2.0))

    selected = _select_collision_safe_pair(
        {"left": [candidate(1.0), candidate(0.1)],
         "right": [candidate(1.0), candidate(0.1)]},
        previous=(np.asarray([0.0]), np.asarray([0.0])),
        checker=Checker(), branch_guard_rad=.30)

    assert selected is not None
    assert selected[0].q[0] == pytest.approx(.1)
    assert selected[1].q[0] == pytest.approx(.1)


def test_first_safe_pair_can_initialize_after_unreachable_opening_rows():
    assert hasattr(bundle, "_continuity_reference")
    previous = (np.asarray([0.0]), np.asarray([0.0]))
    far = SimpleNamespace(
        q=np.asarray([1.0]), pose_cost=0.0,
        joint_limit_margin_rad=1.0, singularity_margin=1.0)

    class Checker:
        @staticmethod
        def state(left, right):
            return SimpleNamespace(valid=True)

        @staticmethod
        def transition(previous_pair, current):
            return SimpleNamespace(valid=True)

    initial = _select_collision_safe_pair(
        {"left": [far], "right": [far]},
        previous=bundle._continuity_reference(False, previous),
        checker=Checker(), branch_guard_rad=.30)
    tracked = _select_collision_safe_pair(
        {"left": [far], "right": [far]},
        previous=bundle._continuity_reference(True, previous),
        checker=Checker(), branch_guard_rad=.30)

    assert initial is not None
    assert tracked is None


def test_initializer_probes_use_search_rows_known_to_have_safe_pairs():
    assert hasattr(bundle, "_initializer_probe_rows")
    selected_result = {
        "sampled_source_rows": [0, 20, 40, 60],
        "disconnected_rows": [0, 20],
    }

    assert bundle._initializer_probe_rows(
        selected_result, source_count=50) == [40]


def test_initializer_prioritizes_start_of_longest_safe_sample_run():
    selected_result = {
        "sampled_source_rows": [0, 10, 20, 30, 40, 50, 60],
        "disconnected_rows": [0, 20, 30],
    }

    rows = bundle._initializer_probe_rows(
        selected_result, source_count=61)

    assert rows[0] == 40


def test_missing_one_side_candidate_triggers_paired_rescue():
    assert hasattr(bundle, "_requires_pair_rescue")

    assert bundle._requires_pair_rescue(
        {"left": object(), "right": None},
        state_safe=True, edge_safe=True)
    assert not bundle._requires_pair_rescue(
        {"left": object(), "right": object()},
        state_safe=True, edge_safe=True)


def test_recovery_step_moves_toward_nearest_safe_pair_without_teleporting():
    assert hasattr(bundle, "_select_safe_recovery_step")
    candidate = lambda value: SimpleNamespace(q=np.asarray([value]))

    class Checker:
        @staticmethod
        def state(left, right):
            return SimpleNamespace(valid=not (
                left[0] < -.5 or right[0] > .5))

        @staticmethod
        def transition(previous, current):
            return SimpleNamespace(valid=True)

    step = bundle._select_safe_recovery_step(
        {"left": [candidate(.8), candidate(-.4)],
         "right": [candidate(.8), candidate(.4)]},
        previous=(np.asarray([0.0]), np.asarray([0.0])),
        checker=Checker(), maximum_step_rad=.30)

    assert step is not None
    np.testing.assert_allclose(step[0], [-.3])
    np.testing.assert_allclose(step[1], [.3])


def test_recovery_waits_until_preinitialized_follow_segment_begins():
    assert hasattr(bundle, "_recovery_allowed")

    assert not bundle._recovery_allowed(39, initialization_row=40)
    assert bundle._recovery_allowed(40, initialization_row=40)


def test_formal_shard_cache_requires_current_solver_protocol(tmp_path):
    summary = tmp_path / "trajectory.summary.json"
    summary.write_text(
        '{"schema":"piperx-multitask-fixed-time-summary-v1"}',
        encoding="utf-8")

    assert not bundle._formal_summary_reusable(summary)

    summary.write_text(
        '{"schema":"piperx-multitask-fixed-time-summary-v1",'
        '"solver_protocol":"piperx-fixed-time-paired-preinit-safe-recovery-v2"}',
        encoding="utf-8")
    assert bundle._formal_summary_reusable(summary)


def test_aggregate_recomputes_accept_and_collision_counts():
    payload = {
        "left_accept": np.asarray([True, True, False, True]),
        "right_accept": np.asarray([True, False, True, True]),
        "both_accept": np.asarray([True, False, False, True]),
        "collision": np.asarray([False, True, False, False]),
        "edge_collision": np.asarray([False, False, True, False]),
        "topology_valid": np.asarray([True, True, False, True]),
        "position_error_m": np.asarray([
            [0.0, 0.0], [0.001, 0.002], [0.003, 0.004], [0.0, 0.0]]),
        "orientation_error_rad": np.zeros((4, 2)),
    }

    row = aggregate_shard(payload)

    assert row["left_accept_frames"] == 3
    assert row["right_accept_frames"] == 3
    assert row["both_accept_frames"] == 2
    assert row["collision_frames"] == 1
    assert row["edge_collision_frames"] == 1
    assert row["topology_invalid_frames"] == 1
    assert row["longest_hold_frames"] == 2
    assert row["maximum_position_error_mm"] == 4.0


def test_shard_arrays_keep_source_time_and_separate_pose_from_dynamics():
    source_time = np.asarray([0.0, 0.1, 0.25])
    qpos = np.arange(54, dtype=float).reshape(3, 18)
    position = np.asarray([[0.0, 0.0], [0.0009, 0.0009], [0.0011, 0.0]])
    orientation = np.zeros((3, 2))
    velocity = np.full((3, 12), 99.0)
    acceleration = np.full((3, 12), 999.0)

    payload = build_shard_arrays(
        mode="upright_table", source_time_s=source_time, qpos=qpos,
        position_error_m=position, orientation_error_rad=orientation,
        collision=np.zeros(3, bool), edge_collision=np.zeros(3, bool),
        topology_valid=np.ones(3, bool), velocity_rad_s=velocity,
        acceleration_rad_s2=acceleration)

    np.testing.assert_array_equal(payload["fixed_time_s"], source_time)
    np.testing.assert_array_equal(payload["qpos"], qpos)
    np.testing.assert_array_equal(payload["both_accept"], [True, True, False])
    assert payload["retiming_applied"] is False
    assert np.max(payload["velocity_rad_s"]) == 99.0
