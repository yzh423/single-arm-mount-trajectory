import numpy as np
import pytest
import csv
from types import SimpleNamespace
from pathlib import Path
import scripts.build_piperx_multitask_fixed_time_bundle as bundle

from factory_bimanual.multitask_fixed_time_study import TrajectorySpec
from factory_bimanual.task_family import TaskFamily

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


def test_recovery_step_requires_collision_and_mount_topology_safety():
    candidate = lambda value: SimpleNamespace(q=np.asarray([value]))

    class CollisionChecker:
        @staticmethod
        def state(left, right):
            return SimpleNamespace(valid=True)

        @staticmethod
        def transition(previous, current):
            return SimpleNamespace(valid=True)

    class TopologyChecker:
        @staticmethod
        def state(left, right):
            return SimpleNamespace(valid=left[0] <= .2 and right[0] <= .2)

        @staticmethod
        def transition(previous, current):
            return SimpleNamespace(valid=True)

    checker = bundle._ConjunctivePairChecker(
        CollisionChecker(), TopologyChecker())
    step = bundle._select_safe_recovery_step(
        {"left": [candidate(.8)], "right": [candidate(.8)]},
        previous=(np.asarray([0.0]), np.asarray([0.0])),
        checker=checker, maximum_step_rad=.30)

    assert step is None


def test_recovery_waits_until_preinitialized_follow_segment_begins():
    assert hasattr(bundle, "_recovery_allowed")

    assert not bundle._recovery_allowed(39, initialization_row=40)
    assert bundle._recovery_allowed(40, initialization_row=40)


def test_formal_shard_cache_requires_exact_input_fingerprint(tmp_path):
    summary = tmp_path / "trajectory.summary.json"
    summary.write_text(
        '{"schema":"piperx-multitask-fixed-time-summary-v1"}',
        encoding="utf-8")

    assert not bundle._formal_summary_reusable(summary, "expected")

    summary.write_text(
        '{"schema":"piperx-multitask-fixed-time-summary-v1",'
        '"solver_protocol":"piperx-fixed-time-paired-topology-safe-recovery-v3",'
        '"formal_fingerprint":"different"}',
        encoding="utf-8")
    assert not bundle._formal_summary_reusable(summary, "expected")

    summary.write_text(
        '{"schema":"piperx-multitask-fixed-time-summary-v1",'
        '"solver_protocol":"piperx-fixed-time-paired-topology-safe-recovery-v3",'
        '"formal_fingerprint":"expected"}',
        encoding="utf-8")
    assert bundle._formal_summary_reusable(summary, "expected")


def test_artifact_resolver_rejects_absolute_and_parent_escape_paths(tmp_path):
    with pytest.raises(ValueError, match="repository-relative"):
        bundle._resolve_artifact(tmp_path / "outside.npz")
    with pytest.raises(ValueError, match="outside repository"):
        bundle._resolve_artifact("../outside.npz")


def _summary_contract_fixture():
    spec = TrajectorySpec(
        family=TaskFamily("8-11", "Fold_Box"), take="161044",
        path=Path("source.csv"), source_sha256="a" * 64, row_count=3)
    shard = {
        "trajectory": spec.key, "family": spec.family.key,
        "take": spec.take, "mode": "upright_table",
    }
    summary = {
        "schema": "piperx-multitask-fixed-time-summary-v1",
        "solver_protocol": bundle.FORMAL_SOLVER_PROTOCOL,
        "formal_fingerprint": "fingerprint",
        "trajectory": spec.key, "family": spec.family.key,
        "take": spec.take, "mode": "upright_table",
        "timing_mode": "fixed_source_time", "retiming_applied": False,
        "source_sha256": spec.source_sha256,
    }
    return spec, shard, summary


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema", "old"), ("solver_protocol", "old"),
        ("formal_fingerprint", "stale"), ("trajectory", "wrong"),
        ("family", "wrong"), ("take", "000000"),
        ("mode", "inverted"), ("timing_mode", "retimed"),
        ("retiming_applied", True), ("source_sha256", "b" * 64),
    ],
)
def test_summary_contract_rejects_identity_or_provenance_drift(field, bad_value):
    spec, shard, summary = _summary_contract_fixture()
    summary[field] = bad_value

    with pytest.raises(ValueError, match="summary contract"):
        bundle._validate_summary_contract(
            summary, shard, spec, expected_fingerprint="fingerprint")


def test_aggregate_recomputes_accept_and_collision_counts():
    payload = {
        "left_accept": np.asarray([True, True, False, True]),
        "right_accept": np.asarray([True, False, True, True]),
        "both_accept": np.asarray([True, False, False, True]),
        "collision": np.asarray([False, True, False, False]),
        "edge_collision": np.asarray([False, False, True, False]),
        "topology_valid": np.asarray([True, True, False, True]),
        "left_source_valid": np.ones(4, dtype=bool),
        "right_source_valid": np.asarray([True, True, False, True]),
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
    assert row["invalid_source_frames"] == 1
    assert row["source_valid_pair_frames"] == 3
    assert row["both_accept_coverage_valid_source"] == pytest.approx(2 / 3)
    assert row["longest_hold_frames"] == 2
    assert row["longest_hold_ratio"] == pytest.approx(.5)
    assert row["maximum_position_error_mm"] == 4.0
    assert row["maximum_accepted_position_error_mm"] == 0.0


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
        topology_valid=np.ones(3, bool),
        left_source_valid=np.ones(3, bool),
        right_source_valid=np.ones(3, bool), velocity_rad_s=velocity,
        acceleration_rad_s2=acceleration)

    np.testing.assert_array_equal(payload["fixed_time_s"], source_time)
    np.testing.assert_array_equal(payload["qpos"], qpos)
    np.testing.assert_array_equal(payload["both_accept"], [True, True, False])
    assert payload["retiming_applied"] is False
    assert np.max(payload["velocity_rad_s"]) == 99.0


def test_aggregate_csv_must_exactly_match_recomputed_shard_rows(tmp_path):
    aggregate = tmp_path / "aggregate.csv"
    expected = [{
        "trajectory": "8-11/Fold_Box/161044",
        "mode": "upright_table",
        "source_frames": 3,
        "both_accept_coverage": 1.0 / 3.0,
        "limits_passed": False,
        "maximum_accepted_position_error_mm": None,
    }]
    with aggregate.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(expected[0]))
        writer.writeheader()
        writer.writerows(expected)

    bundle._validate_aggregate_csv(aggregate, expected)

    aggregate.write_text(
        aggregate.read_text(encoding="utf-8").replace(
            "0.3333333333333333", "0.5"),
        encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate CSV drift"):
        bundle._validate_aggregate_csv(aggregate, expected)


def test_dynamics_summary_is_recomputed_from_formal_arrays():
    payload = {
        "velocity_rad_s": np.asarray([[0.0, 1.1], [0.2, 0.3]]),
        "acceleration_rad_s2": np.asarray([[0.0, 3.0], [4.1, 0.3]]),
    }

    assert bundle._dynamics_summary(payload) == {
        "maximum_velocity_rad_s": 1.1,
        "maximum_acceleration_rad_s2": 4.1,
        "limits_passed": False,
    }


def _formal_evidence_fixture():
    count = 2
    actual = np.asarray([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])
    return {
        "left_actual_tcp": actual.copy(),
        "right_actual_tcp": actual.copy(),
        "left_target_position_m": actual[:, :3].copy(),
        "right_target_position_m": actual[:, :3].copy(),
        "left_target_quaternion_wxyz": actual[:, 3:].copy(),
        "right_target_quaternion_wxyz": actual[:, 3:].copy(),
        "position_error_m": np.zeros((count, 2)),
        "orientation_error_rad": np.zeros((count, 2)),
        "collision": np.zeros(count, dtype=bool),
        "edge_collision": np.zeros(count, dtype=bool),
        "left_discontinuity": np.zeros(count, dtype=bool),
        "right_discontinuity": np.zeros(count, dtype=bool),
        "state_collision_classes": np.asarray(["", ""]),
        "edge_collision_classes": np.asarray(["", ""]),
        "paired_failure_reason": np.asarray(["ok", "ok"]),
        "dls_solve_mode": np.asarray([["hold", "hold"], ["hold", "hold"]]),
    }


def test_formal_evidence_binds_pose_errors_to_target_and_actual_tcp():
    payload = _formal_evidence_fixture()
    bundle._validate_formal_evidence(payload, 2)

    payload["position_error_m"][0, 0] = 0.001
    with pytest.raises(ValueError, match="position error drift"):
        bundle._validate_formal_evidence(payload, 2)


def test_formal_evidence_binds_collision_flags_to_class_evidence():
    payload = _formal_evidence_fixture()
    payload["state_collision_classes"][1] = "cross_arm"

    with pytest.raises(ValueError, match="collision class drift"):
        bundle._validate_formal_evidence(payload, 2)
