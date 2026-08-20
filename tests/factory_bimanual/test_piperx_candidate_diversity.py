import numpy as np
import pytest

from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig,
    normalized_pose_residual,
    piperx_wrist_risk,
    stratified_joint_seeds,
)


def test_stratified_seed_bank_is_deterministic_bounded_and_branch_diverse():
    lower = np.array([-2.6, 0.0, -2.9, -1.55, -1.55, -2.1])
    upper = np.array([2.6, 3.1, 0.0, 1.55, 1.55, 2.1])

    first = np.asarray(stratified_joint_seeds(lower, upper))
    second = np.asarray(stratified_joint_seeds(lower, upper))

    assert np.array_equal(first, second)
    assert first.shape[1] == 6
    assert len(first) >= 16
    assert np.all(first >= lower)
    assert np.all(first <= upper)
    midpoint = (lower + upper) / 2.0
    for joint in (0, 2, 3, 4):
        assert np.any(first[:, joint] < midpoint[joint])
        assert np.any(first[:, joint] > midpoint[joint])


def test_pose_residual_is_normalized_by_declared_strict_tolerances():
    residual = normalized_pose_residual(
        np.array([.001, 0.0, 0.0]),
        np.array([np.deg2rad(1.5), 0.0, 0.0]),
    )

    np.testing.assert_allclose(residual, [1., 0., 0., 1., 0., 0.])


def test_piperx_wrist_risk_strongly_rejects_near_89_degree_candidates():
    neutral = piperx_wrist_risk(np.zeros(6))
    near_limit = piperx_wrist_risk(
        np.array([0., 0., 0., np.deg2rad(88.5), np.deg2rad(-88.5), 0.]))

    assert neutral == pytest.approx(0.0)
    assert near_limit > 100.0


def test_piperx_options_are_explicit_and_disabled_for_other_robots_by_default():
    default = CandidateGeneratorConfig()
    assert not default.stratified_seed_enabled
    assert not default.bounded_optimizer_enabled
    assert not default.wrist_risk_enabled

    piperx = CandidateGeneratorConfig(
        stratified_seed_enabled=True,
        bounded_optimizer_enabled=True,
        wrist_risk_enabled=True,
    )
    assert piperx.stratified_seed_enabled
    assert piperx.bounded_optimizer_enabled
    assert piperx.wrist_risk_enabled
