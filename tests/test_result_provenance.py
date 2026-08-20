import pytest

from scripts.result_provenance import aggregate_scores, validate_provenance


def test_unverified_dependency_requires_explicit_override():
    with pytest.raises(ValueError, match="UNVERIFIED"):
        validate_provenance({"arm_model": "VERIFIED", "approach_axis": "UNVERIFIED"})


def test_aggregate_scores_reports_mean_threshold_count_and_p10():
    summary = aggregate_scores([0.5, 0.8, 1.0], threshold=0.9)
    assert summary["mean"] == pytest.approx(0.7666666667)
    assert summary["threshold_count"] == 1
    assert summary["threshold_total"] == 3
    assert summary["p10"] == pytest.approx(0.56)
