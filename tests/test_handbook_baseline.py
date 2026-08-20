from pathlib import Path


BASELINE = Path(__file__).parents[1] / "docs" / "HANDBOOK_BASELINE.md"


def test_handbook_baseline_records_non_negotiable_project_rules():
    text = BASELINE.read_text(encoding="utf-8")
    required = (
        "selection metric == reporting metric",
        "[x, y, z, yaw]",
        "worst episode",
        "mean + threshold count + p10",
        "validation timeline",
        "showcase timeline",
    )
    assert all(clause in text for clause in required)
