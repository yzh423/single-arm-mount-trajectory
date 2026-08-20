import json
from pathlib import Path

from scripts.build_ten_arm_three_episode_report import provenance_for_report


def test_report_discloses_unqualified_joint_limits(tmp_path: Path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "status": "pass_with_limit_warning",
        "unqualified_ranking_robots": ["arx_x5"],
        "tcp_policy": "official or fallback",
        "geometry_policy": "official native",
    }), encoding="utf-8")
    row = provenance_for_report(gate)
    assert row["unqualified_ranking_robots"] == ["arx_x5"]
    assert row["ranking_disclosure_required"] is True
