import json
from pathlib import Path

from factory_bimanual.defaults import SELECTED_SPACING_M


ROOT = Path(__file__).resolve().parents[2]


def test_invalid_all_zero_scan_does_not_finalize_spacings():
    assert SELECTED_SPACING_M == {}
    evidence = ROOT / "reports/factory_bimanual/spacing/scan_invalid_all_zero.json"
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["status"] == "invalid_for_selection"
    assert set(payload["robots"]) == {"xarm6", "franka_panda", "i2rt_yam", "ur5"}
    assert all(row["all_zero_coverage"] for row in payload["robots"].values())
