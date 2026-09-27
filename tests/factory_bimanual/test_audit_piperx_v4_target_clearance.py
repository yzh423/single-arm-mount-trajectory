import json
from pathlib import Path

import numpy as np

from scripts import audit_piperx_v4_target_clearance as audit


ROOT = Path(__file__).resolve().parents[2]
SHARD = (ROOT / "reports/piperx_controller_event_v4/shards"
         / "8-11/Fold_Box/161044/baseline")
STEM = "8-11_Fold_Box_161044_baseline_event_v4"


def test_point_displacement_bound_includes_both_tcp_translation_errors():
    assert audit.paired_clearance_gain_bound_m(
        .1, .2, position_tolerance_m=.001,
        orientation_tolerance_rad=0.0) == .002


def test_fold_box_raw_target_has_certified_infeasible_clearance_events():
    result = audit.audit_target_clearance(
        SHARD / f"{STEM}.scene.xml",
        SHARD / f"{STEM}.trajectory.npz")

    assert result["event_count"] == 755
    assert result["certified_infeasible_count"] == 9
    assert result["certified_infeasible_indices"] == list(range(99, 108))
    assert 0.002 < result["minimum_exact_target_gap_m"] < 0.003
    assert result["acceptance_upper_bound"] == 746 / 755
    assert np.all(np.asarray(result["exact_target_gap_m"])[99:108] < .015)


def test_certificate_cli_writes_reproducible_json(tmp_path):
    output = tmp_path / "clearance.json"

    audit.main([
        "--scene", str(SHARD / f"{STEM}.scene.xml"),
        "--trajectory", str(SHARD / f"{STEM}.trajectory.npz"),
        "--output", str(output),
    ])

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["certified_infeasible_indices"] == list(range(99, 108))
    assert result["accepted_infeasible_overlap_count"] == 0
