import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
BASE = ROOT / "data" / "processed" / "local_pose_benchmark"


def _eligible_ids(split):
    manifest = json.loads((BASE / "manifest.json").read_text())
    return {row["episode_id"] for row in manifest["episodes"]
            if row["split"] == split and row.get("trajectory_edge_trim_eligible", True)}


def test_validation_export_contains_every_eligible_episode_once():
    rows = json.loads((BASE / "pilot_samples.json").read_text())
    assert {row["episode_id"] for row in rows} == _eligible_ids("validation")
    assert len(rows) == len(_eligible_ids("validation"))


def test_test_export_contains_every_eligible_episode_once():
    rows = json.loads((BASE / "test_samples.json").read_text())
    assert {row["episode_id"] for row in rows} == _eligible_ids("test")
    assert len(rows) == len(_eligible_ids("test"))
