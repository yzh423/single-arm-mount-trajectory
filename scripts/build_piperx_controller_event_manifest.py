"""Build and validate a render manifest for controller-event v4 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from factory_bimanual.multitask_fixed_time_study import STUDY_MODES
from scripts.run_piperx_controller_event_v4 import (
    EVENT_SOLVER_PROTOCOL,
    ROOT,
    validate_acceptance_evidence,
    validate_event_shard,
    validate_zero_safety_evidence,
)
from factory_bimanual.multitask_fixed_time_study import (
    discover_dual_hand_trajectories,
)
from scripts.build_piperx_multitask_fixed_time_bundle import aggregate_shard


def _archive_scalar(payload, key):
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"event shard {key} must be scalar")
    return value.reshape(()).item()


def validate_archive_summary(value, payload, *, spec=None):
    """Recompute release gates from an NPZ instead of trusting its summary."""
    if str(_archive_scalar(payload, "solver_protocol")) != EVENT_SOLVER_PROTOCOL:
        raise ValueError("event trajectory solver protocol mismatch")
    if bool(_archive_scalar(payload, "retiming_applied")):
        raise ValueError("event trajectory cannot apply retiming")
    source_time = np.asarray(payload["source_time_s"], dtype=float)
    fixed_time = np.asarray(payload["fixed_time_s"], dtype=float)
    if not np.array_equal(source_time, fixed_time):
        raise ValueError("event trajectory fixed time differs from source time")
    validate_zero_safety_evidence(payload)
    validate_acceptance_evidence(payload)
    if spec is not None:
        source_prefix = int(value["controller_event_rows"])
        if source_prefix == int(value.get(
                "controller_event_rows_total", source_prefix)):
            source_prefix = None
        validate_event_shard(
            payload, spec, value["mode"], source_prefix=source_prefix)
    metrics = aggregate_shard(payload)
    for key in (
            "source_frames", "collision_frames", "edge_collision_frames",
            "topology_invalid_frames", "both_accept_frames"):
        if int(value[key]) != int(metrics[key]):
            raise ValueError(f"event summary {key} differs from trajectory")
    if not np.isclose(
            float(value["both_accept_coverage"]),
            float(metrics["both_accept_coverage"]), rtol=0.0, atol=1e-12):
        raise ValueError(
            "event summary coverage differs from trajectory")
    return source_time


def build_manifest(output):
    output = Path(output).resolve()
    summaries = sorted((output / "shards").rglob("*.summary.json"))
    if not summaries:
        raise ValueError("no controller-event summaries found")
    rows = []
    timelines = {}
    specs = {
        spec.key: spec for spec in discover_dual_hand_trajectories(
            ROOT / "data/factory")}
    for path in summaries:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("trajectory") not in specs:
            raise ValueError(f"unknown source trajectory: {path}")
        if value.get("solver_protocol") != EVENT_SOLVER_PROTOCOL:
            raise ValueError(f"unexpected solver protocol: {path}")
        if value.get("retiming_applied") is not False:
            raise ValueError(f"retiming is not allowed: {path}")
        if any(int(value.get(name, -1)) != 0 for name in (
                "collision_frames", "edge_collision_frames",
                "topology_invalid_frames")):
            raise ValueError(f"unsafe event shard: {path}")
        artifacts = value.get("artifacts") or {
            "scene_xml": {"path": value["scene"]},
            "trajectory_npz": {"path": value["trajectory_artifact"]},
        }
        value["artifacts"] = artifacts
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        trajectory_path = ROOT / artifacts["trajectory_npz"]["path"]
        with np.load(trajectory_path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        timeline = validate_archive_summary(
            value, payload, spec=specs[value["trajectory"]])
        key = value["trajectory"]
        if key in timelines and not np.array_equal(timelines[key], timeline):
            raise ValueError(f"mount timelines differ for {key}")
        timelines[key] = timeline
        row_key = (key, value["mode"])
        if any((row["trajectory"], row["mode"]) == row_key for row in rows):
            raise ValueError(f"duplicate event shard: {row_key}")
        rows.append({
            "trajectory": key,
            "mode": value["mode"],
            "summary_json": str(path.relative_to(ROOT)),
            "controller_event_rows": int(value["controller_event_rows"]),
            "both_accept_coverage": float(value["both_accept_coverage"]),
        })
    keys = {(row["trajectory"], row["mode"]) for row in rows}
    trajectories = sorted({row["trajectory"] for row in rows})
    expected = {(trajectory, mode) for trajectory in trajectories
                for mode in STUDY_MODES}
    if keys != expected:
        missing = sorted(expected - keys)
        raise ValueError(f"four event shards are required per trajectory: {missing}")
    payload = {
        "schema": "piperx-controller-event-render-manifest-v1",
        "solver_protocol": EVENT_SOLVER_PROTOCOL,
        "timing_mode": "fixed_controller_event_time",
        "retiming_applied": False,
        "trajectory_count": len(trajectories),
        "mode_count": len(STUDY_MODES),
        "shards": rows,
    }
    destination = output / "bundle_manifest.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="OUTPUT_DIR",
        help="Controller-event study directory containing shards/; writes bundle_manifest.json there.",
    )
    args = parser.parse_args(argv)
    print(build_manifest(args.output))


if __name__ == "__main__":
    main()
