"""Build a content-addressed release manifest for the Fixed-time study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from factory_bimanual.video import decode_check_mp4
from scripts import build_piperx_multitask_fixed_time_bundle as bundle
from scripts import render_piperx_multitask_mount_comparisons as renderer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/piperx_multitask_fixed_time_mount_study"
IMPLEMENTATION_FILES = (
    ROOT / "configs/piperx_recommended_v31.json",
    ROOT / "factory_bimanual/multitask_fixed_time_study.py",
    ROOT / "factory_bimanual/robot_contracts.py",
    ROOT / "factory_bimanual/scene_builder.py",
    ROOT / "scripts/strict_urdf_model_audit.py",
    ROOT / "scripts/run_piperx_multitask_fixed_time_mount_study.py",
    ROOT / "scripts/build_piperx_multitask_fixed_time_bundle.py",
    ROOT / "scripts/render_piperx_multitask_mount_comparisons.py",
    ROOT / "scripts/plot_piperx_multitask_mount_results.py",
    ROOT / "scripts/build_piperx_multitask_mount_report.py",
)


def _file_record(path):
    path = Path(path).resolve()
    return {
        "path": bundle._repo_relative(path),
        "bytes": path.stat().st_size,
        "sha256": bundle._sha256(path),
    }


def _validate_file_record(record):
    path = bundle._resolve_artifact(record["path"])
    if (path.stat().st_size != int(record["bytes"])
            or bundle._sha256(path) != record["sha256"]):
        raise ValueError(f"release artifact drift: {record['path']}")
    return path


def _validate_video_provenance(provenance, record, check):
    decoded = {
        "frame_count": check.frame_count,
        "fps": check.fps,
        "duration_s": check.duration_s,
        "width": check.width,
        "height": check.height,
    }
    expected_provenance = {
        **decoded,
        "duration_s": float(check.frame_count / check.fps),
    }
    valid = (
        provenance.get("schema") == "piperx-four-mount-composite-v1"
        and provenance.get("output_sha256") == record["sha256"]
        and isinstance(provenance.get("trajectory"), str)
        and set(provenance.get("panels", {})) == set(bundle.STUDY_MODES)
    )
    for metadata, expected in ((record, decoded),
                               (provenance, expected_provenance)):
        for name, value in expected.items():
            recorded = metadata.get(name)
            try:
                matches = (
                    bool(np.isclose(recorded, value, rtol=0.0, atol=1e-9))
                    if isinstance(value, float) else recorded == value)
            except TypeError:
                matches = False
            valid = valid and matches
    for panel in provenance.get("panels", {}).values():
        evidence = panel.get("evidence")
        if not isinstance(evidence, dict):
            valid = False
            continue
        for name in ("sha256", "provenance_sha256"):
            value = panel.get(name, "")
            try:
                valid = valid and len(value) == 64 and int(value, 16) >= 0
            except (TypeError, ValueError):
                valid = False
    if not valid:
        raise ValueError(f"release video provenance drift: {record['path']}")
    return provenance["trajectory"]


def _validate_published_panel_evidence(provenance, expected_by_mode):
    panels = provenance.get("panels", {})
    if set(panels) != set(expected_by_mode):
        raise ValueError("release panel evidence drift: mount set mismatch")
    for mode, expected in expected_by_mode.items():
        if panels[mode].get("evidence") != expected:
            raise ValueError(
                f"release panel evidence drift: {mode} does not match shard")


def build_release_manifest(output_root=DEFAULT_OUTPUT):
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle.validate_bundle_artifacts(manifest_path)

    summaries = []
    source_hashes = {}
    for shard in manifest["shards"]:
        path = bundle._resolve_artifact(shard["summary_json"])
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(_file_record(path))
        source_hashes[summary["trajectory"]] = summary["source_sha256"]

    figures = [
        _file_record(path)
        for path in sorted((output_root / "figures").glob("*.png"))
    ]
    videos = []
    for path in sorted((output_root / "videos" / "comparisons").glob("*.mp4")):
        check = decode_check_mp4(path, expected_resolution=(1280, 720))
        sidecar = path.with_suffix(".provenance.json")
        record = _file_record(path)
        record.update({
            "frame_count": check.frame_count,
            "fps": check.fps,
            "duration_s": check.duration_s,
            "width": check.width,
            "height": check.height,
            "provenance": _file_record(sidecar),
        })
        videos.append(record)

    report = output_root / "PiperX多任务Fixed-Time四构型对比报告.pdf"
    payload = {
        "schema": "piperx-multitask-fixed-time-release-v1",
        "protocol": {
            "timing": "fixed_source_time",
            "retiming_applied": False,
            "position_tolerance_m": 0.001,
            "orientation_tolerance_deg": 0.5,
            "required_collision_frames": 0,
            "required_edge_collision_frames": 0,
            "required_topology_invalid_frames": 0,
        },
        "bundle_manifest": _file_record(manifest_path),
        "aggregate_csv": _file_record(output_root / "aggregate.csv"),
        "report_pdf": _file_record(report),
        "implementation_files": [
            _file_record(path) for path in IMPLEMENTATION_FILES
        ],
        "source_sha256_by_trajectory": dict(sorted(source_hashes.items())),
        "summary_files": summaries,
        "figures": figures,
        "videos": videos,
    }
    output = output_root / "release_manifest.json"
    bundle._atomic_json(output, payload)
    return output


def validate_release_manifest(path=DEFAULT_OUTPUT / "release_manifest.json"):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "piperx-multitask-fixed-time-release-v1":
        raise ValueError("unexpected Fixed-time release manifest schema")
    expected_protocol = {
        "timing": "fixed_source_time",
        "retiming_applied": False,
        "position_tolerance_m": 0.001,
        "orientation_tolerance_deg": 0.5,
        "required_collision_frames": 0,
        "required_edge_collision_frames": 0,
        "required_topology_invalid_frames": 0,
    }
    if payload.get("protocol") != expected_protocol:
        raise ValueError("Fixed-time release protocol drift")

    manifest_path = _validate_file_record(payload["bundle_manifest"])
    bundle.validate_bundle_artifacts(manifest_path)
    _validate_file_record(payload["aggregate_csv"])
    _validate_file_record(payload["report_pdf"])

    implementation_paths = {
        bundle._repo_relative(path) for path in IMPLEMENTATION_FILES
    }
    implementation_records = payload.get("implementation_files", [])
    if ({record.get("path") for record in implementation_records}
            != implementation_paths):
        raise ValueError("release implementation file set drift")
    for record in implementation_records:
        _validate_file_record(record)

    summary_records = payload.get("summary_files", [])
    figure_records = payload.get("figures", [])
    video_records = payload.get("videos", [])
    if len(summary_records) != 108 or len(figure_records) != 9:
        raise ValueError("release must contain 108 summaries and nine figures")
    if len(video_records) != 27:
        raise ValueError("release must contain 27 comparison videos")

    source_hashes = {}
    expected_panel_evidence = {}
    for record in summary_records:
        summary_path = _validate_file_record(record)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_hashes[summary["trajectory"]] = summary["source_sha256"]
        expected_panel_evidence.setdefault(summary["trajectory"], {})[
            summary["mode"]] = renderer._panel_provenance(summary_path)
    if dict(sorted(source_hashes.items())) != payload.get(
            "source_sha256_by_trajectory"):
        raise ValueError("release source hash index drift")
    for record in figure_records:
        _validate_file_record(record)
    video_trajectories = set()
    for record in video_records:
        video_path = _validate_file_record(record)
        provenance_path = _validate_file_record(record["provenance"])
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        check = decode_check_mp4(video_path, expected_resolution=(1280, 720))
        trajectory = _validate_video_provenance(provenance, record, check)
        if trajectory not in expected_panel_evidence:
            raise ValueError("release video trajectory is not in the bundle")
        _validate_published_panel_evidence(
            provenance, expected_panel_evidence[trajectory])
        if trajectory in video_trajectories:
            raise ValueError("release contains duplicate comparison videos")
        video_trajectories.add(trajectory)
    if video_trajectories != set(source_hashes):
        raise ValueError("release comparison video trajectory set drift")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.validate_only:
        manifest = output / "release_manifest.json"
        validate_release_manifest(manifest)
        print(manifest)
    else:
        print(build_release_manifest(output))


if __name__ == "__main__":
    main()
