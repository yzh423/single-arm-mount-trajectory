"""Build a content-addressed release manifest for the Fixed-time study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

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
_CANONICAL_TEXT_SUFFIXES = frozenset({
    ".csv", ".json", ".md", ".py", ".toml", ".xml", ".yaml", ".yml",
})


def _record_content(path, hash_mode=None):
    path = Path(path)
    expected_mode = (
        "canonical_utf8_lf"
        if path.suffix.lower() in _CANONICAL_TEXT_SUFFIXES else "raw")
    mode = expected_mode if hash_mode is None else hash_mode
    if mode not in ("canonical_utf8_lf", "raw"):
        raise ValueError(f"unsupported release hash mode: {mode!r}")
    if mode != expected_mode:
        raise ValueError(f"release hash mode drift: {path}")
    data = path.read_bytes()
    if mode == "canonical_utf8_lf":
        text = data.decode("utf-8-sig")
        data = text.replace("\r\n", "\n").replace(
            "\r", "\n").encode("utf-8")
    return mode, data


def _file_record(path):
    path = Path(path).resolve()
    hash_mode, data = _record_content(path)
    return {
        "path": bundle._repo_relative(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "hash_mode": hash_mode,
    }


def _validate_file_record(record):
    path = bundle._resolve_artifact(record["path"])
    _mode, data = _record_content(path, record.get("hash_mode"))
    if (len(data) != int(record["bytes"])
            or hashlib.sha256(data).hexdigest() != record["sha256"]):
        raise ValueError(f"release artifact drift: {record['path']}")
    return path


def _scene_mesh_dependencies(scene_path):
    """Resolve every mesh file actually loaded by a published scene."""
    scene_path = Path(scene_path).resolve()
    root = ET.parse(scene_path).getroot()
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", ".") if compiler is not None else ".")
    if meshdir.is_absolute():
        raise ValueError("published scene meshdir must be scene-relative")
    mesh_root = (scene_path.parent / meshdir).resolve()
    dependencies = set()
    for mesh in root.findall("./asset/mesh"):
        filename = Path(mesh.get("file", ""))
        if not filename.parts or filename.is_absolute():
            raise ValueError("published scene mesh file must be relative")
        dependency = (mesh_root / filename).resolve()
        try:
            dependency.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("published scene mesh resolves outside repository") from exc
        if not dependency.is_file():
            raise FileNotFoundError(dependency)
        dependencies.add(dependency)
    if not dependencies:
        raise ValueError("published scene must load real mesh dependencies")
    return tuple(sorted(dependencies, key=lambda path: path.as_posix()))


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
    scene_mesh_dependencies = set()
    for shard in manifest["shards"]:
        path = bundle._resolve_artifact(shard["summary_json"])
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(_file_record(path))
        source_hashes[summary["trajectory"]] = summary["source_sha256"]
        scene = bundle._resolve_artifact(
            summary["artifacts"]["scene_xml"]["path"])
        scene_mesh_dependencies.update(_scene_mesh_dependencies(scene))

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
        "scene_mesh_dependencies": [
            _file_record(path) for path in sorted(
                scene_mesh_dependencies, key=lambda item: item.as_posix())
        ],
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
    expected_scene_dependencies = set()
    for record in summary_records:
        summary_path = _validate_file_record(record)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_hashes[summary["trajectory"]] = summary["source_sha256"]
        expected_panel_evidence.setdefault(summary["trajectory"], {})[
            summary["mode"]] = renderer._panel_provenance(summary_path)
        scene_path = bundle._resolve_artifact(
            summary["artifacts"]["scene_xml"]["path"])
        expected_scene_dependencies.update(
            _scene_mesh_dependencies(scene_path))
    if dict(sorted(source_hashes.items())) != payload.get(
            "source_sha256_by_trajectory"):
        raise ValueError("release source hash index drift")
    dependency_records = payload.get("scene_mesh_dependencies", [])
    expected_dependency_paths = {
        bundle._repo_relative(path) for path in expected_scene_dependencies
    }
    if ({record.get("path") for record in dependency_records}
            != expected_dependency_paths):
        raise ValueError("release scene mesh dependency set drift")
    for record in dependency_records:
        _validate_file_record(record)
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
