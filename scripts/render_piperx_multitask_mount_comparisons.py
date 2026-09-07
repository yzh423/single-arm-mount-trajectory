"""Render synchronized four-panel videos for validated mount-study shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from factory_bimanual.artifacts import FrameDiagnostics
from factory_bimanual.mount_comparison_visuals import (
    MOUNT_AXIS_LABELS,
    MOUNT_COLORS,
    MOUNT_LABELS,
    STUDY_PANEL_ORDER,
    hex_to_bgr,
)
from factory_bimanual.video import (
    VideoRenderConfig,
    decode_check_mp4,
    render_mujoco_mp4,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/piperx_multitask_fixed_time_mount_study"
PANEL_RENDER_PROTOCOL = "piperx-four-mount-panel-v4-per-mode-camera"
COMPOSITE_RENDER_PROTOCOL = "piperx-four-mount-composite-v3"


def _camera_for_mode(mode):
    return {
        "baseline": (225.0, -22.0, 1.48),
        "upright_table": (225.0, -22.0, 1.48),
        "horizontal_wall": (305.0, -8.0, 1.35),
        "inverted": (45.0, 0.0, 1.35),
    }[mode]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(path):
    path = Path(path)
    if path.is_absolute():
        raise ValueError("artifact paths must be repository-relative")
    root = ROOT.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path resolves outside repository") from exc
    return resolved


def _panel_provenance(summary_path):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trajectory = _artifact_path(summary["artifacts"]["trajectory_npz"]["path"])
    scene = _artifact_path(summary["artifacts"]["scene_xml"]["path"])
    return {
        "schema": PANEL_RENDER_PROTOCOL,
        "summary_sha256": _sha256(summary_path),
        "trajectory_sha256": _sha256(trajectory),
        "scene_sha256": _sha256(scene),
    }


def _provenance_path(panel_path):
    return Path(panel_path).with_suffix(".provenance.json")


def _write_panel_provenance(panel_path, summary_path):
    sidecar = _provenance_path(panel_path)
    sidecar.write_text(
        json.dumps(_panel_provenance(summary_path), indent=2) + "\n",
        encoding="utf-8",
    )
    return sidecar


def _composite_panel_record(panel_path):
    panel_path = Path(panel_path)
    provenance_path = _provenance_path(panel_path)
    evidence = json.loads(provenance_path.read_text(encoding="utf-8"))
    return {
        "sha256": _sha256(panel_path),
        "provenance_sha256": _sha256(provenance_path),
        "evidence": evidence,
    }


def _write_composite_provenance(
        output_path, panel_paths, trajectory, check):
    output_path = Path(output_path)
    sidecar = output_path.with_suffix(".provenance.json")
    payload = {
        "schema": COMPOSITE_RENDER_PROTOCOL,
        "trajectory": trajectory,
        "output_sha256": _sha256(output_path),
        "frame_count": int(check.frame_count),
        "fps": float(check.fps),
        "duration_s": float(check.frame_count / check.fps),
        "width": int(check.width),
        "height": int(check.height),
        "panels": {
            mode: _composite_panel_record(panel_paths[mode])
            for mode in STUDY_PANEL_ORDER
        },
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return sidecar


def _cached_panel_is_valid(path, summary_path):
    """Reuse only decodable panels bound to the current shard and scene."""
    try:
        check = decode_check_mp4(Path(path), expected_resolution=(640, 360))
        recorded = json.loads(
            _provenance_path(path).read_text(encoding="utf-8"))
        expected = _panel_provenance(summary_path)
    except (RuntimeError, OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return abs(check.fps - 30.0) <= 1e-6 and recorded == expected


def _load_shard(summary_path):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trajectory = _artifact_path(
        summary["artifacts"]["trajectory_npz"]["path"])
    scene = _artifact_path(summary["artifacts"]["scene_xml"]["path"])
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return summary, arrays, scene


def render_panel(summary_path, output_path):
    summary, arrays, scene = _load_shard(summary_path)
    mode = summary["mode"]
    accept = np.asarray(arrays["both_accept"], dtype=bool)
    collision = (np.asarray(arrays["collision"], dtype=bool)
                 | np.asarray(arrays["edge_collision"], dtype=bool))
    reasons = np.asarray(arrays["paired_failure_reason"], dtype=str)
    source_rows = np.asarray(
        arrays.get("source_poll_row_index", np.arange(len(accept))),
        dtype=int)
    diagnostics = []
    for index, timestamp in enumerate(arrays["source_time_s"]):
        reason = "ok" if accept[index] else str(reasons[index])
        diagnostics.append(FrameDiagnostics(
            source_row_index=int(source_rows[index]),
            source_time_s=float(timestamp),
            left_failure_reason=reason,
            right_failure_reason=reason,
            collision=bool(collision[index]),
            rollback=False,
            source_path=str(summary["trajectory"]),
            execution_index=index,
            execution_time_s=float(timestamp),
            execution_state=(
                "FOLLOW_FIXED_TIME" if accept[index] else "HOLD_FIXED_TIME"),
            state_collision=bool(arrays["collision"][index]),
            incoming_transition_collision=bool(
                arrays["edge_collision"][index]),
        ))
    camera_azimuth, camera_elevation, camera_scale = _camera_for_mode(mode)
    return render_mujoco_mp4(
        scene, output_path, arrays["source_time_s"],
        qpos=arrays["qpos"], diagnostics=diagnostics,
        left_targets=arrays["left_target_position_m"],
        right_targets=arrays["right_target_position_m"],
        follow_success=accept, failure_reasons=reasons,
        timeline_domain="fixed_source_time",
        config=VideoRenderConfig(
            width=640, height=360, fps=30,
            interpolate_states=True,
            camera_azimuth_deg=camera_azimuth,
            camera_elevation_deg=camera_elevation,
            camera_distance_scale=camera_scale,
            title=f"{MOUNT_LABELS[mode]} | fixed source time",
        ))


def compose_four_panel(panel_paths, output_path, *, trajectory_label):
    if tuple(panel_paths) != STUDY_PANEL_ORDER:
        raise ValueError("panel paths must use the published mount order")
    captures = {mode: cv2.VideoCapture(str(panel_paths[mode]))
                for mode in STUDY_PANEL_ORDER}
    try:
        counts = {mode: int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                  for mode, cap in captures.items()}
        fps_values = {mode: float(cap.get(cv2.CAP_PROP_FPS))
                      for mode, cap in captures.items()}
        if len(set(counts.values())) != 1 or any(
                abs(value - 30.0) > 1e-6 for value in fps_values.values()):
            raise ValueError("four panels must have the same 30 fps schedule")
        frame_count = next(iter(counts.values()))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            30.0, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError("cannot open four-panel MP4 encoder")
        try:
            for frame_index in range(frame_count):
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                for panel_index, mode in enumerate(STUDY_PANEL_ORDER):
                    ok, frame = captures[mode].read()
                    if not ok:
                        raise RuntimeError(
                            f"cannot decode {mode} frame {frame_index}")
                    frame = cv2.resize(frame, (640, 320),
                                       interpolation=cv2.INTER_AREA)
                    x = (panel_index % 2) * 640
                    y = (panel_index // 2) * 360
                    color = hex_to_bgr(MOUNT_COLORS[mode])
                    cv2.rectangle(canvas, (x, y), (x + 639, y + 39),
                                  color, -1)
                    cv2.putText(
                        canvas,
                        f"{MOUNT_LABELS[mode]}  |  {MOUNT_AXIS_LABELS[mode]}",
                        (x + 12, y + 26), cv2.FONT_HERSHEY_SIMPLEX,
                        .56, (255, 255, 255), 2, cv2.LINE_AA)
                    canvas[y + 40:y + 360, x:x + 640] = frame
                    cv2.rectangle(canvas, (x, y), (x + 639, y + 359),
                                  color, 3)
                cv2.putText(
                    canvas, trajectory_label, (500, 714),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (225, 225, 225),
                    1, cv2.LINE_AA)
                writer.write(canvas)
        finally:
            writer.release()
    finally:
        for capture in captures.values():
            capture.release()
    return decode_check_mp4(
        output_path, expected_frames=frame_count,
        expected_resolution=(1280, 720))


def render_trajectory(output_root, trajectory):
    manifest = json.loads(
        (Path(output_root) / "bundle_manifest.json").read_text(encoding="utf-8"))
    rows = {item["mode"]: item for item in manifest["shards"]
            if item["trajectory"] == trajectory}
    if set(rows) != set(STUDY_PANEL_ORDER):
        raise ValueError(f"{trajectory}: four validated shards are required")
    safe_name = trajectory.replace("/", "_")
    panel_dir = Path(output_root) / "videos" / "panels" / safe_name
    panel_paths = {}
    for mode in STUDY_PANEL_ORDER:
        panel = panel_dir / f"{safe_name}_{mode}.mp4"
        summary = Path(rows[mode]["summary_json"])
        if not summary.is_absolute():
            summary = ROOT / summary
        if not panel.exists() or not _cached_panel_is_valid(panel, summary):
            render_panel(summary, panel)
            _write_panel_provenance(panel, summary)
        panel_paths[mode] = panel
    output = (Path(output_root) / "videos" / "comparisons"
              / f"{safe_name}_four_mount_fixed_time.mp4")
    check = compose_four_panel(
        panel_paths, output, trajectory_label=trajectory)
    _write_composite_provenance(output, panel_paths, trajectory, check)
    return output, check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory")
    args = parser.parse_args(argv)
    manifest = json.loads(
        (args.output / "bundle_manifest.json").read_text(encoding="utf-8"))
    trajectories = sorted({item["trajectory"] for item in manifest["shards"]})
    if args.trajectory:
        trajectories = [item for item in trajectories
                        if item == args.trajectory]
    if not trajectories:
        raise ValueError("no trajectory matched the rendering filter")
    for trajectory in trajectories:
        output, check = render_trajectory(args.output, trajectory)
        print(output, check.frame_count, flush=True)


if __name__ == "__main__":
    main()
