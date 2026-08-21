"""Recompute and validate the published Fold_Box + Seal_Bag evidence bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "reports/piperx_two_task_complete_follow"
EXPECTED = {
    "fold_box": {"family": "8-11/Fold_Box", "take": "161044", "frames": 1061},
    "seal_bag": {"family": "8-11/Seal_Bag", "take": "161504", "frames": 1757},
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_close(actual, expected, label, *, atol=1e-9):
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=atol):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def _artifact_path(summary_path, record):
    candidate = summary_path.parent / Path(record["path"]).name
    if not candidate.is_file():
        raise ValueError(f"artifact is missing: {candidate}")
    if _sha256(candidate) != record["sha256"]:
        raise ValueError(f"artifact SHA-256 mismatch: {candidate.name}")
    if candidate.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"artifact size mismatch: {candidate.name}")
    return candidate


def _decode_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"video cannot be opened: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return {
        "frame_count": count,
        "fps": fps,
        "duration_s": max(0, count - 1) / fps,
        "width": width,
        "height": height,
    }


def validate_task_bundle(
        summary_path, *, family, take, frames, decode_video=True):
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "piperx-complete-follow-v2":
        raise ValueError("unexpected summary schema")
    if summary["mount"]["family"] != family:
        raise ValueError("task family mismatch")
    if take not in Path(summary["source"]["path"]).stem:
        raise ValueError("source take mismatch")
    if int(summary["source"]["frames_60hz"]) != int(frames):
        raise ValueError("source frame count mismatch")
    source = summary["source"]
    if source["target_basis"] != "registered_resampled_calibrated_tcp":
        raise ValueError("published target basis must be the calibrated TCP")
    if (source.get("raw_hand_trace_preserved") is not True
            or source.get("tracking_reference")
            != "task-level calibrated PiperX TCP"):
        raise ValueError("published source evidence is incomplete")
    tool_frame = summary.get("tool_frame", {})
    if (tool_frame.get("translation_coordinate_frame")
            != "registered source-hand local"
            or tool_frame.get("tracking_error_reference")
            != "calibrated TCP target"):
        raise ValueError("published tool-frame evidence is incomplete")
    if summary["source"].get("sha256"):
        date, task = family.split("/", 1)
        source_matches = list(
            (ROOT / "data/factory" / date / task).glob(f"*{take}*.csv"))
        if len(source_matches) != 1:
            raise ValueError("authoritative source CSV is missing or ambiguous")
        if _sha256(source_matches[0]) != summary["source"]["sha256"]:
            raise ValueError("authoritative source CSV SHA-256 mismatch")
    _assert_close(summary["acceptance"]["position_tolerance_mm"], 1.0,
                  "position tolerance")
    _assert_close(summary["acceptance"]["orientation_tolerance_deg"], 0.5,
                  "orientation tolerance")

    artifacts = {
        name: _artifact_path(summary_path, record)
        for name, record in summary["artifacts"].items()
    }
    trajectory_path = artifacts["trajectory_npz"]
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        required_evidence = {
            "raw_left_hand_position_m",
            "raw_right_hand_position_m",
            "raw_left_hand_quaternion_wxyz",
            "raw_right_hand_quaternion_wxyz",
            "left_target_position_m",
            "right_target_position_m",
            "left_wrist_adaptation_angle_deg",
            "right_wrist_adaptation_angle_deg",
        }
        if not required_evidence.issubset(trajectory.files):
            raise ValueError("raw hand evidence is missing from trajectory")
        source_time = np.asarray(trajectory["source_time_s"], dtype=float)
        execution_time = np.asarray(trajectory["execution_time_s"], dtype=float)
        reached = np.asarray(trajectory["source_reached"], dtype=bool)
        fixed = np.asarray(trajectory["fixed_time_accepted"], dtype=bool)
        collision = np.asarray(trajectory["collision"], dtype=bool)
        execution_collision = np.asarray(
            trajectory["execution_collision"], dtype=bool)
        if len(source_time) != frames or len(reached) != frames:
            raise ValueError("trajectory source frame count mismatch")
        if len(execution_time) != int(summary["protocol"]["execution_frames"]):
            raise ValueError("trajectory execution frame count mismatch")
        if not np.all(np.diff(source_time) > 0.0):
            raise ValueError("source timeline is not strictly increasing")
        if not np.all(np.diff(execution_time) > 0.0):
            raise ValueError("execution timeline is not strictly increasing")
        metrics = summary["metrics"]
        checks = {
            "complete source pose": (np.count_nonzero(reached),
                                     metrics["complete_source_pose_frames"]),
            "fixed time": (np.count_nonzero(fixed),
                           metrics["fixed_time_synchronous_frames"]),
            "source collision": (np.count_nonzero(collision),
                                 metrics["collision_frames"]),
            "execution collision": (np.count_nonzero(execution_collision),
                                    metrics["execution_collision_frames"]),
        }
        for label, (actual, expected) in checks.items():
            if int(actual) != int(expected):
                raise ValueError(f"{label} frame count mismatch")
        if (not np.all(reached) or np.any(collision)
                or np.any(execution_collision)):
            raise ValueError(
                "published bundle requires complete reach and zero collision")
        if not metrics.get("retimed_execution_dynamic_limits_passed", False):
            raise ValueError("published execution violates dynamic limits")
        for side in ("left", "right"):
            raw_position = np.asarray(
                trajectory[f"raw_{side}_hand_position_m"], dtype=float)
            raw_quaternion = np.asarray(
                trajectory[f"raw_{side}_hand_quaternion_wxyz"], dtype=float)
            target_position = np.asarray(
                trajectory[f"{side}_target_position_m"], dtype=float)
            if (raw_position.shape != (frames, 3)
                    or raw_quaternion.shape != (frames, 4)
                    or target_position.shape != (frames, 3)):
                raise ValueError("raw hand evidence shape mismatch")
            translation = np.asarray(
                tool_frame[f"{side}_translation_m"], dtype=float)
            if translation.shape != (3,) or not np.isfinite(translation).all():
                raise ValueError("tool translation evidence is invalid")
            displacement = np.linalg.norm(target_position - raw_position, axis=1)
            if not np.allclose(
                    displacement, np.linalg.norm(translation),
                    rtol=0.0, atol=1e-9):
                raise ValueError("calibrated TCP translation evidence mismatch")
        adaptation = tool_frame.get("wrist_adaptation")
        for side in ("left", "right"):
            angles = np.asarray(
                trajectory[f"{side}_wrist_adaptation_angle_deg"], dtype=float)
            if angles.shape != (frames,) or not np.isfinite(angles).all():
                raise ValueError("wrist adaptation evidence shape mismatch")
            if np.max(np.abs(angles), initial=0.0) > 15.0 + 1e-12:
                raise ValueError("wrist adaptation exceeds 15 degrees")
        if adaptation is None:
            if (np.any(trajectory["left_wrist_adaptation_angle_deg"])
                    or np.any(trajectory["right_wrist_adaptation_angle_deg"])):
                raise ValueError("unexpected wrist adaptation evidence")
        else:
            side = adaptation["side"]
            expected_angle = np.zeros(frames, dtype=float)
            angle = float(adaptation["angle_deg"])
            hold = float(adaptation["hold_until_s"])
            end = float(adaptation["return_until_s"])
            expected_angle[source_time <= hold] = angle
            returning = (source_time > hold) & (source_time < end)
            expected_angle[returning] = (
                angle * (end - source_time[returning]) / (end - hold))
            if not np.allclose(
                    trajectory[f"{side}_wrist_adaptation_angle_deg"],
                    expected_angle, rtol=0.0, atol=1e-12):
                raise ValueError("wrist adaptation evidence mismatch")
        error_fields = (
            ("left_position_error_m", "left_position_error_mm", 1000.0),
            ("right_position_error_m", "right_position_error_mm", 1000.0),
            ("left_orientation_error_rad", "left_orientation_error_deg",
             180.0 / np.pi),
            ("right_orientation_error_rad", "right_orientation_error_deg",
             180.0 / np.pi),
        )
        for array_name, summary_name, scale in error_fields:
            actual = scale * float(np.max(trajectory[array_name]))
            _assert_close(actual, metrics[summary_name]["max"], summary_name)
        if np.max(trajectory["left_position_error_m"]) > 0.001 + 1e-12:
            raise ValueError("left position tolerance exceeded")
        if np.max(trajectory["right_position_error_m"]) > 0.001 + 1e-12:
            raise ValueError("right position tolerance exceeded")
        if np.max(trajectory["left_orientation_error_rad"]) > np.deg2rad(0.5) + 1e-12:
            raise ValueError("left orientation tolerance exceeded")
        if np.max(trajectory["right_orientation_error_rad"]) > np.deg2rad(0.5) + 1e-12:
            raise ValueError("right orientation tolerance exceeded")
        if "execution_state_collision" in trajectory.files:
            state = np.asarray(trajectory["execution_state_collision"], bool)
            incoming = np.asarray(
                trajectory["execution_incoming_transition_collision"], bool)
        else:
            state = incoming = None

    provenance = json.loads(
        artifacts["video_provenance_json"].read_text(encoding="utf-8"))
    if provenance.get("schema_version") != 2:
        raise ValueError("unexpected video provenance schema")
    if provenance.get("timeline_domain") != "execution":
        raise ValueError("video is not rendered on the execution timeline")
    if len(provenance["execution_knots"]) != len(execution_time):
        raise ValueError("provenance execution knot count mismatch")
    if len(provenance["encoded_frames"]) != int(
            provenance["decode_check"]["frame_count"]):
        raise ValueError("encoded frame provenance count mismatch")
    if state is not None:
        knot_state = np.asarray([
            bool(item["state_collision"])
            for item in provenance["execution_knots"]])
        knot_incoming = np.asarray([
            bool(item["incoming_transition_collision"])
            for item in provenance["execution_knots"]])
        if not np.array_equal(knot_state, state):
            raise ValueError("provenance state collision mismatch")
        if not np.array_equal(knot_incoming, incoming):
            raise ValueError("provenance incoming edge collision mismatch")

    expected_decode = summary["video"]["decode_check"]
    for field in ("frame_count", "fps", "duration_s"):
        _assert_close(provenance["decode_check"][field],
                      expected_decode[field], f"provenance video {field}")
    decoded = provenance["decode_check"]
    if decode_video:
        decoded = _decode_video(artifacts["video_mp4"])
        for field in ("frame_count", "fps", "duration_s", "width", "height"):
            _assert_close(decoded[field], expected_decode[field],
                          f"decoded video {field}", atol=1e-6)

    return {
        "task": family.split("/", 1)[1],
        "family": family,
        "take": take,
        "source_frames": frames,
        "pose_frames": int(np.count_nonzero(reached)),
        "pose_coverage": float(np.mean(reached)),
        "fixed_time_frames": int(np.count_nonzero(fixed)),
        "fixed_time_coverage": float(np.mean(fixed)),
        "collision_frames": int(np.count_nonzero(collision)),
        "collision_free_coverage": float(np.mean(~collision)),
        "source_duration_s": float(summary["source"]["duration_s"]),
        "execution_duration_s": float(summary["protocol"]["retimed_duration_s"]),
        "cycle_delay_s": float(summary["protocol"]["cycle_delay_s"]),
        "maximum_velocity_rad_s": float(summary["protocol"]["dynamic_limits"]
                                         ["measured_execution_maximum_velocity_rad_s"]),
        "maximum_acceleration_rad_s2": float(summary["protocol"]["dynamic_limits"]
                                               ["measured_execution_maximum_acceleration_rad_s2"]),
        "left_max_position_mm": float(metrics["left_position_error_mm"]["max"]),
        "right_max_position_mm": float(metrics["right_position_error_mm"]["max"]),
        "left_max_orientation_deg": float(metrics["left_orientation_error_deg"]["max"]),
        "right_max_orientation_deg": float(metrics["right_orientation_error_deg"]["max"]),
        "video_frames": int(decoded["frame_count"]),
        "video_fps": float(decoded["fps"]),
        "video_duration_s": float(decoded["duration_s"]),
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path),
                   "size_bytes": path.stat().st_size}
            for name, path in artifacts.items()
        },
    }


def validate_bundle(bundle_root, *, decode_video=True):
    bundle_root = Path(bundle_root).resolve()
    records = []
    for slug, expected in EXPECTED.items():
        summaries = list((bundle_root / slug).glob("*.summary.json"))
        if len(summaries) != 1:
            raise ValueError(f"{slug}: expected exactly one summary")
        records.append(validate_task_bundle(
            summaries[0], decode_video=decode_video, **expected))
    return records


def write_outputs(bundle_root, records):
    bundle_root = Path(bundle_root).resolve()
    manifest_path = bundle_root / "two_task_manifest.json"
    csv_path = bundle_root / "two_task_summary.csv"
    portable_records = json.loads(json.dumps(records))
    for record in portable_records:
        record["summary"] = Path(record["summary"]).resolve().relative_to(
            ROOT).as_posix()
        for artifact in record["artifacts"].values():
            artifact["path"] = Path(artifact["path"]).resolve().relative_to(
                ROOT).as_posix()
    manifest = {
        "schema": "piperx-two-task-complete-follow-v1",
        "acceptance": {"position_tolerance_mm": 1.0,
                       "orientation_tolerance_deg": 0.5},
        "tasks": portable_records,
        "interpretation": {
            "complete_follow": "all calibrated TCP targets derived from the registered raw 60 Hz hand trace are reached after lossless retiming",
            "fixed_time": "same poses under original source timestamps",
            "collision": "independent MuJoCo state and swept incoming-edge audit; not a hardware safety approval",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    fields = [key for key in records[0] if key not in {
        "artifacts", "summary", "summary_sha256"}]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: record[key] for key in fields}
                         for record in records)
    return manifest_path, csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--skip-video-decode", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    records = validate_bundle(
        options.bundle_root, decode_video=not options.skip_video_decode)
    manifest, csv_path = write_outputs(options.bundle_root, records)
    print(json.dumps({
        "manifest": str(manifest),
        "summary_csv": str(csv_path),
        "tasks": records,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
