"""Build the audited Chinese report for PiperX complete Cartesian follow."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from factory_bimanual.mujoco_collision_adapter import (
    MuJoCoPairedCollisionChecker,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.video import (
    build_realtime_timing,
    decode_check_mp4,
    interpolation_sample,
)
from scripts.build_piperx_recommended_v31_report import (
    PALETTE,
    _paragraph,
    _register_fonts,
    _styles,
    _table,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "reports/piperx_complete_follow"
STEM = "8-11_Fold_Box_161044_complete_follow"
SUMMARY_PATH = ARTIFACT_DIR / f"{STEM}.summary.json"
TRAJECTORY_PATH = ARTIFACT_DIR / f"{STEM}.trajectory.npz"
SCENE_PATH = ARTIFACT_DIR / f"{STEM}.scene.xml"
VIDEO_PATH = ARTIFACT_DIR / f"{STEM}.mp4"
DEFAULT_OUTPUT = ARTIFACT_DIR / "PiperX双臂完全跟随优化与验证报告_v4.0.pdf"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--trajectory", type=Path, default=TRAJECTORY_PATH)
    parser.add_argument("--scene", type=Path, default=SCENE_PATH)
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _header_footer(canvas, document):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#D4DEE3"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont("YaHei", 7)
    canvas.setFillColor(PALETTE["muted"])
    canvas.drawString(18 * mm, 9 * mm, "PiperX 双臂完全跟随优化与验证报告 · v4.0")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def _metric_cards(summary, styles):
    metrics = summary["metrics"]
    source_frames = summary["source"]["frames_60hz"]
    cards = [
        (f"{metrics['complete_source_pose_frames']}/{source_frames}",
         "原始源位姿严格命中"),
        (f"{metrics['fixed_time_synchronous_frames']}/{source_frames}",
         "原 60 Hz 节拍可执行"),
        ("PASS" if metrics["retimed_execution_dynamic_limits_passed"]
         else "FAIL", "重定时动力学约束"),
        (f"{100*metrics['collision_free_strict_coverage']:.2f}%",
         "无碰撞严格覆盖"),
    ]
    cells = []
    for value, label in cards:
        cells.append([
            Paragraph(value, styles["metric"]),
            Paragraph(label, styles["metric_label"]),
        ])
    table = Table([cells], colWidths=[45 * mm] * 4)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALETTE["pale"]),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B4C7D0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _collision_audit(scene_path: Path, arrays):
    model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
    data = mujoco.MjData(model)
    contract = ROBOT_CONTRACTS["piperx"]
    names = {
        side: {
            "joints": contract.prefixed_joint_names(side),
            "site": f"{side}_tcp",
        }
        for side in ("left", "right")
    }
    checker = MuJoCoPairedCollisionChecker(
        model, data, names, transition_steps=11)
    qids = {
        side: np.asarray([
            model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in names[side]["joints"]
        ], dtype=int)
        for side in names
    }
    qpos = arrays["source_qpos"]
    state_counts = Counter()
    transition_counts = Counter()
    for row in np.flatnonzero(arrays["collision"]):
        left = qpos[row, qids["left"]]
        right = qpos[row, qids["right"]]
        state = checker.state(left, right)
        state_counts.update(item.value for item in state.classes)
        if row:
            previous = (
                qpos[row-1, qids["left"]],
                qpos[row-1, qids["right"]],
            )
            transition = checker.transition(previous, (left, right))
        else:
            transition = state
        transition_counts.update(item.value for item in transition.classes)
    return state_counts, transition_counts


def _collision_runs(mask):
    runs = []
    start = None
    for row, collision in enumerate(np.asarray(mask, dtype=bool)):
        if collision and start is None:
            start = row
        if start is not None and (not collision or row == len(mask)-1):
            end = row if collision and row == len(mask)-1 else row-1
            runs.append((start, end))
            start = None
    return runs


def _make_figures(summary, arrays, directory):
    directory.mkdir(parents=True, exist_ok=True)
    comparison = directory / "report_coverage_comparison.png"
    envelope = directory / "report_strict_error_envelope.png"
    collision_path = directory / "report_collision_timeline.png"

    metrics = summary["metrics"]
    labels = ["旧 v3.1\n严格同步", "v4.0\n原始位姿完整", "v4.0\n无碰撞严格"]
    values = [10.556, 100*metrics["complete_source_pose_coverage"],
              100*metrics["collision_free_strict_coverage"]]
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=180)
    bars = ax.bar(labels, values, color=["#9AA9B0", "#2A9D6F", "#E98A2E"], width=.62)
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=3)
    ax.set_ylim(0, 108)
    ax.set_ylabel("覆盖率 / %")
    ax.set_title("同一 Fold_Box 161044：严格覆盖率改进")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(comparison, facecolor="white")
    plt.close(fig)

    frame = np.arange(len(arrays["source_time_s"]))
    series = [
        1000*arrays["left_position_error_m"],
        1000*arrays["right_position_error_m"],
        np.rad2deg(arrays["left_orientation_error_rad"]),
        np.rad2deg(arrays["right_orientation_error_rad"]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.6), dpi=180, sharex=True)
    axes[0].plot(frame, series[0], lw=.8, label="左臂")
    axes[0].plot(frame, series[1], lw=.8, label="右臂")
    axes[0].axhline(1.0, color="#D1495B", ls="--", lw=1, label="1 mm 门槛")
    axes[0].set_ylabel("位置误差 / mm")
    axes[1].plot(frame, series[2], lw=.8, label="左臂")
    axes[1].plot(frame, series[3], lw=.8, label="右臂")
    axes[1].axhline(.5, color="#D1495B", ls="--", lw=1, label="0.5° 门槛")
    axes[1].set_ylabel("姿态误差 / °")
    axes[1].set_xlabel("60 Hz 源帧")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.18)
        axis.legend(frameon=False, ncol=3, fontsize=7, loc="upper left")
    fig.suptitle(f"{len(frame)} 个源位姿的 MuJoCo 正运动学复核")
    fig.tight_layout()
    fig.savefig(envelope, facecolor="white")
    plt.close(fig)

    collision = np.asarray(arrays["collision"], dtype=bool)
    fig, ax = plt.subplots(figsize=(8.4, 1.8), dpi=180)
    ax.fill_between(frame, 0, collision.astype(int), step="mid", color="#D1495B")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1], ["安全", "碰撞"])
    ax.set_xlabel("60 Hz 源帧")
    ax.set_title("状态 + 11 子步扫掠碰撞审计")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    fig.tight_layout()
    fig.savefig(collision_path, facecolor="white")
    plt.close(fig)
    return comparison, envelope, collision_path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _collision_masks(checker, qids, qpos):
    state_collision = np.zeros(len(qpos), dtype=bool)
    incoming_collision = np.zeros(len(qpos), dtype=bool)
    previous_pair = None
    for row, values in enumerate(qpos):
        pair = tuple(values[qids[side]] for side in ("left", "right"))
        state_collision[row] = not checker.state(*pair).valid
        if row:
            incoming_collision[row] = not checker.transition(
                previous_pair, pair).valid
        previous_pair = tuple(value.copy() for value in pair)
    return state_collision, incoming_collision


def _scene_recompute(scene_path, arrays):
    model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))
    data = mujoco.MjData(model)
    contract = ROBOT_CONTRACTS["piperx"]
    names = {
        side: {
            "joints": contract.prefixed_joint_names(side),
            "site": f"{side}_tcp",
        }
        for side in ("left", "right")
    }
    checker = MuJoCoPairedCollisionChecker(
        model, data, names, transition_steps=11)
    qids = {
        side: np.asarray([
            model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in names[side]["joints"]
        ], dtype=int)
        for side in names
    }
    site_ids = {
        side: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, names[side]["site"])
        for side in names
    }
    source_qpos = np.asarray(arrays["source_qpos"], dtype=float)
    output = {
        side: {
            "position_error_m": np.zeros(len(source_qpos)),
            "orientation_error_rad": np.zeros(len(source_qpos)),
        }
        for side in names
    }
    source_state_collision, source_incoming_collision = _collision_masks(
        checker, qids, source_qpos)
    execution_qpos = np.asarray(arrays["execution_qpos"], dtype=float)
    execution_state_collision, execution_incoming_collision = _collision_masks(
        checker, qids, execution_qpos)
    for row, qpos in enumerate(source_qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for side in names:
            site = site_ids[side]
            target_position = np.asarray(
                arrays[f"{side}_target_position_m"][row], dtype=float)
            target_quaternion = np.asarray(
                arrays[f"{side}_target_quaternion_wxyz"][row], dtype=float)
            actual_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(actual_quaternion, data.site_xmat[site])
            residual = np.empty(3)
            mujoco.mju_subQuat(
                residual, target_quaternion, actual_quaternion)
            output[side]["position_error_m"][row] = np.linalg.norm(
                target_position - data.site_xpos[site])
            output[side]["orientation_error_rad"][row] = np.linalg.norm(
                residual)
    return (
        output,
        source_state_collision | source_incoming_collision,
        execution_state_collision,
        execution_incoming_collision,
    )


def _execution_derivatives_with_zero_boundaries(qpos, time_s):
    q = np.asarray(qpos, dtype=float)
    time = np.asarray(time_s, dtype=float)
    dt = np.diff(time)
    velocity = np.diff(q, axis=0) / dt[:, None]
    if not len(velocity):
        return velocity, np.empty((0, q.shape[1]), dtype=float)
    start = velocity[0] / (0.5 * dt[0])
    end = -velocity[-1] / (0.5 * dt[-1])
    if len(velocity) == 1:
        acceleration = np.vstack((start, end))
    else:
        internal = np.diff(velocity, axis=0) / (
            0.5 * (dt[:-1] + dt[1:]))[:, None]
        acceleration = np.vstack((start, internal, end))
    return velocity, acceleration


def _validate_evidence(
        summary, arrays, summary_path, trajectory_path, scene_path, video_path):
    """Independently bind and recompute the complete report evidence chain."""
    if summary.get("schema") != "piperx-complete-follow-v2":
        raise ValueError("report requires piperx-complete-follow-v2 evidence")
    if summary["source"].get("target_basis") != "registered_resampled_raw":
        raise ValueError("report requires registered raw targets")
    if summary["protocol"]["target_conditioning"].get("enabled"):
        raise ValueError("final complete-follow report requires raw targets")
    # This is intentionally a task-specific report, so reject inputs that
    # would make its historical comparisons and failure-window text false.
    if (summary["mount"].get("family") != "8-11/Fold_Box"
            or summary["source"]["frames_60hz"] != 1061
            or Path(summary["source"]["path"]).name
            != "handheld_20260811_161044.csv"
            or summary["protocol"].get("maximum_candidates_per_side") != 8):
        raise ValueError("v4.0 report only accepts the audited Fold_Box 161044 run")

    required = {
        "source_time_s", "source_qpos", "execution_time_s",
        "execution_qpos", "source_index", "execution_state", "source_reached",
        "fixed_time_accepted", "collision", "execution_collision",
        "execution_state_collision",
        "execution_incoming_transition_collision",
        "left_target_position_m", "right_target_position_m",
        "left_target_quaternion_wxyz", "right_target_quaternion_wxyz",
        "left_position_error_m", "right_position_error_m",
        "left_orientation_error_rad", "right_orientation_error_rad",
    }
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"trajectory evidence is missing: {sorted(missing)}")

    source_path = Path(summary["source"]["path"])
    if not source_path.is_file() or _sha256(source_path) != summary["source"]["sha256"]:
        raise ValueError("source trajectory SHA-256 does not match summary")
    provenance_path = Path(video_path).with_suffix(".provenance.json")
    expected_paths = {
        "trajectory_npz": Path(trajectory_path),
        "scene_xml": Path(scene_path),
        "video_mp4": Path(video_path),
        "video_provenance_json": provenance_path,
    }
    manifest = summary.get("artifacts", {})
    for name, path in expected_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        record = manifest.get(name)
        if (record is None or Path(record["path"]).resolve() != path.resolve()
                or int(record["size_bytes"]) != path.stat().st_size
                or record["sha256"] != _sha256(path)):
            raise ValueError(f"artifact manifest mismatch: {name}")
    if (Path(summary["scene_manifest"]["output_xml"]).resolve()
            != Path(scene_path).resolve()):
        raise ValueError("scene manifest points to a different XML")

    frames = int(summary["source"]["frames_60hz"])
    if len(arrays["source_time_s"]) != frames:
        raise ValueError("source frame count differs between JSON and NPZ")
    metrics = summary["metrics"]
    reached = np.asarray(arrays["source_reached"], dtype=bool)
    fixed = np.asarray(arrays["fixed_time_accepted"], dtype=bool)
    collision = np.asarray(arrays["collision"], dtype=bool)
    if len(reached) != frames or not reached.all():
        raise ValueError("raw source poses are not completely reached")
    if int(np.count_nonzero(reached)) != metrics["complete_source_pose_frames"]:
        raise ValueError("complete source-pose count differs from NPZ")
    if int(np.count_nonzero(fixed)) != metrics["fixed_time_synchronous_frames"]:
        raise ValueError("fixed-time count differs between JSON and NPZ")

    (recomputed, recomputed_collision, recomputed_execution_state_collision,
     recomputed_execution_incoming_collision) = _scene_recompute(
         scene_path, arrays)
    if not np.array_equal(recomputed_collision, collision):
        raise ValueError("scene recomputation changed the collision mask")
    if int(np.count_nonzero(collision)) != metrics["collision_frames"]:
        raise ValueError("collision count differs between JSON and NPZ")
    stored_execution_state_collision = np.asarray(
        arrays["execution_state_collision"], dtype=bool)
    stored_execution_incoming_collision = np.asarray(
        arrays["execution_incoming_transition_collision"], dtype=bool)
    stored_execution_collision = np.asarray(
        arrays["execution_collision"], dtype=bool)
    if (not np.array_equal(
            recomputed_execution_state_collision,
            stored_execution_state_collision)
            or not np.array_equal(
                recomputed_execution_incoming_collision,
                stored_execution_incoming_collision)
            or not np.array_equal(
                stored_execution_collision,
                stored_execution_state_collision
                | stored_execution_incoming_collision)):
        raise ValueError("scene recomputation changed execution collision evidence")
    if int(np.count_nonzero(stored_execution_collision)) != metrics[
            "execution_collision_frames"]:
        raise ValueError("execution collision count differs between JSON and NPZ")
    position_limit_m = summary["acceptance"]["position_tolerance_mm"] / 1000.0
    orientation_limit_rad = np.deg2rad(
        summary["acceptance"]["orientation_tolerance_deg"])
    for side in ("left", "right"):
        position = recomputed[side]["position_error_m"]
        orientation = recomputed[side]["orientation_error_rad"]
        if (not np.allclose(
                position, arrays[f"{side}_position_error_m"],
                rtol=0.0, atol=1e-12)
                or not np.allclose(
                    orientation, arrays[f"{side}_orientation_error_rad"],
                    rtol=0.0, atol=1e-12)):
            raise ValueError(f"{side} stored FK audit differs from scene recomputation")
        if float(np.max(position)) > position_limit_m + 1e-12:
            raise ValueError(f"{side} position tolerance is not satisfied")
        if float(np.max(orientation)) > orientation_limit_rad + 1e-12:
            raise ValueError(f"{side} orientation tolerance is not satisfied")
        position_summary = metrics[f"{side}_position_error_mm"]
        orientation_summary = metrics[f"{side}_orientation_error_deg"]
        if (not np.isclose(position_summary["max"], 1000*np.max(position), atol=1e-9)
                or not np.isclose(position_summary["mean"], 1000*np.mean(position), atol=1e-9)
                or not np.isclose(orientation_summary["max"], np.rad2deg(np.max(orientation)), atol=1e-9)
                or not np.isclose(orientation_summary["mean"], np.rad2deg(np.mean(orientation)), atol=1e-9)):
            raise ValueError(f"{side} summary FK statistics differ from recomputation")

    execution_time = np.asarray(arrays["execution_time_s"], dtype=float)
    execution_q = np.asarray(arrays["execution_qpos"], dtype=float)
    velocity, acceleration = _execution_derivatives_with_zero_boundaries(
        execution_q, execution_time)
    limits = summary["protocol"]["dynamic_limits"]
    measured_velocity = float(np.max(np.abs(velocity), initial=0.0))
    measured_acceleration = float(np.max(np.abs(acceleration), initial=0.0))
    if measured_velocity > limits["maximum_velocity_rad_s"] + 1e-10:
        raise ValueError("retimed execution violates velocity limit")
    if measured_acceleration > limits["maximum_acceleration_rad_s2"] + 1e-9:
        raise ValueError("retimed execution violates acceleration limit")
    if (not np.isclose(
            measured_velocity,
            limits["measured_execution_maximum_velocity_rad_s"], atol=1e-9)
            or not np.isclose(
                measured_acceleration,
                limits["measured_execution_maximum_acceleration_rad_s2"],
                atol=1e-9)):
        raise ValueError("summary dynamic extrema differ from recomputation")

    video = Path(video_path)
    if Path(summary["video"]["path"]).resolve() != video.resolve():
        raise ValueError("summary points to a different video")
    expected_decode = summary["video"]["decode_check"]
    decoded = decode_check_mp4(
        video,
        expected_frames=int(expected_decode["frame_count"]),
        expected_resolution=(
            int(expected_decode["width"]), int(expected_decode["height"])),
        expected_duration_s=float(expected_decode["duration_s"]),
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    execution_knots = provenance.get("execution_knots", [])
    encoded_frames = provenance.get("encoded_frames", [])
    if (provenance.get("schema_version") != 2
            or provenance.get("timeline_domain") != "execution"
            or not provenance.get("timing_method", "").startswith(
                "constant-fps linear interpolation")
            or len(execution_knots) != len(execution_time)
            or len(encoded_frames) != decoded.frame_count
            or encoded_frames[-1]["source_row_index"] != frames - 1
            or encoded_frames[-1]["execution_index"]
            != len(execution_time) - 1):
        raise ValueError("video provenance timeline is incomplete or mismatched")
    source_index = np.asarray(arrays["source_index"], dtype=int)
    execution_state = np.asarray(arrays["execution_state"]).astype(str)
    if not (
            len(source_index) == len(execution_time)
            == len(execution_state)
            == len(stored_execution_collision)):
        raise ValueError("execution arrays have inconsistent lengths")
    expected_knot_state = np.asarray([
        (f"{state}_COLLISION" if collision_flag else state)
        for state, collision_flag in zip(
            execution_state, stored_execution_collision)
    ])
    for index, row in enumerate(execution_knots):
        if (int(row["execution_index"]) != index
                or not np.isclose(
                    float(row["execution_time_s"]), execution_time[index],
                    rtol=0.0, atol=1e-12)
                or int(row["source_row_index"]) != source_index[index]
                or str(row["execution_state"]) != expected_knot_state[index]
                or bool(row["collision"]) != stored_execution_collision[index]
                or bool(row["state_collision"])
                != stored_execution_state_collision[index]
                or bool(row["incoming_transition_collision"])
                != stored_execution_incoming_collision[index]):
            raise ValueError("video provenance execution knots differ from NPZ")

    timing = build_realtime_timing(execution_time, float(provenance["fps"]))
    encoded_qpos = np.empty((len(encoded_frames), execution_q.shape[1]))
    for index, row in enumerate(encoded_frames):
        encoded_time = float(row["encoded_time_s"])
        lower, upper, alpha = interpolation_sample(execution_time, encoded_time)
        moving = upper != lower and alpha > 1e-12
        audit_index = upper if moving else lower
        expected_collision = bool(
            stored_execution_incoming_collision[audit_index]
            if moving else stored_execution_state_collision[audit_index])
        expected_state = str(execution_state[audit_index])
        if expected_collision:
            expected_state += "_COLLISION"
        if (int(row["encoded_frame_index"]) != index
                or not np.isclose(
                    encoded_time, timing.encoded_time_s[index],
                    rtol=0.0, atol=1e-12)
                or int(row["interpolation_lower_index"]) != lower
                or int(row["interpolation_upper_index"]) != upper
                or not np.isclose(
                    float(row["interpolation_alpha"]), alpha,
                    rtol=0.0, atol=1e-12)
                or int(row["audit_execution_index"]) != audit_index
                or str(row["audit_scope"])
                != ("incoming_edge" if moving else "knot")
                or bool(row["collision"]) != expected_collision
                or str(row["execution_state"]) != expected_state):
            raise ValueError("encoded-frame provenance audit is inconsistent")
        encoded_qpos[index] = (
            (1.0-alpha)*execution_q[lower] + alpha*execution_q[upper])
    encoded_dt = np.diff(timing.encoded_time_s)
    if np.any(encoded_dt <= 0):
        raise ValueError("encoded provenance timestamps are not strictly increasing")
    encoded_vmax = float(np.max(
        np.abs(np.diff(encoded_qpos, axis=0) / encoded_dt[:, None]),
        initial=0.0))
    if encoded_vmax > limits["maximum_velocity_rad_s"] + 1e-9:
        raise ValueError("provenance-reconstructed video exceeds velocity limit")
    if Path(provenance["scene_xml"]).resolve() != Path(scene_path).resolve():
        raise ValueError("video provenance points to a different scene")
    if (decoded.frame_count != provenance["decode_check"]["frame_count"]
            or decoded.width != provenance["decode_check"]["width"]
            or decoded.height != provenance["decode_check"]["height"]):
        raise ValueError("actual MP4 decode differs from provenance")
    return {
        "measured_velocity_rad_s": measured_velocity,
        "measured_acceleration_rad_s2": measured_acceleration,
        "encoded_reconstructed_velocity_rad_s": encoded_vmax,
        "summary_path": str(Path(summary_path).resolve()),
        "provenance_path": str(provenance_path.resolve()),
        "artifact_hashes_verified": sorted(expected_paths),
    }


def build_report(summary_path, trajectory_path, scene_path, video_path, output_path):
    _register_fonts()
    styles = _styles()
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    arrays = np.load(trajectory_path, allow_pickle=False)
    evidence_audit = _validate_evidence(
        summary, arrays, summary_path, trajectory_path, scene_path, video_path)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figures = _make_figures(summary, arrays, output_path.parent)
    state_counts, transition_counts = _collision_audit(Path(scene_path), arrays)
    collision_runs = _collision_runs(arrays["collision"])
    tcp_distance = np.linalg.norm(
        arrays["left_target_position_m"]-arrays["right_target_position_m"], axis=1)
    metrics = summary["metrics"]
    mount = summary["mount"]
    tool = summary["tool_frame"]
    video = summary["video"]
    source_frames = int(summary["source"]["frames_60hz"])
    last_source_row = source_frames - 1
    execution_state_counts = Counter(
        str(value) for value in arrays["execution_state"])
    hold_frames = sum(
        count for state, count in execution_state_counts.items()
        if "HOLD" in state)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=16*mm, bottomMargin=20*mm,
        title="PiperX 双臂完全跟随优化与验证报告 v4.0",
        author="single-arm-mount project",
    )
    story = []
    p = lambda text, name="body": _paragraph(text, styles[name])

    story.extend([
        Spacer(1, 11*mm),
        p("PiperX 双臂完全跟随<br/>优化与验证报告", "title"),
        p("Fold_Box 161044 · 严格 1 mm / 0.5° · complete-follow v4.0", "subtitle"),
        Spacer(1, 8*mm),
        _metric_cards(summary, styles),
        Spacer(1, 8*mm),
        p("结论", "h1"),
        p(
            "在不放宽门槛、不丢弃位姿且不启用目标平滑的条件下，PiperX 双臂对 "
            f"8-11/Fold_Box/161044 的 <b>{metrics['complete_source_pose_frames']}/{source_frames}</b> 个注册后原始 60 Hz 位姿全部按顺序通过 "
            "MuJoCo 正运动学复核：位置误差 ≤ 1 mm、姿态误差 ≤ 0.5°。"
        ),
        p(
            "相较旧 v3.1 的 112/1061（10.56%）严格同步覆盖，本次通过任务级固定工具坐标映射、"
            "左右臂独立暖启动状态机、碰撞触发的双侧候选扩展和连续腕部分支选择，将同一轨迹的"
            f"位姿路径覆盖提升到 100%。原始 {summary['source']['duration_s']:.3f} s 节拍并不满足配置的关节动力学限制；"
            f"无损局部重定时后执行时长为 {summary['protocol']['retimed_duration_s']:.3f} s。"
        ),
        p("重要部署边界", "h2"),
        p(
            "100% 指原始位姿路径完整命中，不等于原 60 Hz 节拍可执行，也不等于 100% 可安全上真机。"
            "源轨迹存在双手交叉和腕部自碰区段；"
            f"状态与 11 子步扫掠审计记录 {metrics['collision_frames']} 帧碰撞，无碰撞严格覆盖为 "
            f"{100*metrics['collision_free_strict_coverage']:.2f}%。因此本产物是完整 IK 跟随证据，"
            "不是免碰撞实机部署许可。"
        ),
        p(
            "生成日期：2026-08-20 · Windows · "
            f"MuJoCo {mujoco.__version__} · "
            f"Python {sys.version_info.major}.{sys.version_info.minor}",
            "small",
        ),
    ])

    story.extend([
        PageBreak(),
        p("1. 三份补充报告如何改变方案", "h1"),
        _table([
            ["证据", "关键观察", "本次采用"],
            ["26 条轨迹综合报告", "底部构型赢 16/26；Fold_Box 161044 报告为 FULL / 100%", "保留底部桌面 mount；优先修复 IK 协议和工具映射"],
            ["Seal_Bag 三构型报告", "底部同步 96.5%；frame 0 为 40×200，全程左右臂独立 warm-start/HOLD", "40 个确定性全局种子；左右臂分离状态，失败只重启失败侧"],
            ["单轨迹优化分析", "SG window=9/poly=3；目标偏差限 5 mm/1°；DLS λ=.3、误差裁剪 .05 m/.3 rad、关节步 .3 rad", "保留 DLS 参数；最终验收禁用 SG，直接追踪原始注册目标"],
            ["原四臂四位姿报告", "共享 base、任务分族、分支守卫、显式 rescue", "保留按日期/任务配置、共享 base 与周期关节最短差"],
        ], [43*mm, 69*mm, 68*mm]),
        Spacer(1, 5*mm),
        p("根因诊断", "h2"),
        p(
            "旧方案把另一条任务得到的右臂 R_tool 锁定到本轨迹：右臂在 45–52、75–117 等连续窗口完全没有严格候选；"
            "双臂耦合调度又把单侧失解放大成同步失败。简单旋转底座 yaw（±90°）不改变这些代表帧的可行性，"
            "因此主因不是重启次数，也不是阈值，而是任务坐标映射与状态机耦合。"
        ),
        p("优化选择", "h2"),
        p(
            "采用“固定任务级 R_tool + 独立单臂 warm-start + 失败侧全局恢复”。R_tool 在整条轨迹上保持刚性不变，"
            "只在离线阶段沿连续腕部分支做 SLERP 可行性扫描；不会逐帧修改工具坐标来制造低误差。"
            "当暖启动配对碰撞时，左右臂同时扩展全局候选，再按状态碰撞、扫掠碰撞、关节步和腕风险排序。"
        ),
        Image(str(figures[0]), width=180*mm, height=73*mm),
    ])

    story.extend([
        PageBreak(),
        p("2. Mount 与任务级工具坐标", "h1"),
        p(
            "安装保留报告推荐的 bottom/table 共享 base，并将报告坐标通过与手轨迹相同的刚体注册变换映射到 MuJoCo 世界系。"
            "这避免了为提高单轨迹覆盖而任意移动底座，同时保持与既有报告可比。"
        ),
        _table([
            ["项目", "左臂", "右臂"],
            ["世界 base / m",
             "("+", ".join(f"{v:+.6f}" for v in [*mount["xy"]["left"], mount["shared_base_z_m"]])+")",
             "("+", ".join(f"{v:+.6f}" for v in [*mount["xy"]["right"], mount["shared_base_z_m"]])+")"],
            ["base yaw", f"{mount['yaw']['left']:.1f}°", f"{mount['yaw']['right']:.1f}°"],
            ["安装形态", mount["mode"], mount["mode"]],
            ["底座间距", f"{mount['base_distance_m']:.6f} m", f"{mount['base_distance_m']:.6f} m"],
        ], [48*mm, 66*mm, 66*mm], align="CENTER"),
        Spacer(1, 5*mm),
        p("固定 R_tool（w, x, y, z）", "h2"),
        _table([
            ["侧", "固定四元数", "连续分支选择"],
            ["左", "["+", ".join(f"{v:+.9f}" for v in tool["left_offset_quaternion_wxyz"])+"]", "旧标定 → 次可行腕分支，SLERP 0.40"],
            ["右", "["+", ".join(f"{v:+.9f}" for v in tool["right_offset_quaternion_wxyz"])+"]", "场景中点 → 次可行腕分支，SLERP 0.70"],
        ], [22*mm, 98*mm, 60*mm]),
        p(
            f"筛选不是只看某一帧：候选必须同时覆盖起点、中段、旧失败窗口 203/220/306/350/628/650 和尾段 1017–{last_source_row}；"
            f"最终仍以完整 {source_frames} 帧连续求解为唯一通过条件。"
        ),
        p("目标预处理", "h2"),
        _table([
            ["最终方法", "窗口/阶数", "允许偏离", "实测偏离"],
            [summary['protocol']['target_conditioning']['method'], "禁用", "0 mm / 0°",
             f"{summary['protocol']['target_conditioning']['measured_max_position_deviation_mm']:.4f} mm / "
             f"{summary['protocol']['target_conditioning']['measured_max_orientation_deviation_deg']:.4f}°"],
        ], [48*mm, 34*mm, 43*mm, 55*mm], align="CENTER"),
        p("单轨迹报告给出的 SG 5 mm / 1° 仅保留为可选实验开关；本最终证据明确关闭该开关，避免把平滑后目标冒充原始轨迹。", "small"),
    ])

    story.extend([
        PageBreak(),
        p("3. IK、翻腕与跟踪限制处理", "h1"),
        _table([
            ["阶段", "行为", "硬约束"],
            ["frame 0 锚定", "每臂 40 个确定性分层种子，全局 DLS 搜索", "200 迭代；严格 FK 门"],
            ["逐帧 FOLLOW", "每臂仅以上一指令 q 独立 warm-start", "λ=.3；误差裁剪 .05 m/.3 rad；每步≤.3 rad"],
            ["单侧失解", "只对失败侧重新打开全局候选；成功侧保持本分支", "不让一侧失败清空另一侧状态"],
            ["双臂配对", "暖启动配对碰撞时双侧扩展全局候选；按状态/扫掠碰撞、关节步、腕风险排序", "位姿完整性与碰撞审计分开输出"],
            ["关节分支", "周期关节用最短角差解包；整条路径固定 R_tool", "普通 FOLLOW 不逐帧翻转工具坐标"],
            ["执行重定时", "先按 .3 rad 分支守卫插值，再仅拉长超限局部时间段", "1 rad/s、4 rad/s²；源位姿顺序和数值不变"],
        ], [28*mm, 91*mm, 61*mm]),
        Spacer(1, 5*mm),
        p("翻腕方案", "h2"),
        p(
            "不再把翻腕等同于临时放宽 IK。离线 R_tool 连续扫描首先选择一条能覆盖所有代表难点的固定腕部映射；"
            "在线阶段用上一帧 q 与周期关节最短差保持连续，只在该侧失解时进行全局分支恢复。"
            f"本轨迹插入 {summary['protocol']['inserted_transition_frames']} 个分支转移状态，且该状态明确标记为 "
            "RETIMED_TRANSITION，不会计作源位姿命中。"
        ),
        p("为什么达到完全 follow", "h2"),
        _table([
            ["旧限制", "处理", "结果"],
            ["跨任务 R_tool 导致连续零候选", "任务级固定映射 + 连续 SLERP 探针", "中段与尾段全部恢复严格候选"],
            ["左右臂状态强耦合", "独立 warm-start；碰撞时协调扩展", f"全局候选扩展：左 {metrics['global_rescue_frames']['left']} 帧、右 {metrics['global_rescue_frames']['right']} 帧"],
            ["失败即 HOLD 或丢帧", "完整源路径 + 无损局部 retime", f"HOLD={hold_frames}，未到达={metrics['unreached_source_pose_frames']}，转移状态={summary['protocol']['inserted_transition_frames']}"],
            ["末端帧可能漏出视频", "最后编码帧显式映射精确源终点", f"视频末帧对应 source row {last_source_row}"],
        ], [49*mm, 83*mm, 48*mm]),
    ])

    accepted = metrics["reached_only"]
    story.extend([
        PageBreak(),
        p(f"4. {source_frames} 帧严格量化验证", "h1"),
        Image(str(figures[1]), width=180*mm, height=98.5*mm),
        Spacer(1, 3*mm),
        _table([
            ["指标", "左臂", "右臂", "门槛/结果"],
            ["最大位置误差",
             f"{accepted['left_max_position_mm']:.6f} mm",
             f"{accepted['right_max_position_mm']:.6f} mm", "≤ 1.000000 mm / 通过"],
            ["平均位置误差",
             f"{metrics['left_position_error_mm']['mean']:.6f} mm",
             f"{metrics['right_position_error_mm']['mean']:.6f} mm", "审计项"],
            ["最大姿态误差",
             f"{accepted['left_max_orientation_deg']:.6f}°",
             f"{accepted['right_max_orientation_deg']:.6f}°", "≤ 0.500000° / 通过"],
            ["平均姿态误差",
             f"{metrics['left_orientation_error_deg']['mean']:.6f}°",
             f"{metrics['right_orientation_error_deg']['mean']:.6f}°", "审计项"],
        ], [43*mm, 43*mm, 43*mm, 51*mm], align="CENTER"),
        Spacer(1, 4*mm),
        p(
            f"完整源位姿覆盖：{metrics['complete_source_pose_frames']}/{summary['source']['frames_60hz']}；"
            f"原时间轴同步覆盖：{metrics['fixed_time_synchronous_frames']}/{summary['source']['frames_60hz']}；"
            f"未到达源位姿：{metrics['unreached_source_pose_frames']}；"
            f"执行帧：{summary['protocol']['execution_frames']}；插入转移帧：{summary['protocol']['inserted_transition_frames']}；"
            f"节拍延迟：{summary['protocol']['cycle_delay_s']:.3f} s。"
        ),
        p("原节拍与重定时执行的动力学审计", "h2"),
        _table([
            ["指标", "原 60 Hz 路径", "局部重定时执行", "配置上限"],
            ["最大关节速度",
             f"{metrics['original_time_maximum_velocity_rad_s']:.3f} rad/s",
             f"{summary['protocol']['dynamic_limits']['measured_execution_maximum_velocity_rad_s']:.3f} rad/s",
             f"{summary['protocol']['dynamic_limits']['maximum_velocity_rad_s']:.3f} rad/s"],
            ["最大离散关节加速度",
             f"{metrics['original_time_maximum_acceleration_rad_s2']:.3f} rad/s²",
             f"{summary['protocol']['dynamic_limits']['measured_execution_maximum_acceleration_rad_s2']:.3f} rad/s²",
             f"{summary['protocol']['dynamic_limits']['maximum_acceleration_rad_s2']:.3f} rad/s²"],
            ["执行时长", f"{summary['source']['duration_s']:.3f} s",
             f"{summary['protocol']['retimed_duration_s']:.3f} s", "路径不丢位姿"],
        ], [44*mm, 47*mm, 49*mm, 40*mm], align="CENTER"),
        p("所有数字直接读取最终 summary JSON / trajectory NPZ，并在实际 MuJoCo 场景上重算 FK；报告构建器同时校验源 CSV 哈希、误差门、动力学上限、碰撞计数和视频末帧。", "small"),
    ])

    story.extend([
        PageBreak(),
        p("5. 碰撞审计与实机边界", "h1"),
        Image(str(figures[2]), width=180*mm, height=38.6*mm),
        Spacer(1, 4*mm),
        _table([
            ["项目", "结果"],
            ["碰撞帧 / 无碰撞严格覆盖", f"{metrics['collision_frames']} / {100*metrics['collision_free_strict_coverage']:.2f}%"],
            ["碰撞区间（源帧，含扫掠边）", ", ".join(f"{a}–{b}" for a, b in collision_runs)],
            ["状态碰撞类型计数", ", ".join(f"{key}: {value}" for key, value in state_counts.items())],
            ["扫掠碰撞类型计数", ", ".join(f"{key}: {value}" for key, value in transition_counts.items())],
            ["两目标 TCP 最小中心距", f"{1000*tcp_distance.min():.3f} mm（源帧 {int(tcp_distance.argmin())}）"],
        ], [61*mm, 119*mm]),
        Spacer(1, 5*mm),
        p("为何不能同时宣称 100% 无碰撞完全 follow", "h2"),
        p(
            f"第 121–158 帧附近，两目标末端进入同一狭小空间；在第 {int(tcp_distance.argmin())} 帧，"
            f"两目标 TCP 中心仅相距 {1000*tcp_distance.min():.2f} mm，小于两套真实夹爪碰撞体所需空间。"
            f"严格保持两个 6-DoF 目标时，每臂最多 {summary['protocol']['maximum_candidates_per_side']} 个严格候选的配对搜索仍留下这些碰撞源帧。"
            "其中 0–22 主要是右臂 link4/link6 自碰，121–158 与尾段主要是双夹爪交叉接触。"
            "因此该段若要上真机，必须改变工艺目标：错峰执行、增加手间距、重定时其中一臂或重规划 Cartesian 路径；"
            "这些都会改变原始双臂 Cartesian 任务，不能冒充原轨迹的无碰撞完全跟随。"
        ),
        p("部署建议", "h2"),
        _table([
            ["目标", "建议"],
            ["复现原轨迹位姿", "使用本 v4.0 IK 结果；仅限仿真/离线分析"],
            ["真机安全执行", "把碰撞区间交给双臂时空协调器；至少重规划 0–22、121–158、1050–1060"],
            ["仍要求 1 mm / 0.5°", "必须修改工艺示教本身；单纯换 IK 求解器无法消除两个实体占据同一空间"],
            ["上线前", "重新标定 TCP，核对 PiperX 固件 J4/J5 符号、速度/加速度、急停、桌面/工件/线缆包络"],
        ], [48*mm, 132*mm]),
    ])

    qa_paths = [Path(video_path).resolve().parent/f"qa_{name}.png"
                for name in ("start", "middle", "end")]
    decode = video["decode_check"]
    story.extend([
        PageBreak(),
        p("6. 真实 MuJoCo 渲染视频", "h1"),
        p(
            f"最终 MP4 使用 MuJoCo {mujoco.__version__} 离屏渲染、官方 PiperX URDF/网格和最终求解 qpos；"
            "画面中的蓝/紫轨迹为左右目标路径，红点为当前目标，绿点为实际 TCP。"
            "视频按重定时执行 knot 线性插值编码，包含显式 RETIMED_TRANSITION 与组合碰撞标记；"
            "插值帧审计 upper knot 的 incoming edge，精确 knot 只审计自身 state collision；"
            "它不是简化骨架动画，也不是实体机械臂录像。"
        ),
        Table([
            [Image(str(path), width=58*mm, height=32.625*mm) for path in qa_paths],
            [p("开始 · source 0", "small"), p("中段 · TRACKING", "small"), p(f"结束 · source {last_source_row}", "small")],
        ], colWidths=[60*mm]*3, style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), .4, colors.HexColor("#C8D4DA")),
            ("INNERGRID", (0, 0), (-1, -1), .25, colors.HexColor("#DDE5E9")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])),
        Spacer(1, 6*mm),
        _table([
            ["视频核验", "结果"],
            ["编码/解码", "MP4 / mp4v；OpenCV 完整解码"],
            ["帧数 / 帧率", f"{decode['frame_count']} / {decode['fps']:.1f} fps"],
            ["显示时长 / 分辨率", f"{decode['duration_s']:.2f} s / {decode['width']}×{decode['height']}"],
            ["执行时间跨度", f"{summary['protocol']['retimed_duration_s']:.6f} s；末编码帧显式绑定 source row {last_source_row}"],
            ["状态采样", video["state_sampling"]],
            ["碰撞采样", "interpolated frame: upper incoming edge；exact knot: state collision"],
            ["渲染器", video["renderer"]],
        ], [55*mm, 125*mm]),
        p(f"视频文件：{escape(str(Path(video_path).resolve()))}", "small"),
    ])

    story.extend([
        PageBreak(),
        p("7. 复现、产物与外部资料", "h1"),
        p("一键复现", "h2"),
        p("<font name='Courier'>python -m scripts.run_piperx_recommended_v31 --maximum-candidates 8 --output-dir reports/piperx_complete_follow</font>"),
        p("<font name='Courier'>python -m scripts.build_piperx_complete_follow_report</font>"),
        p("核心产物", "h2"),
        p(
            f"• {escape(str(Path(summary_path).resolve()))}<br/>"
            f"• {escape(str(Path(trajectory_path).resolve()))}<br/>"
            f"• {escape(str(Path(scene_path).resolve()))}<br/>"
            f"• {escape(str(Path(video_path).resolve()))}<br/>"
            f"• {escape(str(Path(video_path).with_suffix('.provenance.json').resolve()))}",
            "small",
        ),
        p("官方/原始资料", "h2"),
        _table([
            ["资料", "用于本项目的结论"],
            ["AgileX 官方 Piper X URDF README", "确认 Piper X 是独立模型资产；本场景直接使用对应 URDF/网格"],
            ["AgileX 官方 Piper 固件参考", "不同固件对 Piper X J4/J5 符号约定存在差异，真机部署前必须核对"],
            ["MoveIt 官方 pick_ik 文档", "全局/局部优化器分工与显式位置/姿态阈值，支持本次全局锚定 + 局部 warm-start 结构"],
            ["TRAC-IK, Humanoids 2015, DOI 10.1109/HUMANOIDS.2015.7363472", "作为关节限位附近多策略 IK 的方法参考；本实现仍以 MuJoCo FK 为最终门"],
        ], [66*mm, 114*mm]),
        p(
            "链接：<br/>"
            "https://github.com/agilexrobotics/agx_arm_urdf/blob/main/README_EN.md<br/>"
            "https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/piper/firmware_reference.md<br/>"
            "https://moveit.picknik.ai/main/doc/how_to_guides/pick_ik/pick_ik_tutorial.html",
            "small",
        ),
        p("可追溯性", "h2"),
        p(
            f"源轨迹 SHA-256：{summary['source']['sha256']}。"
            "所有验收数字来自最终 JSON/NPZ 和真实 MuJoCo 模型；碰撞类型由最终 scene.xml 上的碰撞体重新计算。",
            "small",
        ),
    ])

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path


def main(argv=None):
    options = parse_args(argv)
    output = build_report(
        options.summary, options.trajectory, options.scene,
        options.video, options.output)
    print(output)


if __name__ == "__main__":
    main()
