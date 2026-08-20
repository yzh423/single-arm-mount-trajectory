"""Original-time video scheduling and auditable frame provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from .artifacts import FrameDiagnostics


@dataclass(frozen=True)
class RealtimeVideoTiming:
    fps: float
    source_duration_s: float
    encoded_duration_s: float
    encoded_time_s: np.ndarray
    frame_source_indices: np.ndarray


@dataclass(frozen=True)
class VideoRenderConfig:
    width: int = 640
    height: int = 480
    fps: float = 60.0
    camera: str | int | None = None
    trajectory_radius_m: float = 0.007
    interpolate_states: bool = False
    camera_azimuth_deg: float = 315.0
    camera_elevation_deg: float = -27.0
    camera_distance_scale: float = 1.35
    title: str = "Dual xArm6 trajectory follow"


@dataclass(frozen=True)
class _InterpolationAudit:
    execution_index: int
    scope: str
    collision: bool
    execution_state: str


def interpolation_sample(source_time_s: np.ndarray, encoded_time_s: float):
    """Return the bracketing source rows and interpolation fraction."""
    relative = np.asarray(source_time_s, float) - float(source_time_s[0])
    upper = int(np.searchsorted(relative, encoded_time_s, side="right"))
    upper = min(max(upper, 1), len(relative) - 1)
    lower = upper - 1
    span = relative[upper] - relative[lower]
    alpha = float(np.clip((encoded_time_s-relative[lower]) / span, 0.0, 1.0))
    if encoded_time_s >= relative[-1]:
        return len(relative)-1, len(relative)-1, 0.0
    return lower, upper, alpha


def _interpolation_audit_sample(
        diagnostics: Sequence[FrameDiagnostics], lower: int, upper: int,
        alpha: float) -> _InterpolationAudit:
    """Select collision evidence for a knot or its active incoming edge."""
    moving = upper != lower and alpha > 1e-12
    index = int(upper if moving else lower)
    item = diagnostics[index]
    state_collision = (
        bool(item.collision) if item.state_collision is None
        else bool(item.state_collision)
    )
    incoming_collision = (
        bool(item.collision) if item.incoming_transition_collision is None
        else bool(item.incoming_transition_collision)
    )
    collision = incoming_collision if moving else state_collision
    raw_state = str(item.execution_state or item.left_failure_reason)
    base_state = raw_state.removesuffix("_COLLISION")
    execution_state = f"{base_state}_COLLISION" if collision else base_state
    return _InterpolationAudit(
        execution_index=index,
        scope="incoming_edge" if moving else "knot",
        collision=collision,
        execution_state=execution_state,
    )


def is_replanned_transition(q0, q1, interval_s: float, fps: float) -> bool:
    """Identify visibly retimed motion, excluding duplicate/slow source rows."""
    motion = float(np.linalg.norm(np.asarray(q1, float) - np.asarray(q0, float)))
    return interval_s > 2.0 / fps and motion > np.deg2rad(2.0)


def execution_status_label(
        execution_state: str, *, ok: bool,
        retimed_transition: bool = False, recovering: bool = False):
    """Return an honest overlay label for combined execution conditions."""
    state = str(execution_state)
    collision = state.endswith("_COLLISION")
    base = state.removesuffix("_COLLISION")
    if collision and base == "RETIMED_TRANSITION":
        return "RETIMED TRANSITION - COLLISION AUDIT", (35, 35, 235)
    if collision and base == "FOLLOW_RETIMED":
        return "RETIMED SOURCE POSE - COLLISION AUDIT", (35, 35, 235)
    if collision:
        return "TRACKING - COLLISION AUDIT", (35, 35, 235)
    if base == "RETIMED_TRANSITION":
        return "RETIMED TRANSITION - TCP PATH DEVIATES", (0, 165, 255)
    if base == "FOLLOW_RETIMED":
        return "TRACKING - RETIMED SOURCE POSE", (0, 145, 235)
    if retimed_transition:
        return (
            "REPLANNED RECOVERY - TCP PATH DEVIATES" if recovering else
            "REPLANNED BRANCH CHANGE - TCP PATH DEVIATES",
            (0, 165, 255),
        )
    if not ok:
        return f"CANNOT FOLLOW - HOLD: {execution_state}", (35, 35, 235)
    return "TRACKING", (35, 175, 75)


@dataclass(frozen=True)
class VideoDecodeCheck:
    frame_count: int
    fps: float
    duration_s: float
    width: int
    height: int


@dataclass(frozen=True)
class VideoRenderResult:
    video_path: Path
    provenance_path: Path
    check: VideoDecodeCheck


def _path_render_indices(frame_count: int, maximum_points: int = 450) -> np.ndarray:
    if frame_count < 2:
        return np.arange(frame_count, dtype=int)
    return np.unique(np.linspace(0, frame_count - 1, min(frame_count, maximum_points))
                     .round().astype(int))


def _sphere(scene, position, radius, color) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.asarray([radius, 0.0, 0.0]), np.asarray(position, float),
                        np.eye(3).reshape(-1), np.asarray(color, np.float32))
    scene.ngeom += 1


def _connector(scene, start, end, radius, color) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.asarray(color, np.float32))
    function = getattr(mujoco, "mjv_connector", None) or mujoco.mjv_makeConnector
    function(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
             np.asarray(start, float), np.asarray(end, float))
    scene.ngeom += 1


def build_realtime_timing(source_time_s: np.ndarray, fps: float = 60.0) -> RealtimeVideoTiming:
    times = np.asarray(source_time_s, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("source timestamps must be a non-empty finite 1-D array")
    if np.any(np.diff(times) <= 0):
        raise ValueError("source timestamps must be strictly increasing")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    relative = times - times[0]
    duration = float(relative[-1])
    # Include both temporal endpoints. Playback spans (N-1) frame intervals;
    # selecting N this way preserves source duration to at most half an interval.
    frame_count = max(1, int(round(duration * fps)) + 1)
    encoded_duration = (frame_count - 1) / fps
    encoded_times = np.minimum(np.arange(frame_count, dtype=float) / fps, duration)
    # Quantizing a non-integral duration can otherwise leave the final source
    # pose just beyond the last encoded sample when round() chooses the lower
    # frame count.  The container still plays at constant FPS; the provenance
    # timestamp on the last frame records that it represents the exact endpoint.
    encoded_times[-1] = duration
    # Causal floor/zero-order hold: an encoded frame never shows a future source
    # state. All source rows additionally remain in the provenance sidecar.
    indices = np.searchsorted(relative, encoded_times + 1e-12, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 1)
    return RealtimeVideoTiming(float(fps), duration, float(encoded_duration), encoded_times, indices.astype(int))


def write_video_provenance(
    path: Path, timing: RealtimeVideoTiming,
    diagnostics: Sequence[FrameDiagnostics], *,
    timeline_domain: str = "source", interpolate_states: bool = False,
) -> Path:
    if len(diagnostics) == 0:
        raise ValueError("diagnostics may not be empty")
    if timeline_domain not in {"source", "execution"}:
        raise ValueError("timeline_domain must be source or execution")
    timeline_times = np.asarray([
        (item.execution_time_s if timeline_domain == "execution"
         else item.source_time_s)
        for item in diagnostics
    ], dtype=float)
    if np.any(~np.isfinite(timeline_times)):
        raise ValueError(
            f"diagnostics are missing finite {timeline_domain} timestamps")
    rows = []
    for encoded_index, (encoded_t, source_index) in enumerate(
        zip(timing.encoded_time_s, timing.frame_source_indices)
    ):
        if source_index >= len(diagnostics):
            raise ValueError("timing references missing diagnostics")
        lower = upper = int(source_index)
        alpha = 0.0
        if interpolate_states:
            lower, upper, alpha = interpolation_sample(
                timeline_times,
                float(encoded_t),
            )
        audit = _interpolation_audit_sample(
            diagnostics, lower, upper, alpha)
        row = asdict(diagnostics[audit.execution_index])
        row.update(
            collision=audit.collision,
            execution_state=audit.execution_state,
            audit_execution_index=audit.execution_index,
            audit_scope=audit.scope,
        )
        if timeline_domain == "execution":
            row.update(
                left_failure_reason=audit.execution_state,
                right_failure_reason=audit.execution_state,
            )
        row.update(encoded_frame_index=encoded_index, encoded_time_s=float(encoded_t))
        if interpolate_states:
            row.update(
                interpolation_lower_index=int(lower),
                interpolation_upper_index=int(upper),
                interpolation_alpha=float(alpha),
            )
        rows.append(row)
    timing_method = (
        "constant-fps linear interpolation over authoritative timestamps"
        if interpolate_states else
        "constant-fps zero-order hold over authoritative timestamps"
    )
    payload = {
        "schema_version": 2,
        "playback_speed": 1.0,
        "timeline_domain": timeline_domain,
        "timing_method": timing_method,
        "fps": timing.fps,
        f"{timeline_domain}_duration_s": timing.source_duration_s,
        "encoded_duration_s": timing.encoded_duration_s,
        ("source_frames" if timeline_domain == "source"
         else "execution_knots"): [asdict(row) for row in diagnostics],
        "encoded_frames": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def decode_check_mp4(path: Path, *, expected_frames: int | None = None,
                     expected_resolution: tuple[int, int] | None = None,
                     expected_duration_s: float | None = None) -> VideoDecodeCheck:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("MP4 decode preflight failed: OpenCV is unavailable") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"MP4 decode preflight failed: cannot open {path}")
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if frames <= 0 or fps <= 0:
        raise RuntimeError("MP4 decode check returned invalid frame count or fps")
    # Presentation duration is the interval from first to last source-time frame.
    duration = max(0.0, (frames - 1) / fps)
    if expected_frames is not None and frames != expected_frames:
        raise RuntimeError(f"MP4 frame mismatch: decoded {frames}, expected {expected_frames}")
    if expected_resolution is not None and (width, height) != expected_resolution:
        raise RuntimeError(f"MP4 resolution mismatch: decoded {(width, height)}, expected {expected_resolution}")
    if expected_duration_s is not None and abs(duration - expected_duration_s) > 1.0 / fps + 1e-9:
        raise RuntimeError(f"MP4 duration mismatch: decoded {duration}, expected {expected_duration_s}")
    return VideoDecodeCheck(frames, fps, duration, width, height)


def render_mujoco_mp4(scene_xml: Path, output_path: Path, source_time_s: np.ndarray, *,
                      qpos: np.ndarray, diagnostics: Sequence[FrameDiagnostics],
                      left_targets: np.ndarray | None = None,
                      right_targets: np.ndarray | None = None,
                      follow_success: np.ndarray | None = None,
                      failure_reasons: Sequence[str] | None = None,
                      timeline_domain: str = "source",
                      config: VideoRenderConfig = VideoRenderConfig()) -> VideoRenderResult:
    """Render q states and optional left/right mocap targets on source-time schedule."""
    try:
        import cv2
        import mujoco
    except ImportError as exc:
        raise RuntimeError(f"renderer preflight failed: {exc}") from exc
    if config.width < 2 or config.height < 2:
        raise ValueError("video dimensions must be at least 2 pixels")
    timing = build_realtime_timing(source_time_s, config.fps)
    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or qpos.shape[0] != len(source_time_s):
        raise ValueError("qpos must have one row per source timestamp")
    if len(diagnostics) != len(source_time_s):
        raise ValueError("diagnostics must have one row per source timestamp")
    if follow_success is not None and np.asarray(follow_success).shape != (len(source_time_s),):
        raise ValueError("follow_success must have one value per source timestamp")
    if failure_reasons is not None and len(failure_reasons) != len(source_time_s):
        raise ValueError("failure_reasons must have one value per source timestamp")
    targets = [left_targets, right_targets]
    for target in targets:
        if target is not None and np.asarray(target).shape != (len(source_time_s), 3):
            raise ValueError("target positions must have shape (source_frames, 3)")
    try:
        model = mujoco.MjModel.from_xml_path(str(Path(scene_xml).resolve()))
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=config.height, width=config.width, max_geom=2048)
    except Exception as exc:
        raise RuntimeError(f"renderer preflight failed: {type(exc).__name__}: {exc}") from exc
    if qpos.shape[1] != model.nq:
        renderer.close()
        raise ValueError(f"qpos has {qpos.shape[1]} columns but scene requires nq={model.nq}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             config.fps, (config.width, config.height))
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError("renderer preflight failed: OpenCV MP4 encoder is unavailable")
    render_indices = [_path_render_indices(len(source_time_s)) for _ in targets]
    traversed_colors = ([0.10, 0.42, 0.90, 0.95], [0.82, 0.28, 0.78, 0.95])
    render_camera = -1 if config.camera is None else config.camera
    if config.camera is None and any(target is not None for target in targets):
        cloud = np.vstack([np.asarray(target) for target in targets if target is not None])
        low, high = cloud.min(axis=0), cloud.max(axis=0)
        render_camera = mujoco.MjvCamera()
        render_camera.lookat[:] = 0.5 * (low + high)
        render_camera.lookat[2] -= 0.08
        render_camera.distance = max(1.05, min(1.85, config.camera_distance_scale * np.linalg.norm(high - low)))
        render_camera.azimuth = config.camera_azimuth_deg
        render_camera.elevation = config.camera_elevation_deg
    try:
        for encoded_t, source_index in zip(timing.encoded_time_s, timing.frame_source_indices):
            index = int(source_index)
            lower, upper, alpha = interpolation_sample(source_time_s, float(encoded_t))
            if config.interpolate_states:
                data.qpos[:] = (1.0-alpha)*qpos[lower] + alpha*qpos[upper]
            else:
                data.qpos[:] = qpos[index]
            for mocap_index, target in enumerate(targets):
                if target is not None:
                    if mocap_index >= model.nmocap:
                        raise ValueError("scene has fewer mocap bodies than supplied targets")
                    points = np.asarray(target)
                    data.mocap_pos[mocap_index] = ((1.0-alpha)*points[lower] + alpha*points[upper]
                                                   if config.interpolate_states else points[index])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=render_camera)
            for side_index, target in enumerate(targets):
                if target is None:
                    continue
                points = np.asarray(target)
                indices = render_indices[side_index]
                for first, second in zip(indices[:-1], indices[1:]):
                    color = (traversed_colors[side_index] if second <= index
                             else [0.52, 0.54, 0.56, 0.38])
                    _connector(renderer.scene, points[first], points[second],
                               config.trajectory_radius_m, color)
                _sphere(renderer.scene, points[index], 0.016, [0.86, 0.08, 0.08, 0.95])
                site_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_SITE,
                    "left_tcp" if side_index == 0 else "right_tcp")
                if site_id >= 0:
                    _sphere(renderer.scene, data.site_xpos[site_id], 0.012,
                            [0.02, 0.78, 0.38, 1.0])
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            title_size, _ = cv2.getTextSize(
                config.title, cv2.FONT_HERSHEY_SIMPLEX, .72, 2)
            panel_right = min(
                config.width - 18, max(520, 34 + title_size[0] + 16))
            cv2.rectangle(
                bgr, (18, 16), (panel_right, 112), (245, 245, 245), -1)
            cv2.putText(bgr, config.title, (34, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, .72, (25, 25, 25), 2, cv2.LINE_AA)
            legend = (((255, 110, 30), "Left traversed"),
                      ((200, 70, 210), "Right traversed"),
                      ((140, 140, 140), "Future path"))
            for item, (color, label) in enumerate(legend):
                x = 34 + item * 160
                cv2.line(bgr, (x, 76), (x + 35, 76), color, 5, cv2.LINE_AA)
                cv2.putText(bgr, label, (x, 101), cv2.FONT_HERSHEY_SIMPLEX,
                            .43, (35, 35, 35), 1, cv2.LINE_AA)
            if follow_success is not None:
                audit = _interpolation_audit_sample(
                    diagnostics, lower, upper,
                    alpha if config.interpolate_states else 0.0)
                audit_index = audit.execution_index
                ok = bool(np.asarray(follow_success)[audit_index])
                execution_state = audit.execution_state
                base_execution_state = execution_state.removesuffix(
                    "_COLLISION")
                explicit_execution_state = base_execution_state in {
                    "FOLLOW", "FOLLOW_RETIMED", "RETIMED_TRANSITION"}
                retimed_transition = (not explicit_execution_state and
                    config.interpolate_states and upper != lower and
                    is_replanned_transition(qpos[lower], qpos[upper],
                                            source_time_s[upper]-source_time_s[lower],
                                            config.fps))
                recovering = bool(
                    failure_reasons is not None
                    and (failure_reasons[lower] != "ok"
                         or failure_reasons[upper] != "ok"))
                reason, color = execution_status_label(
                    execution_state, ok=ok,
                    retimed_transition=retimed_transition,
                    recovering=recovering)
                cv2.rectangle(bgr, (18, config.height - 70),
                              (min(config.width - 18, 900), config.height - 18), (245, 245, 245), -1)
                cv2.putText(bgr, f"t={encoded_t:.2f}s  {reason}",
                            (34, config.height - 35), cv2.FONT_HERSHEY_SIMPLEX,
                            .72, color, 2, cv2.LINE_AA)
            writer.write(bgr)
    finally:
        writer.release()
        renderer.close()
    check = decode_check_mp4(output_path, expected_frames=len(timing.frame_source_indices),
                             expected_resolution=(config.width, config.height),
                             expected_duration_s=timing.encoded_duration_s)
    provenance = write_video_provenance(
        output_path.with_suffix(".provenance.json"), timing, diagnostics,
        timeline_domain=timeline_domain,
        interpolate_states=config.interpolate_states)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["decode_check"] = asdict(check)
    payload["scene_xml"] = str(Path(scene_xml).resolve())
    provenance.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return VideoRenderResult(output_path, provenance, check)
