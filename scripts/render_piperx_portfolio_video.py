"""Render original tool targets and executed PiperX TCP paths in MuJoCo.

The complete controller-event timeline is retained.  This script changes only
the camera, lighting, trace styling, and editorial overlay of a MuJoCo render.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np

from factory_bimanual.multitask_fixed_time_study import discover_dual_hand_trajectories
from factory_bimanual.video import build_realtime_timing, decode_check_mp4, interpolation_sample
from scripts.build_piperx_controller_event_manifest import validate_archive_summary
from scripts.run_piperx_controller_event_v4 import ROOT


WIDTH = 1280
HEIGHT = 720
FPS = 30
SCENE_TOP = 62
SCENE_HEIGHT = 592
RENDER_PROTOCOL = "piperx-v4-portfolio-target-vs-actual-v4"
TARGET_HISTORY_S = 1.0
TARGET_FUTURE_S = 0.4
ACTUAL_HISTORY_S = 1.5


def _repo_artifact(relative_path: str) -> Path:
    path = (ROOT / relative_path.replace("\\", "/")).resolve()
    path.relative_to(ROOT.resolve())
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_validated_shard(summary_path: Path):
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mode") != "baseline":
        raise ValueError("portfolio view requires a complete baseline shard")
    if summary.get("controller_event_rows") != summary.get("controller_event_rows_total"):
        raise ValueError("portfolio view requires the complete recording")
    artifacts = summary["artifacts"]
    trajectory_path = _repo_artifact(artifacts["trajectory_npz"]["path"])
    scene_path = _repo_artifact(artifacts["scene_xml"]["path"])
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    specs = {spec.key: spec for spec in discover_dual_hand_trajectories(ROOT / "data/factory")}
    if summary["trajectory"] not in specs:
        raise ValueError("unknown source trajectory")
    validate_archive_summary(summary, arrays, spec=specs[summary["trajectory"]])
    return summary, arrays, scene_path, trajectory_path


def _extract_tracks(arrays: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return (raw target position, executed TCP position) for each hand."""
    count = len(arrays["source_time_s"])
    tracks = {}
    for side in ("left", "right"):
        target = np.asarray(arrays[f"{side}_target_position_m"], dtype=float)
        actual_pose = np.asarray(arrays[f"{side}_actual_tcp"], dtype=float)
        if target.shape != (count, 3) or actual_pose.shape != (count, 7):
            raise ValueError(f"{side} target/actual TCP shapes do not match event count")
        if not np.isfinite(target).all() or not np.isfinite(actual_pose).all():
            raise ValueError(f"{side} target/actual TCP must be finite")
        tracks[side] = target, actual_pose[:, :3]
    return tracks


def _trace_segments(points: np.ndarray, times: np.ndarray, time_s: float, *,
                    full_path: bool, dashed: bool,
                    current_point: np.ndarray | None = None,
                    history_s: float | None = None,
                    future_s: float = 0.0) -> np.ndarray:
    """Select source-bound path geometry without drawing future executed motion."""
    relative_times = np.asarray(times, dtype=float) - float(times[0])
    if (history_s is not None and history_s < 0) or future_s < 0:
        raise ValueError("trace windows must be nonnegative")
    start = 0 if full_path or history_s is None else int(np.searchsorted(
        relative_times, time_s - history_s, side="left"))
    end = len(relative_times) if full_path else int(np.searchsorted(
        relative_times, time_s + future_s, side="right"))
    if end <= start:
        return np.empty((0, 2, 3), dtype=float)
    path = np.asarray(points, dtype=float)[start:end]
    if (not full_path and current_point is not None and future_s == 0
            and time_s > relative_times[end - 1] + 1e-9):
        path = np.vstack((path, np.asarray(current_point, dtype=float)))
    if len(path) < 2:
        return np.empty((0, 2, 3), dtype=float)
    segments = np.stack((path[:-1], path[1:]), axis=1)
    if dashed:
        segments = segments[np.arange(len(segments)) % 3 != 2]
    return segments


def _trajectory_layers(points: np.ndarray, times: np.ndarray, time_s: float, *,
                       history_s: float, future_s: float = 0.0,
                       current_point: np.ndarray | None = None,
                       dashed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return the complete archived path and its local playback emphasis."""
    complete = _trace_segments(points, times, time_s,
                               full_path=True, dashed=dashed)
    local = _trace_segments(points, times, time_s,
                            full_path=False, dashed=dashed,
                            history_s=history_s, future_s=future_s,
                            current_point=current_point)
    return complete, local


def _camera(targets: np.ndarray, model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    low = targets.min(axis=0)
    high = targets.max(axis=0)
    target_center = (low + high) / 2
    base_center = (model.body("left_base_mount").pos
                   + model.body("right_base_mount").pos) / 2
    camera.lookat[:] = 0.5 * (target_center + base_center)
    camera.distance = 1.40
    camera.azimuth = 225
    camera.elevation = -22
    return camera


def _style_model(model: mujoco.MjModel) -> None:
    model.vis.headlight.ambient[:] = (0.40, 0.43, 0.46)
    model.vis.headlight.diffuse[:] = (0.65, 0.68, 0.71)
    model.vis.headlight.specular[:] = (0.08, 0.08, 0.08)
    for geom_index in range(model.ngeom):
        name = model.geom(geom_index).name
        if name == "workbench":
            model.geom_rgba[geom_index] = (0.34, 0.44, 0.49, 1)
        elif name == "left_target_marker":
            model.geom_rgba[geom_index] = (0.12, 0.54, 0.95, 0.9)
        elif name == "right_target_marker":
            model.geom_rgba[geom_index] = (0.95, 0.31, 0.67, 0.9)
        elif name.startswith("left_"):
            model.geom_rgba[geom_index] = (0.58, 0.71, 0.78, 1)
        elif name.startswith("right_"):
            model.geom_rgba[geom_index] = (0.75, 0.69, 0.59, 1)


def _draw_path(scene, segments: np.ndarray,
               color: tuple[float, float, float, float], radius: float) -> None:
    connector = getattr(mujoco, "mjv_connector", None) or mujoco.mjv_makeConnector
    for first, second in segments:
        if np.linalg.norm(second - first) < 1e-7:
            continue
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo trace geometry capacity exceeded")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(color, np.float32))
        connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, first, second)
        scene.ngeom += 1


def _draw_tcp_marker(scene, point: np.ndarray,
                     color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo TCP marker capacity exceeded")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, 0.012), point, np.eye(3).ravel(),
                        np.asarray(color, np.float32))
    scene.ngeom += 1


def _draw_comparison_legend(canvas: np.ndarray) -> None:
    cv2.rectangle(canvas, (22, SCENE_TOP + 14), (470, SCENE_TOP + 92),
                  (40, 45, 49), -1)
    for y, label, original, actual in (
            (SCENE_TOP + 44, "LEFT", (255, 217, 107), (235, 119, 41)),
            (SCENE_TOP + 77, "RIGHT", (178, 107, 249), (53, 171, 255))):
        cv2.putText(canvas, label, (36, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (230, 235, 238), 1, cv2.LINE_AA)
        for x in (106, 122, 138):
            cv2.line(canvas, (x, y), (x + 9, y), original, 3, cv2.LINE_AA)
        cv2.putText(canvas, "ORIGINAL", (163, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (205, 215, 222), 1, cv2.LINE_AA)
        cv2.line(canvas, (284, y), (317, y), actual, 4, cv2.LINE_AA)
        cv2.putText(canvas, "ACTUAL TCP", (327, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (205, 215, 222), 1, cv2.LINE_AA)


def _draw_xy_path_overview(canvas: np.ndarray, *, tracks: dict,
                           times: np.ndarray, time_s: float,
                           target_now: dict, actual_now: dict) -> None:
    """Show both complete XY paths and emphasize the elapsed actual segment."""
    cv2.rectangle(canvas, (22, SCENE_TOP + 112), (470, SCENE_TOP + 306),
                  (40, 45, 49), -1)
    cv2.putText(canvas, "COMPLETE XY PATHS  /  TARGET vs ACTUAL", (36, SCENE_TOP + 137),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (205, 215, 222), 1, cv2.LINE_AA)
    colors = {
        "left": ((255, 217, 107), (235, 119, 41)),
        "right": ((178, 107, 249), (53, 171, 255)),
    }
    for side, left_x in (("left", 36), ("right", 252)):
        target, actual = tracks[side]
        target_color, actual_color = colors[side]
        x0, x1 = left_x, left_x + 202
        y0, y1 = SCENE_TOP + 166, SCENE_TOP + 290
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (61, 68, 73), 1)
        cv2.putText(canvas, side.upper(), (x0 + 7, y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (221, 229, 233), 1, cv2.LINE_AA)
        bounds = np.vstack((target[:, :2], actual[:, :2]))
        low = bounds.min(axis=0)
        high = bounds.max(axis=0)
        extent = np.maximum(high - low, 1e-9)
        scale = min((x1 - x0 - 22) / extent[0], (y1 - y0 - 22) / extent[1])
        center = (low + high) / 2

        def project(point: np.ndarray) -> tuple[int, int]:
            return (round((x0 + x1) / 2 + (point[0] - center[0]) * scale),
                    round((y0 + y1) / 2 - (point[1] - center[1]) * scale))

        for segment in _trace_segments(target, times, time_s,
                                       full_path=True, dashed=True):
            cv2.line(canvas, project(segment[0]), project(segment[1]),
                     target_color, 1, cv2.LINE_AA)
        complete_actual, elapsed_actual = _trajectory_layers(
            actual, times, time_s, history_s=time_s,
            current_point=actual_now[side])
        subdued_color = tuple(round(0.60 * channel + 0.40 * background)
                              for channel, background in zip(actual_color, (40, 45, 49)))
        for segment in complete_actual:
            cv2.line(canvas, project(segment[0]), project(segment[1]),
                     subdued_color, 1, cv2.LINE_AA)
        for segment in elapsed_actual:
            cv2.line(canvas, project(segment[0]), project(segment[1]),
                     actual_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, project(target_now[side]), 4, target_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, project(actual_now[side]), 3, actual_color, -1, cv2.LINE_AA)


def _blend_background(rgb: np.ndarray) -> np.ndarray:
    """Replace only the unlit scene background with a muted studio gradient."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    top = np.array((46, 40, 34), dtype=np.float32)
    bottom = np.array((74, 68, 58), dtype=np.float32)
    gradient = np.linspace(top, bottom, h, dtype=np.float32)[:, None, :]
    gradient = np.broadcast_to(gradient, (h, w, 3))
    brightness = bgr.max(axis=2).astype(np.float32)
    opacity = np.clip((brightness - 3) / 20, 0, 1)[:, :, None]
    return np.asarray(bgr * opacity + gradient * (1 - opacity), dtype=np.uint8)


def _draw_editorial_frame(scene_bgr: np.ndarray, *, time_s: float,
                          duration_s: float, following: bool,
                          coverage: float,
                          position_error_mm: dict[str, float],
                          tracks: dict, times: np.ndarray,
                          target_now: dict, actual_now: dict) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (32, 35, 36), dtype=np.uint8)
    canvas[SCENE_TOP:SCENE_TOP + SCENE_HEIGHT] = scene_bgr
    _draw_comparison_legend(canvas)
    _draw_xy_path_overview(canvas, tracks=tracks, times=times, time_s=time_s,
                           target_now=target_now, actual_now=actual_now)
    cv2.putText(canvas, "FOLD BOX  /  PIPERX DUAL ARM", (34, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.73, (241, 243, 244), 2, cv2.LINE_AA)
    cv2.putText(canvas, "MUJOCO KINEMATIC REPLAY  |  FIXED SOURCE TIME", (730, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (189, 202, 209), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, HEIGHT - 66), (WIDTH, HEIGHT), (32, 35, 36), -1)
    dot_color = (92, 211, 129) if following else (76, 176, 239)
    cv2.circle(canvas, (43, HEIGHT - 35), 7, dot_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "FOLLOWING" if following else "SAFE HOLD", (61, HEIGHT - 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (241, 243, 244), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"L {position_error_mm['left']:.1f} mm", (244, HEIGHT - 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 119, 41), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"R {position_error_mm['right']:.1f} mm", (440, HEIGHT - 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (53, 171, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{coverage * 100:.1f}% pose coverage", (671, HEIGHT - 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (188, 202, 210), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{time_s:05.2f} / {duration_s:05.2f} s", (1060, HEIGHT - 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 243, 244), 1, cv2.LINE_AA)
    bar_x0, bar_x1, bar_y = 34, WIDTH - 34, HEIGHT - 8
    cv2.line(canvas, (bar_x0, bar_y), (bar_x1, bar_y), (81, 91, 98), 3, cv2.LINE_AA)
    progress = 1.0 if duration_s == 0 else np.clip(time_s / duration_s, 0, 1)
    cv2.line(canvas, (bar_x0, bar_y), (round(bar_x0 + progress * (bar_x1 - bar_x0)), bar_y),
             (217, 184, 121), 3, cv2.LINE_AA)
    return canvas


def render(summary_path: Path, output_path: Path, poster_path: Path) -> dict:
    summary, arrays, scene_path, trajectory_path = _load_validated_shard(summary_path)
    times = np.asarray(arrays["source_time_s"], dtype=float)
    timing = build_realtime_timing(times, FPS)
    qpos = np.asarray(arrays["qpos"], dtype=float)
    tracks = _extract_tracks(arrays)
    left, right = tracks["left"][0], tracks["right"][0]
    accepted = np.asarray(arrays["both_accept"], dtype=bool)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    _style_model(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, SCENE_HEIGHT)
    data = mujoco.MjData(model)
    camera = _camera(np.vstack((left, right, tracks["left"][1],
                                tracks["right"][1])), model)
    left_mocap = int(model.body_mocapid[model.body("left_target").id])
    right_mocap = int(model.body_mocapid[model.body("right_target").id])
    tcp_sites = {side: model.site(f"{side}_tcp").id for side in tracks}
    output_path = Path(output_path)
    poster_path = Path(poster_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    poster_frame_index = round(timing.source_duration_s * 0.38 * FPS)
    try:
        with mujoco.Renderer(model, width=WIDTH, height=SCENE_HEIGHT,
                             max_geom=4096) as renderer:
            for frame_index, encoded_time in enumerate(timing.encoded_time_s):
                lower, upper, alpha = interpolation_sample(times, float(encoded_time))
                data.qpos[:] = (1 - alpha) * qpos[lower] + alpha * qpos[upper]
                data.mocap_pos[left_mocap] = (1 - alpha) * left[lower] + alpha * left[upper]
                data.mocap_pos[right_mocap] = (1 - alpha) * right[lower] + alpha * right[upper]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                actual_now = {side: data.site_xpos[site_id].copy()
                              for side, site_id in tcp_sites.items()}
                target_now = {"left": data.mocap_pos[left_mocap].copy(),
                              "right": data.mocap_pos[right_mocap].copy()}
                for side, color in (("left", (0.42, 0.85, 1.0, 0.66)),
                                    ("right", (0.98, 0.42, 0.70, 0.66))):
                    original, _ = tracks[side]
                    complete, local = _trajectory_layers(
                        original, times, float(encoded_time), dashed=True,
                        history_s=TARGET_HISTORY_S,
                        future_s=TARGET_FUTURE_S)
                    _draw_path(renderer.scene, complete,
                               (*color[:3], 0.16), 0.0014)
                    _draw_path(renderer.scene, local, color, 0.003)
                for side, color in (("left", (0.16, 0.47, 0.92, 1.0)),
                                    ("right", (1.0, 0.67, 0.21, 1.0))):
                    _, actual = tracks[side]
                    complete, local = _trajectory_layers(
                        actual, times, float(encoded_time),
                        history_s=ACTUAL_HISTORY_S,
                        current_point=actual_now[side])
                    _draw_path(renderer.scene, complete,
                               (*color[:3], 0.18), 0.0017)
                    _draw_path(renderer.scene, local, color, 0.006)
                    _draw_tcp_marker(renderer.scene, actual_now[side], color)
                scene_bgr = _blend_background(renderer.render())
                # An interpolated frame inherits the incoming edge's acceptance.
                audit_index = upper if alpha > 1e-12 else lower
                frame = _draw_editorial_frame(
                    scene_bgr, time_s=float(encoded_time),
                    duration_s=timing.source_duration_s,
                    following=bool(accepted[audit_index]),
                    coverage=float(summary["both_accept_coverage"]),
                    position_error_mm={side: float(np.linalg.norm(
                        actual_now[side] - target_now[side]) * 1000)
                        for side in tracks},
                    tracks=tracks, times=times,
                    target_now=target_now, actual_now=actual_now)
                if frame_index == poster_frame_index:
                    if not cv2.imwrite(str(poster_path), frame):
                        raise RuntimeError("could not write portfolio poster")
                process.stdin.write(frame.tobytes())
                if frame_index % 150 == 0:
                    print(f"rendered {frame_index + 1}/{len(timing.encoded_time_s)}", flush=True)
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"H.264 encode failed: {stderr}")
    except Exception:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    check = decode_check_mp4(output_path, expected_frames=len(timing.encoded_time_s),
                             expected_resolution=(WIDTH, HEIGHT),
                             expected_duration_s=timing.source_duration_s)
    provenance = {
        "schema": RENDER_PROTOCOL,
        "trajectory": summary["trajectory"],
        "mode": summary["mode"],
        "source_summary": Path(summary_path).resolve().relative_to(ROOT.resolve()).as_posix(),
        "source_summary_sha256": _sha256(Path(summary_path)),
        "trajectory_sha256": _sha256(trajectory_path),
        "scene_sha256": _sha256(scene_path),
        "video_sha256": _sha256(output_path),
        "poster_sha256": _sha256(poster_path),
        "controller_event_count": len(times),
        "source_duration_s": timing.source_duration_s,
        "encoded_duration_s": check.duration_s,
        "frame_count": check.frame_count,
        "fps": check.fps,
        "strict_paired_coverage": float(summary["both_accept_coverage"]),
        "collision_frames": int(summary["collision_frames"]),
        "edge_collision_frames": int(summary["edge_collision_frames"]),
        "topology_invalid_frames": int(summary["topology_invalid_frames"]),
        "retiming_applied": False,
        "playback_speed": 1.0,
        "dynamics_enforced": bool(summary["dynamics_enforced"]),
        "original_track_arrays": {side: f"{side}_target_position_m"
                                  for side in tracks},
        "actual_track_arrays": {side: f"{side}_actual_tcp[:,:3]"
                                for side in tracks},
        "actual_cursor": "MuJoCo forward-kinematics TCP of interpolated saved qpos",
        "target_trail_window_s": [TARGET_HISTORY_S, TARGET_FUTURE_S],
        "actual_trail_history_s": ACTUAL_HISTORY_S,
        "xy_path_overview": "complete XY target and actual TCP; elapsed actual path highlighted",
        "complete_3d_paths": "full archived target and actual TCP traces shown faintly in MuJoCo",
        "render_mode": "MuJoCo kinematic replay of saved joint states; no forward dynamics",
    }
    output_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render(args.summary, args.output, args.poster), indent=2))


if __name__ == "__main__":
    main()
