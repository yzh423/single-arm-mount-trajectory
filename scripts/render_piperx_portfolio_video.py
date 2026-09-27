"""Render a legible portfolio cut from a source-bound PiperX v4 shard.

The complete controller-event timeline is retained.  This script changes only
the camera, lighting, trail length, and editorial overlay of a MuJoCo render.
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
TRAIL_SECONDS = 1.7
RENDER_PROTOCOL = "piperx-v4-portfolio-full-timeline-v1"


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


def _draw_trail(scene, points: np.ndarray, times: np.ndarray, time_s: float,
                color: tuple[float, float, float, float]) -> None:
    end = int(np.searchsorted(times, time_s, side="right"))
    start = int(np.searchsorted(times, time_s - TRAIL_SECONDS, side="left"))
    if end - start < 2:
        return
    selected = np.unique(np.linspace(start, end - 1, min(end - start, 65)).round().astype(int))
    connector = getattr(mujoco, "mjv_connector", None) or mujoco.mjv_makeConnector
    for first, second in zip(selected[:-1], selected[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(color, np.float32))
        connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004,
                  points[first], points[second])
        scene.ngeom += 1


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
                          coverage: float) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (32, 35, 36), dtype=np.uint8)
    canvas[SCENE_TOP:SCENE_TOP + SCENE_HEIGHT] = scene_bgr
    cv2.putText(canvas, "FOLD BOX  /  PIPERX DUAL ARM", (34, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.73, (241, 243, 244), 2, cv2.LINE_AA)
    cv2.putText(canvas, "FULL RECORDING  |  FIXED CONTROLLER TIME", (810, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (189, 202, 209), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, HEIGHT - 66), (WIDTH, HEIGHT), (32, 35, 36), -1)
    dot_color = (92, 211, 129) if following else (76, 176, 239)
    cv2.circle(canvas, (43, HEIGHT - 35), 7, dot_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "FOLLOWING" if following else "SAFE HOLD", (61, HEIGHT - 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (241, 243, 244), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{coverage * 100:.1f}% strict pose coverage", (242, HEIGHT - 27),
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
    left = np.asarray(arrays["left_target_position_m"], dtype=float)
    right = np.asarray(arrays["right_target_position_m"], dtype=float)
    accepted = np.asarray(arrays["both_accept"], dtype=bool)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    _style_model(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, SCENE_HEIGHT)
    data = mujoco.MjData(model)
    camera = _camera(np.vstack((left, right)), model)
    left_mocap = int(model.body_mocapid[model.body("left_target").id])
    right_mocap = int(model.body_mocapid[model.body("right_target").id])
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
                             max_geom=1024) as renderer:
            for frame_index, encoded_time in enumerate(timing.encoded_time_s):
                lower, upper, alpha = interpolation_sample(times, float(encoded_time))
                data.qpos[:] = (1 - alpha) * qpos[lower] + alpha * qpos[upper]
                data.mocap_pos[left_mocap] = (1 - alpha) * left[lower] + alpha * left[upper]
                data.mocap_pos[right_mocap] = (1 - alpha) * right[lower] + alpha * right[upper]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                _draw_trail(renderer.scene, left, times, float(encoded_time),
                            (0.12, 0.54, 0.95, 0.84))
                _draw_trail(renderer.scene, right, times, float(encoded_time),
                            (0.95, 0.31, 0.67, 0.84))
                scene_bgr = _blend_background(renderer.render())
                # An interpolated frame inherits the incoming edge's acceptance.
                audit_index = upper if alpha > 1e-12 else lower
                frame = _draw_editorial_frame(
                    scene_bgr, time_s=float(encoded_time),
                    duration_s=timing.source_duration_s,
                    following=bool(accepted[audit_index]),
                    coverage=float(summary["both_accept_coverage"]))
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
