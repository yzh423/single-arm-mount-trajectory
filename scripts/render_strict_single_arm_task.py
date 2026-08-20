"""Render one real-URDF trajectory at original source speed and 30 fps."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.solve_strict_urdf_task_cache import build_model
from scripts.strict_urdf_model_audit import MODELS
from scripts.strict_mujoco_ik import wrapped_joint_delta
from scripts.video_timeline import showcase_timeline, validation_timeline


WIDTH, HEIGHT, FPS, PLAYBACK_SPEED = 960, 540, 30, 1.0


def playback_frame_count(source_time_s: np.ndarray, fps: int = FPS, speed: float = PLAYBACK_SPEED) -> int:
    duration = float(source_time_s[-1] - source_time_s[0]) / speed
    return max(2, int(round(duration * fps)))


def interpolated_failure_reason(
    success: np.ndarray, failure_reason: np.ndarray, lower: int, upper: int,
) -> str:
    """Report the endpoint that actually makes an interpolated frame fail."""
    if bool(success[lower]) and bool(success[upper]):
        return "none"
    failed_index = lower if not bool(success[lower]) else upper
    return str(failure_reason[failed_index])


def path_render_indices(frame_count: int, maximum_points: int = 500) -> np.ndarray:
    if frame_count < 2:
        return np.arange(frame_count, dtype=int)
    count = min(frame_count, maximum_points)
    return np.unique(np.linspace(0, frame_count - 1, count).round().astype(int))


def interpolate_joint_positions(lower: np.ndarray, upper: np.ndarray, alpha: float,
                                periodic: np.ndarray) -> np.ndarray:
    """Interpolate periodic joints through their shortest physical rotation."""
    return np.asarray(lower, dtype=float) + float(alpha) * wrapped_joint_delta(upper, lower, periodic)


def interpolate_quaternion_wxyz(lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc SLERP for a displayed target TCP orientation."""
    first = np.asarray(lower, dtype=float)
    second = np.asarray(upper, dtype=float)
    first = first / max(float(np.linalg.norm(first)), 1e-12)
    second = second / max(float(np.linalg.norm(second)), 1e-12)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = first + float(alpha) * (second - first)
        return result / max(float(np.linalg.norm(result)), 1e-12)
    angle = math.acos(dot)
    scale = math.sin(angle)
    return (math.sin((1.0 - float(alpha)) * angle) / scale * first
            + math.sin(float(alpha) * angle) / scale * second)


def font(size: int):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sphere(scene, position, radius, color):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.asarray([radius, 0.0, 0.0]), np.asarray(position, float),
                        np.eye(3).reshape(-1), np.asarray(color, np.float32))
    scene.ngeom += 1


def connector(scene, start, end, radius, color):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(color, np.float32))
    function = getattr(mujoco, "mjv_connector", None) or mujoco.mjv_makeConnector
    function(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(start, float), np.asarray(end, float))
    scene.ngeom += 1


def orientation_triad(scene, origin, rotation, *, length: float, radius: float, alpha: float):
    colors = ((0.92, 0.12, 0.10, alpha), (0.10, 0.78, 0.18, alpha),
              (0.10, 0.34, 0.95, alpha))
    origin = np.asarray(origin, dtype=float)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    for axis, color in enumerate(colors):
        connector(scene, origin, origin + length * rotation[:, axis], radius, color)


def configure_visual_only_render(model: mujoco.MjModel) -> mujoco.MjvOption:
    """Keep collision geometry active in physics while hiding it from RGB output."""
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "_collision" in name:
            model.geom_group[geom_id] = 3
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    return option


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("local", "droid", "egodex"), required=True)
    parser.add_argument("--robot", choices=tuple(MODELS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path,
                        help="explicit NPZ cache, required when several episodes share one task name")
    parser.add_argument("--timeline", choices=("validation", "showcase"), default="validation")
    args = parser.parse_args()
    cache_path = args.cache or (ROOT / "videos/single_arm/strict_cache" / args.domain / args.robot / f"{args.task}.npz")
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    audit_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(cache_path)
    cache = np.load(cache_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    model = build_model(args.robot, cache["base_xyz_m"], float(cache["tilt_deg"]),
                        float(cache["yaw_deg"]) if "yaw_deg" in cache else 0.0,
                        float(cache["roll_deg"]) if "roll_deg" in cache else 0.0,
                        render_studio=True)
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    entry = MODELS[args.robot]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in entry.joints]
    addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    joint_ranges = np.asarray([model.jnt_range[joint_id] for joint_id in joint_ids])
    periodic_joints = (joint_ranges[:, 1] - joint_ranges[:, 0]) >= (2.0 * np.pi - 1e-6)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "strict_tracking_tcp")
    q_path = np.asarray(cache["q"])
    targets = np.asarray(cache["target_xyz_m"])
    target_quaternions = (np.asarray(cache["target_quaternion_wxyz"], dtype=float)
                          if "target_quaternion_wxyz" in cache
                          else np.tile((1.0, 0.0, 0.0, 0.0), (len(targets), 1)))
    source_time_s = np.asarray(cache["time_s"]) if "time_s" in cache else np.arange(len(q_path), dtype=float)
    if args.timeline == "showcase":
        position_motion = np.linalg.norm(np.diff(targets, axis=0), axis=1) > 1e-9
        if "target_quaternion_wxyz" in cache:
            target_quat = np.asarray(cache["target_quaternion_wxyz"], dtype=float)
            target_quat /= np.maximum(np.linalg.norm(target_quat, axis=1, keepdims=True), 1e-12)
            orientation_motion = np.abs(np.sum(target_quat[:-1] * target_quat[1:], axis=1)) < (1.0 - 1e-10)
        else:
            orientation_motion = np.zeros(len(targets) - 1, dtype=bool)
        timeline_mapping = showcase_timeline(
            source_time_s, position_motion | orientation_motion)
        display_time_s = np.asarray(timeline_mapping["display_time_s"], dtype=float)
    else:
        display_time_s = validation_timeline(source_time_s)
        timeline_mapping = {
            "mode": "validation_original_time", "display_only": False,
            "source_time_s": source_time_s.tolist(),
            "display_time_s": display_time_s.tolist(),
        }
    frame_count = playback_frame_count(display_time_s)
    success = np.asarray(cache["success"], dtype=bool)
    failure_reason = (np.asarray(cache["failure_reason"]).astype(str)
                      if "failure_reason" in cache else np.where(success, "none", "pose_failure"))
    clouds = [targets]
    for q in q_path:
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        clouds.append(data.xpos[1:].copy())
    cloud = np.concatenate(clouds)
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    lookat = 0.5 * (low + high)
    span = float(np.linalg.norm(high - low))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = max(1.15, min(2.8, span * 1.45 + 0.25))
    # View from the task side of the table. The former 135-degree view placed
    # the pedestal between the camera and the target path for common mounts.
    camera.azimuth = 315
    camera.elevation = -24
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH, max_geom=2048)
    scene_option = configure_visual_only_render(model)
    output = args.output or ROOT / "videos/single_arm/strict_per_arm_task" / args.domain / args.robot / f"{args.task}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)
    ], stdin=subprocess.PIPE)
    title_font, body_font = font(25), font(17)
    try:
        render_indices = path_render_indices(len(targets))
        for frame in range(frame_count):
            desired_display_time = (display_time_s[0] + frame
                                    * (display_time_s[-1] - display_time_s[0])
                                    / (frame_count - 1))
            desired_time = float(np.interp(
                desired_display_time, display_time_s, source_time_s))
            upper = int(np.searchsorted(source_time_s, desired_time, side="right")); upper = min(max(upper, 1), len(q_path)-1)
            lower = upper - 1
            interval = max(float(source_time_s[upper] - source_time_s[lower]), 1e-12)
            alpha = min(1.0, max(0.0, float((desired_time - source_time_s[lower]) / interval)))
            phase = lower + alpha
            q = interpolate_joint_positions(q_path[lower], q_path[upper], alpha, periodic_joints)
            target = (1.0 - alpha) * targets[lower] + alpha * targets[upper]
            target_quaternion = interpolate_quaternion_wxyz(
                target_quaternions[lower], target_quaternions[upper], alpha)
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            camera.azimuth = 312 + 8 * math.sin(2 * math.pi * frame / frame_count)
            renderer.update_scene(data, camera, scene_option=scene_option)
            for first, second in zip(render_indices[:-1], render_indices[1:]):
                attempted = second <= phase
                color = [0.12, 0.42, 0.86, 0.95] if attempted else [0.52, 0.54, 0.56, 0.45]
                connector(renderer.scene, targets[first], targets[second], 0.0045, color)
            sphere(renderer.scene, target, 0.016, [0.86, 0.08, 0.08, 0.95])
            sphere(renderer.scene, data.site_xpos[site_id], 0.012, [0.02, 0.78, 0.38, 1.0])
            target_rotation = np.empty(9)
            mujoco.mju_quat2Mat(target_rotation, target_quaternion)
            orientation_triad(renderer.scene, target, target_rotation, length=0.065,
                              radius=0.0022, alpha=0.42)
            orientation_triad(renderer.scene, data.site_xpos[site_id], data.site_xmat[site_id],
                              length=0.052, radius=0.0030, alpha=0.95)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rounded_rectangle((18, 15, WIDTH - 18, 88), 10, fill=(250, 250, 250, 225))
            frame_ok = bool(success[lower] and success[upper])
            draw.text((34, 24), f"{args.robot} | {args.domain} | {args.task}", font=title_font, fill=(18, 22, 28, 255))
            reason = interpolated_failure_reason(
                success, failure_reason, lower, upper)
            timeline_label = "SHOWCASE dwell-compressed" if args.timeline == "showcase" else "VALIDATION original-time"
            draw.text((34, 56), f"t={frame/FPS:4.1f}s  {timeline_label}  {'FOLLOW' if frame_ok else 'FAIL: '+reason}  red=target  green=actual TCP  RGB=TCP axes", font=body_font,
                      fill=((20, 130, 72, 255) if frame_ok else (205, 48, 48, 255)))
            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
    finally:
        if process.stdin:
            process.stdin.close()
        code = process.wait()
        renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg exited {code}")
    try:
        output_label = str(output.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        output_label = str(output.resolve()).replace("\\", "/")
    metadata = {**audit, "renderer": "MuJoCo real URDF/MJCF", "width": WIDTH, "height": HEIGHT,
                "fps": FPS, "frames": frame_count, "duration_s": frame_count / FPS,
                "source_duration_s": float(source_time_s[-1] - source_time_s[0]),
                "playback_speed": f"{PLAYBACK_SPEED:g}x", "timeline_mode": args.timeline,
                "display_only_timing": args.timeline == "showcase", "output": output_label}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output.with_suffix(".timeline.json").write_text(
        json.dumps(timeline_mapping, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
