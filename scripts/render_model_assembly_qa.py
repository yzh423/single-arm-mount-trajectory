"""Render close-up assembly/TCP evidence for native third-party robot models."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.strict_urdf_model_audit import MODELS, load_native_spec
from scripts.high_quality_robot_scene import align_robot_to_pedestal, decorate_high_quality_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robots", nargs="*", default=["i2rt_yam", "piperx", "arx_x5"])
    parser.add_argument("--output", type=Path, default=ROOT / "reports/single_arm/assembly_qa_v2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260808)
    report: dict[str, object] = {"robots": {}}
    for name in args.robots:
        entry = MODELS[name]
        spec = load_native_spec(entry)
        parent = spec.body(entry.tcp_parent)
        # Draw the red flange marker only when it will not hide the green TCP.
        if np.linalg.norm(entry.tool_translation_m) > 1e-9:
            parent.add_site(name="qa_flange", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.025, 0, 0], rgba=[0.95, 0.08, 0.04, 1])
        parent.add_site(name="qa_tcp", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        pos=entry.tool_translation_m, quat=entry.tool_quaternion_wxyz,
                        size=[0.025, 0, 0], rgba=[0.02, 0.95, 0.18, 1])
        if np.linalg.norm(entry.tool_translation_m) > 1e-9:
            parent.add_site(name="qa_flange_tcp_axis", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                            fromto=[0.0, 0.0, 0.0, *entry.tool_translation_m],
                            size=[0.006, 0, 0], rgba=[1.0, 0.72, 0.02, 0.9])
        mount_offset_m = align_robot_to_pedestal(spec, entry.joints)
        decorate_high_quality_scene(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in entry.joints]
        addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
        limits = [tuple(model.jnt_range[joint_id]) if model.jnt_limited[joint_id] else (-np.pi, np.pi)
                  for joint_id in joint_ids]
        poses = {
            "zero": np.zeros(len(addresses)),
            "middle": np.asarray([(lo + hi) / 2 for lo, hi in limits]),
            "random": np.asarray([rng.uniform(lo * .65, hi * .65) for lo, hi in limits]),
        }
        flange_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "qa_flange")
        tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "qa_tcp")
        rows = []
        for pose_name, q in poses.items():
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            # Mesh centers are the correct framing evidence. Body origins can
            # sit outside the actual visual envelope (notably YAM finger tips).
            visible = np.flatnonzero((model.geom_rgba[:, 3] > 0.01) & (model.geom_group == 1))
            points = np.vstack((data.geom_xpos[visible], data.site_xpos[flange_id], data.site_xpos[tcp_id]))
            lo, hi = points.min(axis=0), points.max(axis=0)
            center = (lo + hi) / 2
            span = max(float(np.linalg.norm(hi - lo)), 0.45)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = center
            camera.distance = 1.8 * span
            camera.azimuth = 145
            camera.elevation = -24
            # Some source MJCFs retain MuJoCo's 640 px offscreen framebuffer.
            renderer = mujoco.Renderer(model, height=480, width=640, max_geom=2048)
            scene_option = mujoco.MjvOption()
            scene_option.geomgroup[:] = 0
            scene_option.geomgroup[1] = 1  # real visual meshes only
            scene_option.geomgroup[2] = 1  # table/pedestal scene
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            image = renderer.render()
            output = args.output / f"{name}__{pose_name}.png"
            cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            renderer.close()
            distance = float(np.linalg.norm(data.site_xpos[tcp_id] - data.site_xpos[flange_id]))
            rows.append({"pose": pose_name, "flange_tcp_distance_m": distance, "image": str(output)})
        report["robots"][name] = {
            "source_model": str(entry.path), "tcp_parent": entry.tcp_parent,
            "flange_tcp_translation_m": list(entry.tool_translation_m),
            "mount_surface_offset_m": mount_offset_m, "poses": rows,
        }
        print(name, "pass", flush=True)
    report["status"] = "pass"
    (args.output / "qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
