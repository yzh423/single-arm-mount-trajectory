"""Render a complete factory handheld trajectory with two upright xArm6 arms."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.artifacts import FrameDiagnostics
from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.source_data import load_factory_task
from factory_bimanual.video import VideoRenderConfig, render_mujoco_mp4


def solve_position_follow(model, task, joint_names, *, iterations=50):
    data = mujoco.MjData(model)
    qpos = np.zeros((len(task.time_s), model.nq))
    actual = {side: np.zeros((len(task.time_s), 3)) for side in ("left", "right")}
    errors = {side: np.zeros(len(task.time_s)) for side in ("left", "right")}
    for side in ("left", "right"):
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                  for name in joint_names[side]]
        qids = np.asarray([model.jnt_qposadr[j] for j in joints])
        dids = np.asarray([model.jnt_dofadr[j] for j in joints])
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        targets = getattr(task, f"{side}_position_m")
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        for row, target in enumerate(targets):
            for _ in range(iterations):
                mujoco.mj_forward(model, data)
                error = target - data.site_xpos[site]
                if np.linalg.norm(error) < 5e-4:
                    break
                mujoco.mj_jacSite(model, data, jacp, jacr, site)
                jac = jacp[:, dids]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.0025 * np.eye(3), error)
                data.qpos[qids] += np.clip(0.75 * dq, -0.12, 0.12)
                ranges = model.jnt_range[joints]
                limited = model.jnt_limited[joints].astype(bool)
                data.qpos[qids[limited]] = np.clip(data.qpos[qids[limited]],
                                                    ranges[limited, 0], ranges[limited, 1])
            mujoco.mj_forward(model, data)
            actual[side][row] = data.site_xpos[site]
            errors[side][row] = np.linalg.norm(target - data.site_xpos[site])
            qpos[row, qids] = data.qpos[qids]
    return qpos, actual, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.50)
    parser.add_argument("--table-height", type=float, default=0.75)
    parser.add_argument("--z-offset", type=float, default=0.15)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()

    source = load_factory_task(args.csv, "seal_bag")
    all_points = np.vstack((source.left_position_m, source.right_position_m))
    translation = np.array([-all_points[:, 0].mean(), -all_points[:, 1].mean(),
                            args.table_height + args.z_offset - all_points[:, 2].min()])
    task = register_task(source, RigidTaskRegistration(np.eye(3), translation))
    scene = args.output.with_suffix(".scene.xml")
    manifest = build_same_model_scene(ROBOT_CONTRACTS["xarm6"], args.spacing, scene,
                                      table_height_m=args.table_height)
    model = mujoco.MjModel.from_xml_path(str(scene))
    names = {side: getattr(manifest, f"{side}_joint_names") for side in ("left", "right")}
    qpos, actual, errors = solve_position_follow(model, task, names)
    diagnostics = [FrameDiagnostics(int(i), float(t), "", "", False, False,
                                    str(source.source_path)) for i, t in enumerate(task.time_s)]
    result = render_mujoco_mp4(scene, args.output, task.time_s, qpos=qpos,
        diagnostics=diagnostics, left_targets=task.left_position_m,
        right_targets=task.right_position_m,
        config=VideoRenderConfig(width=1280, height=720, fps=args.fps,
                                 trajectory_radius_m=0.008))
    summary = {
        "source_csv": str(source.source_path), "source_rows": len(task.time_s),
        "source_duration_s": float(task.time_s[-1] - task.time_s[0]),
        "robot": "xarm6", "arm_count": 2, "mount_tilt_deg": [0.0, 0.0],
        "base_spacing_m": args.spacing, "table_height_m": args.table_height,
        "z_rule": "registered minimum source TCP z + offset",
        "source_min_z_m": float(all_points[:, 2].min()), "z_offset_m": args.z_offset,
        "registered_min_z_m": float(min(task.left_position_m[:, 2].min(), task.right_position_m[:, 2].min())),
        "registration_translation_m": translation.tolist(),
        "mount_xy_m": {"left": [-args.spacing / 2, 0.0], "right": [args.spacing / 2, 0.0]},
        "left_error_m": {"mean": float(errors["left"].mean()), "p95": float(np.quantile(errors["left"], .95)), "max": float(errors["left"].max())},
        "right_error_m": {"mean": float(errors["right"].mean()), "p95": float(np.quantile(errors["right"], .95)), "max": float(errors["right"].max())},
        "video": str(result.video_path), "decode_check": asdict(result.check),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".trajectory.npz"), qpos=qpos,
                        left_target=task.left_position_m, right_target=task.right_position_m,
                        left_actual=actual["left"], right_actual=actual["right"],
                        left_error_m=errors["left"], right_error_m=errors["right"])
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
