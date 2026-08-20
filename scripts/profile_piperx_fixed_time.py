"""Profile a short, deterministic window of the PiperX fixed-time solver."""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from scripts.render_factory_dual_piperx_fixed_time import (
    ROOT,
    TABLE_HEIGHT_M,
    _registered_task,
    mount_separation_for_run,
    task_spec,
)
import scripts.render_factory_dual_xarm6_se3_follow as solver_module


def _slice_task(task, start, count):
    total = len(task.time_s)
    values = {}
    for name, value in task.__dict__.items():
        array = np.asarray(value)
        values[name] = array[start:start + count].copy() if (
            array.ndim and len(array) == total) else value
    values["time_s"] = np.asarray(values["time_s"], float)
    values["time_s"] -= values["time_s"][0]
    values["source_row_indices"] = np.arange(start, start + count)
    return SimpleNamespace(**values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("fold_box", "seal_bag"))
    parser.add_argument("mount_json", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--transition-steps", type=int, default=5)
    parser.add_argument(
        "--solver", choices=("paired", "xarm6-style"), default="paired")
    args = parser.parse_args(argv)

    mount_payload = json.loads(args.mount_json.read_text(encoding="utf-8"))
    mount = mount_payload.get("selected_mount", mount_payload)
    separation = mount_separation_for_run(mount, allow_legacy_mount=True)
    scene = (args.profile or Path("piperx_fixed_time.prof")).with_suffix(
        ".scene.xml")
    build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], separation, scene,
        table_height_m=TABLE_HEIGHT_M,
        mount_xy_m=mount["xy"], mount_yaw_deg=mount["yaw"],
        mount_adapter_height_m=float(mount["shared_base_z_m"]) - TABLE_HEIGHT_M,
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    task, _ = _registered_task(task_spec(args.task))
    task, mapped = solver_module.prepare_follow_targets(model, task)
    window = _slice_task(task, args.start, args.rows)
    mapped_window = {
        side: np.asarray(mapped[side])[args.start:args.start + args.rows]
        for side in ("left", "right")
    }

    profile_path = args.profile or (
        ROOT / "reports/factory_bimanual" /
        f"{args.task}_piperx_fixed_time_{args.start}_{args.rows}.prof")
    profiler = cProfile.Profile()
    original_config = solver_module.CollisionSafeFollowConfig
    original_checker = solver_module.MuJoCoPairedCollisionChecker

    def configured_planner(**kwargs):
        kwargs.update(
            beam_width=args.beam_width,
            maximum_pairs_per_row=args.max_pairs,
            maximum_candidates_per_tier=args.max_candidates,
        )
        return original_config(**kwargs)

    def configured_checker(model, data, names, *, transition_steps=5,
                           **kwargs):
        return original_checker(
            model, data, names, transition_steps=args.transition_steps,
            **kwargs)

    profiler.enable()
    with patch.object(
            solver_module, "CollisionSafeFollowConfig", configured_planner), \
         patch.object(
            solver_module, "MuJoCoPairedCollisionChecker", configured_checker):
        if args.solver == "paired":
            result = solver_module.solve_collision_safe_bimanual_method(
                model, window, mapped_window, velocity_limit_rad_s=3.0,
                robot_name="piperx", strict_pose_only=True,
                reference_aware_enabled=True,
                orientation_adaptation_enabled=True)
        else:
            result = solver_module.solve_multibranch_single_arm_method(
                model, window, mapped_window, horizon=12,
                beam_width=args.beam_width, candidate_iterations=60,
                velocity_limit_rad_s=3.0, global_retimed_no_flip=False,
                robot_name="piperx")
    profiler.disable()
    qpos, _, _, _, strict, _, _, _ = result
    audit_started = time.perf_counter()
    collision, _, _ = solver_module.audit_bimanual_collisions(
        model, qpos, robot_name="piperx")
    audit_s = time.perf_counter() - audit_started
    synchronous = strict["left"] & strict["right"] & ~collision
    profiler.dump_stats(str(profile_path))
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    stats.print_stats(30)
    print(json.dumps({
        "task": args.task,
        "start": args.start,
        "rows": args.rows,
        "total_s": stats.total_tt,
        "seconds_per_row": stats.total_tt / args.rows,
        "beam_width": args.beam_width,
        "maximum_pairs_per_row": args.max_pairs,
        "maximum_candidates_per_tier": args.max_candidates,
        "transition_steps": args.transition_steps,
        "solver": args.solver,
        "synchronous_strict_coverage": float(synchronous.mean()),
        "collision_frames": int(collision.sum()),
        "audit_s": audit_s,
        "profile": str(profile_path),
    }))


if __name__ == "__main__":
    main()
