"""Coarse kinematic search for a 730 mm M0609-topology desktop arm.

The visual meshes are intentionally not rescaled; this experiment changes joint
origins only and is therefore a kinematic design study, not a production MJCF.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.easy_ik import DualArmEasyIKPVT
from doosan_teleop.four_robot_sim_app import (
    DatasetPlayback,
    SCENE,
    _align_targets,
    _names,
    robot_config,
)


def candidate_scene(
    source: Path,
    output: Path,
    upper: float,
    forearm: float,
    wrist: float,
    spacing: float,
    base_height: float,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    for side in ("left", "right"):
        edits = {
            f"doosan_{side}_link_3": (0, upper),
            f"doosan_{side}_link_4": (1, -forearm),
            f"doosan_{side}_link_6": (1, -wrist),
        }
        for body_name, (axis, value) in edits.items():
            body = root.find(f".//body[@name='{body_name}']")
            if body is None:
                raise RuntimeError(f"Missing body {body_name}")
            position = [float(item) for item in body.get("pos", "").split()]
            position[axis] = value
            body.set("pos", " ".join(f"{item:.9g}" for item in position))
        base = root.find(f".//body[@name='doosan_{side}_base']")
        position = [float(item) for item in base.get("pos", "").split()]
        position[0] = 1.35 + (-0.5 if side == "left" else 0.5) * spacing
        position[2] = base_height
        base.set("pos", " ".join(f"{item:.9g}" for item in position))
    tree.write(output, encoding="utf-8", xml_declaration=True)


def evaluate(
    scene: Path,
    dataset: Path,
    duration: float,
    rate: float,
    z_scale: float = 1.0,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 1.0 / rate
    data = mujoco.MjData(model)
    controller = DualArmEasyIKPVT(
        model,
        data,
        robot_config("doosan"),
        name_map=_names("doosan"),
        robot_kind="doosan",
    )
    controller.initialize_home()
    _align_targets(data, {"doosan": controller})
    playback = DatasetPlayback(
        dataset, data, {"doosan": controller}, 1.0, False, "canonical", z_scale
    )
    playback.apply_at(0.0)
    # A geometry candidate must not be penalized merely because it inherited
    # the 900 mm arm's hand-taught home branch.  Solve the common start pose
    # from deterministic multi-start seeds, independently for each arm.
    rng = np.random.default_rng(730)
    for arm in controller.arms.values():
        original = data.qpos[arm.qpos_ids].copy()
        seeds = [original]
        seed_lower = np.maximum(arm.q_min, np.radians(-175.0))
        seed_upper = np.minimum(arm.q_max, np.radians(175.0))
        for _ in range(20):
            seeds.append(rng.uniform(seed_lower, seed_upper))
        best_score = float("inf")
        best_q = original
        for seed in seeds:
            data.qpos[arm.qpos_ids] = np.clip(
                seed, arm.q_min + 1e-4, arm.q_max - 1e-4
            )
            data.qvel[arm.dof_ids] = 0.0
            controller._dq_filtered[arm.side][:] = 0.0
            mujoco.mj_forward(model, data)
            for _ in range(180):
                controller.step_arm(arm)
            position_error = float(arm.last_debug["pos_err"])
            orientation_error = float(arm.last_debug["rot_err"])
            sigma = float(arm.last_debug["sigma_min"])
            score = (
                1000.0 * position_error
                + 5.0 * orientation_error
                + 0.25 * max(0.0, 0.04 - sigma) / 0.04
            )
            if score < best_score:
                best_score = score
                best_q = data.qpos[arm.qpos_ids].copy()
        data.qpos[arm.qpos_ids] = best_q
        controller._dq_filtered[arm.side][:] = 0.0
        mujoco.mj_forward(model, data)
    for _ in range(round(2.0 * rate)):
        controller.step()

    metrics = {
        side: {"position": [], "sigma": [], "margin": []}
        for side in ("left", "right")
    }
    collision_frames = 0
    penetration_frames = 0
    steps = round(min(duration, float(playback.time[-1])) * rate) + 1
    for step in range(steps):
        playback.apply_at(step / rate)
        controller.step()
        collision = controller.collision_diagnostics(data)
        collision_frames += float(collision["penalty"]) > 0.0
        penetration_frames += float(collision["penetrating_pairs"]) > 0.0
        for side, arm in controller.arms.items():
            q = data.qpos[arm.qpos_ids]
            metrics[side]["position"].append(float(arm.last_debug["pos_err"]))
            metrics[side]["sigma"].append(float(arm.last_debug["sigma_min"]))
            metrics[side]["margin"].append(
                float(np.min(np.minimum(q - arm.q_min, arm.q_max - q)))
            )

    arms = {}
    for side, values in metrics.items():
        error = np.asarray(values["position"])
        arms[side] = {
            "rmse_mm": float(1000.0 * np.sqrt(np.mean(error**2))),
            "p95_mm": float(1000.0 * np.percentile(error, 95)),
            "max_mm": float(1000.0 * np.max(error)),
            "minimum_sigma": float(min(values["sigma"])),
            "minimum_joint_margin_deg": float(np.degrees(min(values["margin"]))),
        }
    worst_rmse = max(arm["rmse_mm"] for arm in arms.values())
    minimum_sigma = min(arm["minimum_sigma"] for arm in arms.values())
    minimum_margin = min(arm["minimum_joint_margin_deg"] for arm in arms.values())
    # Feasibility dominates; among feasible designs prefer conditioning and
    # useful joint margin.  Collision frames remain a separately visible term.
    score = (
        worst_rmse
        + 20.0 * max(0.0, 0.05 - minimum_sigma)
        + 0.05 * max(0.0, 15.0 - minimum_margin)
        + 0.01 * (100.0 * collision_frames / steps)
    )
    return {
        "score": score,
        "collision_frame_percent": 100.0 * collision_frames / steps,
        "penetration_frame_percent": 100.0 * penetration_frames / steps,
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "COFFAIL/benchmark/coffee_dual_active_set2_70s.csv",
    )
    parser.add_argument("--reach", type=float, default=0.730)
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.467,
        help="Scaled desktop shoulder height for the 730 mm geometry.",
    )
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tuning/doosan_730_geometry_search.json",
    )
    args = parser.parse_args()

    results = []
    generated: list[Path] = []
    # J3-origin, J4-origin and wrist offset retain the exact M0609 axis
    # topology; each triple sums to the requested maximum nominal reach.
    wrist_values = (0.080, 0.095, 0.110)
    upper_values = (0.315, 0.335, 0.355)
    spacing_values = (0.64, 0.70, 0.76)
    try:
        for wrist in wrist_values:
            for upper in upper_values:
                forearm = args.reach - wrist - upper
                for spacing in spacing_values:
                    handle = tempfile.NamedTemporaryFile(
                        prefix="scan_doosan730_",
                        suffix=".xml",
                        dir=args.scene.resolve().parent,
                        delete=False,
                    )
                    handle.close()
                    scene = Path(handle.name)
                    generated.append(scene)
                    candidate_scene(
                        args.scene.resolve(),
                        scene,
                        upper,
                        forearm,
                        wrist,
                        spacing,
                        args.base_height,
                    )
                    print(
                        f"[geometry] upper={upper:.3f} forearm={forearm:.3f} "
                        f"wrist={wrist:.3f} spacing={spacing:.3f}",
                        flush=True,
                    )
                    metrics = evaluate(
                        scene,
                        args.dataset.resolve(),
                        args.duration,
                        args.control_hz,
                        args.z_scale,
                    )
                    result = {
                        "upper_m": upper,
                        "forearm_m": forearm,
                        "wrist_m": wrist,
                        "base_spacing_m": spacing,
                        "base_height_m": args.base_height,
                        **metrics,
                    }
                    results.append(result)
                    print(json.dumps(result), flush=True)
    finally:
        for scene in generated:
            scene.unlink(missing_ok=True)

    results.sort(key=lambda item: item["score"])
    document = {
        "topology": "M0609 joint-axis topology; joint-origin-only kinematic study",
        "nominal_reach_m": args.reach,
        "base_height_m": args.base_height,
        "dataset": str(args.dataset.resolve()),
        "duration_s": args.duration,
        "control_hz": args.control_hz,
        "z_scale": args.z_scale,
        "best": results[0],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print("[best]", json.dumps(results[0], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
