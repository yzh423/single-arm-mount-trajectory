"""Locally refine the Fold_Box complete-follow bottom mount."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig,
    MuJoCoCandidateGenerator,
)
from factory_bimanual.piperx_recommended import (
    load_recommended_config,
    world_mount_for_family,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.task_family import TaskFamily
from scripts.run_piperx_recommended_v31 import (
    _registered_source,
    _source_path,
    condition_complete_follow_targets,
    resample_task_60hz,
)
from scripts.render_factory_dual_piperx_fixed_time import TABLE_HEIGHT_M


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".tmp/piperx_complete_mount_refine"
ROWS = {
    "left": (0, 75, 500, 1000, 1018, 1030, 1055, 1060),
    "right": (0, 45, 75, 117, 203, 220, 306, 350, 500, 628, 650, 1000, 1060),
}


def _generator(model):
    contract = ROBOT_CONTRACTS["piperx"]
    names = {
        side: {
            "joints": contract.prefixed_joint_names(side),
            "site": f"{side}_tcp",
        }
        for side in ("left", "right")
    }
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MuJoCoCandidateGenerator(
        model,
        data,
        contract,
        name_map=names,
        config=CandidateGeneratorConfig(
            position_tolerance_m=0.001,
            orientation_tolerance_rad=np.deg2rad(0.5),
            damping=0.3,
            step_scale=1.0,
            maximum_step_rad=0.3,
            position_error_clip_m=0.05,
            orientation_error_clip_rad=0.3,
            max_iterations=200,
            global_seed_count=12,
            maximum_candidates=1,
            constrained_fallback_enabled=False,
            stratified_seed_enabled=False,
            dedup_rad=np.deg2rad(0.25),
            rolling_early_stop_candidates=2,
        ),
    )


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    config = load_recommended_config()
    family = TaskFamily.parse("8-11/Fold_Box")
    source, registration = _registered_source(
        _source_path(family, "161044"), family)
    source = resample_task_60hz(source, rate_hz=60.0)
    spec = config.mounts[family.key]
    offsets = {
        "left": spec.left_tool_offset_quaternion_wxyz,
        "right": spec.right_tool_offset_quaternion_wxyz,
    }
    task, mapped, _ = condition_complete_follow_targets(source, offsets)
    mount = world_mount_for_family(
        config,
        family,
        registration.rotation_world_from_vr,
        registration.translation_world_m,
    )
    records = []
    for right_yaw in (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0):
        dx, dy, dz = 0.0, 0.0, 0.0
        left_dx, left_dy = 0.0, 0.0
        left_xyz = mount.left_xyz_m.copy()
        right_xyz = mount.right_xyz_m.copy()
        left_xyz[:2] += np.asarray((left_dx, left_dy))
        left_xyz[2] += dz
        right_xyz += np.asarray((dx, dy, dz))
        scene = WORK / (
            f"ldx_{left_dx:+.2f}_ldy_{left_dy:+.2f}.xml")
        build_same_model_scene(
            ROBOT_CONTRACTS["piperx"],
            float(np.linalg.norm(left_xyz-right_xyz)),
            scene,
            table_height_m=TABLE_HEIGHT_M,
            mount_xy_m={
                "left": left_xyz[:2].tolist(),
                "right": right_xyz[:2].tolist(),
            },
            mount_yaw_deg={"left": 0.0, "right": right_yaw},
            mount_adapter_height_m=float(left_xyz[2]-TABLE_HEIGHT_M),
            mount_support_mode=mount.mode,
        )
        model = mujoco.MjModel.from_xml_path(str(scene))
        generator = _generator(model)
        side_hits = {"left": [True] * len(ROWS["left"])}
        for side in ("right",):
            hits = []
            for row in ROWS[side]:
                candidates = generator.generate_target(
                    side,
                    getattr(task, f"{side}_position_m")[row],
                    mapped[side][row],
                    force_stratified=True,
                )
                hits.append(bool(candidates))
            side_hits[side] = hits
        record = {
            "dx_m": dx,
            "dy_m": dy,
            "dz_m": dz,
            "left_dx_m": left_dx,
            "left_dy_m": left_dy,
            "right_yaw_deg": right_yaw,
            "left_hits": int(sum(side_hits["left"])),
            "right_hits": int(sum(side_hits["right"])),
            "total_hits": int(sum(map(sum, side_hits.values()))),
            "left_mask": side_hits["left"],
            "right_mask": side_hits["right"],
            "left_xyz_m": left_xyz.tolist(),
            "right_xyz_m": right_xyz.tolist(),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    return sorted(
        records,
        key=lambda item: (
            -item["total_hits"], -item["right_hits"],
            abs(item["dx_m"])+abs(item["dy_m"])+abs(item["dz_m"]),
        ),
    )


def main():
    ranked = run()
    print("TOP")
    print(json.dumps(ranked[:10], indent=2))


if __name__ == "__main__":
    main()
