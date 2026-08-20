"""Refine the exhaustive right-mount proxy shortlist with real MuJoCo IK."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig, MuJoCoCandidateGenerator,
)
from factory_bimanual.mujoco_collision_adapter import MuJoCoPairedCollisionChecker
from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.source_data import load_factory_task
from scripts.render_factory_dual_xarm6_se3_follow import (
    CSV, ROOT, SELECTED_MOUNT, map_source_quaternions_to_tcp,
)
from scripts.rolling_multibranch_ik import BranchCandidate, select_receding_horizon_path
from scripts.search_seal_bag_right_mount import (
    longest_failure_run, whole_trajectory_window_indices,
)
from scripts.strict_mujoco_ik import joint_periodic_mask


REPORT = ROOT / "reports/factory_bimanual/seal_bag_dual_xarm6"
INPUT = REPORT / "right_mount_fk_proxy_rank.json"
OUTPUT = REPORT / "right_mount_shortlist_real_ik.json"


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    proxy = json.loads(INPUT.read_text(encoding="utf-8"))
    shortlist = list(proxy["diverse_shortlist_64"])
    baseline = {
        "right_x_m": SELECTED_MOUNT["xy"]["right"][0],
        "right_y_m": SELECTED_MOUNT["xy"]["right"][1],
        "right_yaw_deg": SELECTED_MOUNT["yaw"]["right"],
        "source": "fixed_time_baseline",
    }
    candidates = [baseline] + [dict(item, source="fk_proxy_diverse") for item in shortlist]
    source = load_factory_task(CSV, "seal_bag")
    points = np.vstack((source.left_position_m, source.right_position_m))
    task = register_task(source, RigidTaskRegistration(np.eye(3), np.array([
        -points[:, 0].mean(), -points[:, 1].mean(), .9 - points[:, 2].min()])))
    windows = list(whole_trajectory_window_indices(
        frame_count=len(task.time_s), window_count=8, window_length=6))
    windows += [np.arange(start, start + 10) for start in (72, 1040, 1830, 2266)]
    contract = ROBOT_CONTRACTS["xarm6"]
    names = {side: {"joints": contract.prefixed_joint_names(side),
                    "site": f"{side}_tcp"} for side in ("left", "right")}
    results = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="seal_shortlist_") as directory:
        for number, candidate in enumerate(candidates):
            right_xy = (candidate["right_x_m"], candidate["right_y_m"])
            xy = {"left": SELECTED_MOUNT["xy"]["left"], "right": right_xy}
            yaw = {"left": SELECTED_MOUNT["yaw"]["left"],
                   "right": candidate["right_yaw_deg"]}
            distance = float(np.linalg.norm(np.asarray(xy["left"]) - right_xy))
            xml = Path(directory) / f"{number}.xml"
            build_same_model_scene(
                contract, distance, xml, table_height_m=.75,
                mount_xy_m=xy, mount_yaw_deg=yaw, mount_adapter_height_m=.12)
            model = mujoco.MjModel.from_xml_path(str(xml))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            joints = [mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"right_joint{i}")
                for i in range(1, 7)]
            qids = np.asarray(model.jnt_qposadr[joints])
            ranges = model.jnt_range[joints]
            home = mujoco.MjData(model)
            home.qpos[qids] = np.mean(ranges, axis=1)
            mujoco.mj_forward(model, home)
            site = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
            home_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(home_quaternion, home.site_xmat[site])
            values = task.__dict__.copy()
            values["right_quaternion_wxyz"] = map_source_quaternions_to_tcp(
                task.right_quaternion_wxyz, home_quaternion)
            candidate_task = SimpleNamespace(**values)
            generator = MuJoCoCandidateGenerator(
                model, data, contract, name_map=names,
                config=CandidateGeneratorConfig(
                    max_iterations=35, dedup_rad=np.deg2rad(1),
                    maximum_candidates=4, global_seed_count=6))
            checker = MuJoCoPairedCollisionChecker(
                model, data, names, transition_steps=2)
            periodic = joint_periodic_mask(model, joints)
            failures = []
            empty_frames = 0
            margins = []
            window_results = []
            for indices in windows:
                generator.reset()
                layers = []
                for row in indices:
                    proposals = generator(
                        model, contract, candidate_task, int(row), "right")
                    layer = tuple(BranchCandidate(
                        q=item.q.copy(), pose_valid=True,
                        collision_free=checker.side_state("right", item.q).valid,
                        position_error_m=float(item.position_error_m),
                        orientation_error_rad=float(item.orientation_error_rad),
                        joint_limit_margin=float(item.joint_limit_margin_rad),
                        singularity_margin=float(item.singularity_margin),
                        index=int(item.branch_index),
                    ) for item in proposals)
                    layers.append(layer)
                    empty_frames += int(not any(item.collision_free for item in layer))
                sample_time = task.time_s[indices]
                intervals = np.r_[sample_time[1] - sample_time[0], np.diff(sample_time)]
                path = select_receding_horizon_path(
                    layers=layers, initial_q=np.mean(ranges, axis=1),
                    periodic=periodic, dt_s=intervals,
                    velocity_limit_rad_s=np.full(6, 3.14),
                    cap_rad=np.full(6, np.deg2rad(35)), horizon=len(indices),
                    beam_width=8, reseed_after_empty_frames=None,
                    minimum_joint_limit_margin_rad=0,
                    minimum_singularity_margin=0,
                    maximum_recovery_wrist_distance_rad=np.inf,
                    transition_valid=None,
                )
                failed = np.asarray(path.recovery_mode) != "none"
                failed[0] = False
                failures.extend(failed.tolist())
                window_results.append({
                    "start": int(indices[0]), "end": int(indices[-1]),
                    "failed_edges": int(failed.sum()),
                    "longest_failed_run": longest_failure_run(failed),
                })
                margins.extend(next((item.joint_limit_margin
                                     for item in layers[row]
                                     if item.index == selected), np.nan)
                               for row, selected in enumerate(path.selected_indices))
            failures = np.asarray(failures, dtype=bool)
            result = {
                "right_x_m": right_xy[0], "right_y_m": right_xy[1],
                "right_yaw_deg": yaw["right"], "source": candidate["source"],
                "failed_edges": int(failures.sum()),
                "longest_failed_run": longest_failure_run(failures),
                "empty_frames": empty_frames,
                "min_selected_margin_rad": float(np.nanmin(margins)),
                "windows": window_results,
            }
            result["score"] = [
                result["failed_edges"], result["longest_failed_run"],
                result["empty_frames"], -result["min_selected_margin_rad"]]
            results.append(result)
            ranked = sorted(results, key=lambda item: tuple(item["score"]))
            _atomic_json(OUTPUT, {
                "status": "running" if number + 1 < len(candidates) else "complete",
                "completed": number + 1, "total": len(candidates),
                "elapsed_s": time.perf_counter() - started,
                "method": "12 source-time contiguous windows with real MuJoCo IK and state collision",
                "ranked_results": ranked,
            })
            print(f"{number + 1}/{len(candidates)} "
                  f"fail={result['failed_edges']} longest={result['longest_failed_run']} "
                  f"xy={right_xy} yaw={yaw['right']}", flush=True)


if __name__ == "__main__":
    main()
