"""Search one fixed left/right handheld-frame to PiperX TCP calibration."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig,
    MuJoCoCandidateGenerator,
)
from factory_bimanual.mujoco_collision_adapter import (
    MuJoCoPairedCollisionChecker,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.tool_frame_calibration import (
    CalibrationArtifact,
    apply_fixed_tool_rotation,
    fixed_offset_quaternion,
    rank_calibration_result,
    representative_quaternion_indices,
)
from scripts import render_factory_dual_piperx_fixed_time as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/factory_bimanual/piperx_tool_frame_calibration.json"
SEARCH_REPORT = OUTPUT.with_name("piperx_tool_frame_calibration_search.json")
WORK = ROOT / ".tmp/piperx_tool_frame_calibration"

REFERENCE_MOUNTS = (
    {
        "xy": {"left": [-.35, .25], "right": [-.30, -.45]},
        "yaw": {"left": 15.0, "right": 15.0},
        "shared_base_z_m": .81,
    },
    {
        "xy": {"left": [-.38, .30], "right": [-.27, -.40]},
        "yaw": {"left": 0.0, "right": 30.0},
        "shared_base_z_m": .90,
    },
)


@dataclass
class Context:
    key: str
    task_name: str
    model: mujoco.MjModel
    checker: MuJoCoPairedCollisionChecker
    task: object
    indices: np.ndarray


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representative_indices(task, maximum):
    count = len(task.time_s)
    uniform = np.rint(np.linspace(0, count - 1, max(2, maximum // 2))).astype(int)
    orientation = []
    for side in ("left", "right"):
        orientation.extend(representative_quaternion_indices(
            getattr(task, f"{side}_quaternion_wxyz"),
            maximum=max(2, maximum // 2)).tolist())
    values = np.unique(np.r_[0, uniform, orientation, count - 1]).astype(int)
    if len(values) > maximum:
        keep = np.rint(np.linspace(0, len(values) - 1, maximum)).astype(int)
        values = values[keep]
    return values


def _contexts(maximum_frames):
    WORK.mkdir(parents=True, exist_ok=True)
    output = []
    contract = ROBOT_CONTRACTS["piperx"]
    names = {side: {
        "joints": contract.prefixed_joint_names(side),
        "site": f"{side}_tcp",
    } for side in ("left", "right")}
    for task_name in ("fold_box", "seal_bag"):
        task, _ = runner._registered_task(runner.task_spec(task_name))
        indices = _representative_indices(task, maximum_frames)
        for mount_index, mount in enumerate(REFERENCE_MOUNTS):
            left = np.asarray(mount["xy"]["left"], dtype=float)
            right = np.asarray(mount["xy"]["right"], dtype=float)
            separation = float(np.linalg.norm(right - left))
            scene = WORK / f"{task_name}_mount_{mount_index}.scene.xml"
            runner.build_same_model_scene(
                contract, separation, scene,
                table_height_m=runner.TABLE_HEIGHT_M,
                mount_xy_m=mount["xy"], mount_yaw_deg=mount["yaw"],
                mount_adapter_height_m=(
                    mount["shared_base_z_m"] - runner.TABLE_HEIGHT_M),
            )
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            output.append(Context(
                f"{task_name}:{mount_index}", task_name, model,
                MuJoCoPairedCollisionChecker(
                    model, data, names, transition_steps=5,
                    clearance_margin_m=.015),
                task, indices,
            ))
    return output


def _layers_for_side(contexts, side, axis_index, local_xyz_deg):
    offset = fixed_offset_quaternion(axis_index, local_xyz_deg)
    result = {}
    contract = ROBOT_CONTRACTS["piperx"]
    for context in contexts:
        names = {arm: {
            "joints": contract.prefixed_joint_names(arm),
            "site": f"{arm}_tcp",
        } for arm in ("left", "right")}
        data = mujoco.MjData(context.model)
        mujoco.mj_forward(context.model, data)
        generator = MuJoCoCandidateGenerator(
            context.model, data, contract, name_map=names,
            config=CandidateGeneratorConfig(
                max_iterations=45,
                global_seed_count=0,
                maximum_candidates=8,
                constrained_fallback_enabled=False,
                dedup_rad=np.deg2rad(1.0),
                stratified_seed_enabled=True,
                bounded_optimizer_enabled=True,
                wrist_risk_enabled=True,
                stratified_refresh_interval=4,
            ),
        )
        mapped = apply_fixed_tool_rotation(
            getattr(context.task, f"{side}_quaternion_wxyz"), offset)
        layers = []
        for row in context.indices:
            layers.append(tuple(generator.generate_target(
                side,
                getattr(context.task, f"{side}_position_m")[row],
                mapped[row],
            )))
        result[context.key] = layers
    return result


def _side_rank(layers):
    all_layers = [layer for values in layers.values() for layer in values]
    available = [layer for layer in all_layers if layer]
    coverage = len(available) / max(1, len(all_layers))
    count = float(np.mean([len(layer) for layer in all_layers]))
    sigma = [max(item.singularity_margin for item in layer)
             for layer in available]
    margin = [max(item.joint_limit_margin_rad for item in layer)
              for layer in available]
    return (-coverage, -count,
            -float(np.quantile(sigma, .1)) if sigma else np.inf,
            -float(np.quantile(margin, .1)) if margin else np.inf)


def _longest_false(values):
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def _pair_metrics(contexts, left_layers, right_layers):
    successes = []
    connectable = []
    longest = 0
    sigmas = []
    margins = []
    errors = []
    for context in contexts:
        context_success = []
        previous_pairs = None
        for left, right in zip(
                left_layers[context.key], right_layers[context.key]):
            safe = [(l, r) for l in left for r in right
                    if context.checker.state(l.q, r.q).valid]
            followed = bool(safe)
            context_success.append(followed)
            successes.append(followed)
            if safe:
                best = min(safe, key=lambda pair: (
                    -context.checker.clearance(
                        pair[0].q, pair[1].q).minimum_m,
                    pair[0].wrist_risk + pair[1].wrist_risk,
                    pair[0].pose_cost + pair[1].pose_cost,
                    -min(pair[0].singularity_margin,
                         pair[1].singularity_margin)))
                sigmas.append(min(best[0].singularity_margin,
                                  best[1].singularity_margin))
                margins.append(min(best[0].joint_limit_margin_rad,
                                   best[1].joint_limit_margin_rad))
                errors.append(
                    best[0].position_error_m / .001 +
                    best[0].orientation_error_rad / np.deg2rad(1.5) +
                    best[1].position_error_m / .001 +
                    best[1].orientation_error_rad / np.deg2rad(1.5))
            if previous_pairs is None:
                linked = followed
            else:
                linked = any(
                    max(np.max(np.abs(l.q - old_l.q)),
                        np.max(np.abs(r.q - old_r.q))) <= np.deg2rad(35.0)
                    and context.checker.transition(
                        (old_l.q, old_r.q), (l.q, r.q)).valid
                    for old_l, old_r in previous_pairs
                    for l, r in safe)
            connectable.append(linked)
            previous_pairs = safe if safe else previous_pairs
        longest = max(longest, _longest_false(context_success))
    return {
        "synchronous_strict_coverage": float(np.mean(successes)),
        "longest_failure_run_frames": int(longest),
        "connectable_safe_branch_ratio": float(np.mean(connectable)),
        "minimum_singularity_margin": float(min(sigmas)) if sigmas else 0.0,
        "minimum_joint_limit_margin_rad": float(min(margins)) if margins else 0.0,
        "mean_normalized_pose_error": float(np.mean(errors)) if errors else 1e9,
    }


def calibrate(*, maximum_frames=14, top_axes=5, local_refinement=True):
    contexts = _contexts(maximum_frames)
    discrete = {"left": {}, "right": {}}
    for side in ("left", "right"):
        for axis in range(24):
            layers = _layers_for_side(contexts, side, axis, (0., 0., 0.))
            discrete[side][axis] = layers
            print(side, "axis", axis, "rank", _side_rank(layers), flush=True)
    top = {side: sorted(discrete[side],
                        key=lambda axis: _side_rank(discrete[side][axis]))[
                            :top_axes]
           for side in ("left", "right")}
    records = []
    for left_axis in top["left"]:
        for right_axis in top["right"]:
            metrics = _pair_metrics(
                contexts, discrete["left"][left_axis],
                discrete["right"][right_axis])
            records.append({
                "left_axis_rotation_index": left_axis,
                "right_axis_rotation_index": right_axis,
                "left_local_xyz_deg": [0., 0., 0.],
                "right_local_xyz_deg": [0., 0., 0.],
                **metrics,
            })
    best = min(records, key=rank_calibration_result)
    left_layers = discrete["left"][best["left_axis_rotation_index"]]
    right_layers = discrete["right"][best["right_axis_rotation_index"]]
    angles = {"left": np.zeros(3), "right": np.zeros(3)}
    if local_refinement:
        for step in (10.0, 5.0, 2.5):
            improved = True
            while improved:
                improved = False
                for side in ("left", "right"):
                    for axis in range(3):
                        for direction in (-1.0, 1.0):
                            proposal = angles[side].copy()
                            proposal[axis] += direction * step
                            if np.any(np.abs(proposal) > 15.0):
                                continue
                            layers = _layers_for_side(
                                contexts, side,
                                best[f"{side}_axis_rotation_index"], proposal)
                            candidate_left = layers if side == "left" else left_layers
                            candidate_right = layers if side == "right" else right_layers
                            metrics = _pair_metrics(
                                contexts, candidate_left, candidate_right)
                            record = {
                                **best,
                                f"{side}_local_xyz_deg": proposal.tolist(),
                                **metrics,
                            }
                            records.append(record)
                            if rank_calibration_result(record) < rank_calibration_result(best):
                                best = record
                                angles[side] = proposal
                                if side == "left":
                                    left_layers = layers
                                else:
                                    right_layers = layers
                                improved = True
                                print("refine", side, proposal.tolist(), metrics,
                                      flush=True)
                                break
                        if improved:
                            break
                    if improved:
                        break
    fingerprints = {
        name: _fingerprint(runner.task_spec(name).csv)
        for name in ("fold_box", "seal_bag")
    }
    artifact = CalibrationArtifact(
        version=1, robot="piperx", tasks=("fold_box", "seal_bag"),
        source_fingerprints=fingerprints,
        left_offset_quaternion_wxyz=tuple(fixed_offset_quaternion(
            best["left_axis_rotation_index"], best["left_local_xyz_deg"])),
        right_offset_quaternion_wxyz=tuple(fixed_offset_quaternion(
            best["right_axis_rotation_index"], best["right_local_xyz_deg"])),
        left_axis_rotation_index=best["left_axis_rotation_index"],
        right_axis_rotation_index=best["right_axis_rotation_index"],
        left_local_xyz_deg=tuple(best["left_local_xyz_deg"]),
        right_local_xyz_deg=tuple(best["right_local_xyz_deg"]),
        metrics={key: value for key, value in best.items()
                 if key not in {
                     "left_axis_rotation_index", "right_axis_rotation_index",
                     "left_local_xyz_deg", "right_local_xyz_deg"}},
    )
    artifact.write(OUTPUT)
    SEARCH_REPORT.write_text(json.dumps({
        "reference_mounts": REFERENCE_MOUNTS,
        "representative_frames_per_task": maximum_frames,
        "top_axes_per_side": top_axes,
        "best": best,
        "records": sorted(records, key=rank_calibration_result),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "best": best},
                     ensure_ascii=False), flush=True)
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-frames", type=int, default=14)
    parser.add_argument("--top-axes", type=int, default=5)
    parser.add_argument("--skip-local-refinement", action="store_true")
    args = parser.parse_args(argv)
    calibrate(
        maximum_frames=args.maximum_frames,
        top_axes=args.top_axes,
        local_refinement=not args.skip_local_refinement,
    )


if __name__ == "__main__":
    main()
