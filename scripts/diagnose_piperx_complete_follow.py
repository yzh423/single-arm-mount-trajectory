"""Probe PiperX pose feasibility under competing source-to-TCP mappings.

This is a deterministic diagnosis utility for the strict 1 mm / 0.5 degree
complete-follow pipeline.  It intentionally evaluates one arm and a small set
of frames without changing any production artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig,
    MuJoCoCandidateGenerator,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.task_family import TaskFamily
from factory_bimanual.tool_frame_calibration import (
    CalibrationArtifact,
    apply_fixed_tool_rotation,
    fixed_offset_quaternion,
)
from scripts.render_factory_dual_piperx_fixed_time import CALIBRATION_PATH
from scripts.render_factory_dual_xarm6_se3_follow import prepare_follow_targets
from scripts.run_piperx_recommended_v31 import (
    _registered_source,
    _source_path,
    resample_task_60hz,
    smooth_follow_targets,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = (
    ROOT / "reports/piperx_recommended_v31/"
    "8-11_Fold_Box_161044_recommended_v31.scene.xml"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--family", default="8-11/Fold_Box")
    parser.add_argument("--source-take", default="161044")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--rows", type=int, nargs="+", default=(75,))
    parser.add_argument("--axis-scan", action="store_true")
    parser.add_argument("--axes", type=int, nargs="+")
    parser.add_argument("--slerp-axis", type=int)
    parser.add_argument(
        "--slerp-start", choices=("midpoint", "locked"), default="midpoint")
    parser.add_argument("--slerp-steps", type=int, default=10)
    parser.add_argument("--bounded", action="store_true")
    parser.add_argument("--global-seeds", type=int, default=40)
    parser.add_argument("--follow", action="store_true")
    return parser.parse_args(argv)


def _task(options, model):
    family = TaskFamily.parse(options.family)
    source, _ = _registered_source(
        _source_path(family, options.source_take), family)
    source = resample_task_60hz(source, rate_hz=60.0)
    calibration = CalibrationArtifact.read(CALIBRATION_PATH)
    calibrated_task, calibrated = prepare_follow_targets(
        model, source, calibration=calibration)
    calibrated_task, calibrated = smooth_follow_targets(
        calibrated_task, calibrated)
    midpoint_task, midpoint = prepare_follow_targets(
        model, source, calibration=None)
    midpoint_task, midpoint = smooth_follow_targets(midpoint_task, midpoint)
    return source, calibrated_task, calibrated, midpoint_task, midpoint


def _generator(model, options):
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
            max_iterations=200,
            global_seed_count=options.global_seeds,
            maximum_candidates=8,
            constrained_fallback_enabled=True,
            constrained_fallback_seed_count=20,
            constrained_fallback_max_iterations=300,
            stratified_seed_enabled=True,
            bounded_optimizer_enabled=options.bounded,
            dedup_rad=np.deg2rad(0.25),
            rolling_early_stop_candidates=8,
        ),
    )


def _probe(model, options, task, mapped, label):
    output = []
    for row in options.rows:
        generator = _generator(model, options)
        candidates = generator.generate_target(
            options.side,
            getattr(task, f"{options.side}_position_m")[row],
            mapped[options.side][row],
            force_stratified=True,
        )
        misses = list(generator._near_misses)
        best = min(
            misses,
            key=lambda item: max(
                item[2] / 0.001,
                item[3] / np.deg2rad(0.5),
            ),
            default=None,
        )
        output.append({
            "mapping": label,
            "row": int(row),
            "candidate_count": len(candidates),
            "best_position_error_mm": (
                None if best is None else 1000.0 * float(best[2])
            ),
            "best_orientation_error_deg": (
                None if best is None else float(np.rad2deg(best[3]))
            ),
            "best_normalized_error": (
                None if best is None else float(max(
                    best[2] / 0.001,
                    best[3] / np.deg2rad(0.5),
                ))
            ),
        })
    return output


def _follow_side(model, options, task, mapped, label):
    """Reproduce report algorithm two as an independent single-arm state."""
    generator = _generator(model, options)
    count = len(task.time_s)
    accepted = np.zeros(count, dtype=bool)
    failures = {"anchor": 0, "yam_ik": 0, "branch_guard": 0}
    previous = None
    max_step = 0.0
    for row in range(count):
        target_p = getattr(task, f"{options.side}_position_m")[row]
        target_q = mapped[options.side][row]
        if row == 0:
            candidates = generator.generate_target(
                options.side, target_p, target_q, force_stratified=True)
            candidate = candidates[0] if candidates else None
        else:
            candidate = generator.generate_warm_start_candidate(
                options.side,
                target_p,
                target_q,
                reference_q=previous,
            )
        if candidate is None:
            failures["anchor" if row == 0 else "yam_ik"] += 1
            continue
        step = 0.0 if previous is None else float(
            np.max(np.abs(candidate.q - previous)))
        max_step = max(max_step, step)
        if previous is not None and step > 0.30 + 1e-12:
            failures["branch_guard"] += 1
            continue
        accepted[row] = True
        previous = candidate.q.copy()
    false_runs = []
    start = None
    for row, value in enumerate(accepted):
        if not value and start is None:
            start = row
        if value and start is not None:
            false_runs.append((start, row - 1))
            start = None
    if start is not None:
        false_runs.append((start, count - 1))
    return {
        "mapping": label,
        "side": options.side,
        "frames": count,
        "accepted_frames": int(np.count_nonzero(accepted)),
        "coverage": float(np.mean(accepted)),
        "failure_counts": failures,
        "failure_runs": false_runs,
        "maximum_candidate_step_rad": max_step,
    }


def run(options):
    model = mujoco.MjModel.from_xml_path(str(options.scene.resolve()))
    source, calibrated_task, calibrated, midpoint_task, midpoint = _task(
        options, model)
    if options.follow:
        return [
            _follow_side(
                model, options, calibrated_task, calibrated,
                "locked_calibration"),
            _follow_side(
                model, options, midpoint_task, midpoint,
                "scene_midpoint"),
        ]
    records = []
    records.extend(_probe(
        model, options, calibrated_task, calibrated, "locked_calibration"))
    records.extend(_probe(
        model, options, midpoint_task, midpoint, "scene_midpoint"))
    if options.slerp_axis is not None:
        source_quaternion = getattr(
            source, f"{options.side}_quaternion_wxyz")[0]
        source_inverse = source_quaternion.copy()
        source_inverse[1:] *= -1.0
        midpoint_offset = np.empty(4)
        mujoco.mju_mulQuat(
            midpoint_offset, source_inverse, midpoint[options.side][0])
        midpoint_offset /= np.linalg.norm(midpoint_offset)
        locked_offset = np.empty(4)
        mujoco.mju_mulQuat(
            locked_offset, source_inverse, calibrated[options.side][0])
        locked_offset /= np.linalg.norm(locked_offset)
        start_offset = (
            midpoint_offset if options.slerp_start == "midpoint"
            else locked_offset
        )
        axis_offset = fixed_offset_quaternion(
            options.slerp_axis, (0.0, 0.0, 0.0))
        endpoints = Rotation.from_quat(np.asarray([
            start_offset[[1, 2, 3, 0]],
            axis_offset[[1, 2, 3, 0]],
        ]))
        fractions = np.linspace(0.0, 1.0, options.slerp_steps+1)
        interpolated = Slerp([0.0, 1.0], endpoints)(fractions).as_quat()
        for fraction, xyzw in zip(fractions, interpolated):
            offset = xyzw[[3, 0, 1, 2]]
            mapped = {
                side: apply_fixed_tool_rotation(
                    getattr(source, f"{side}_quaternion_wxyz"), offset)
                for side in ("left", "right")
            }
            task, mapped = smooth_follow_targets(source, mapped)
            records.extend(_probe(
                model, options, task, mapped,
                f"slerp_{options.slerp_start}_axis_"
                f"{options.slerp_axis:02d}_{fraction:.3f}"))
    axes = range(24) if options.axis_scan else (options.axes or ())
    for axis in axes:
            offset = fixed_offset_quaternion(axis, (0.0, 0.0, 0.0))
            mapped = {
                side: apply_fixed_tool_rotation(
                    getattr(source, f"{side}_quaternion_wxyz"), offset)
                for side in ("left", "right")
            }
            task, mapped = smooth_follow_targets(source, mapped)
            records.extend(_probe(
                model, options, task, mapped, f"axis_{axis:02d}"))
    return records


def main(argv=None):
    print(json.dumps(run(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
