"""Retiming-only preprocessing for a recorded dual-mocap trace."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def _nlerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    result = (1.0 - amount) * q0 + amount * q1
    return result / max(float(np.linalg.norm(result)), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--linear-speed", type=float, default=0.8)
    parser.add_argument("--angular-speed", type=float, default=2.0)
    parser.add_argument("--hz", type=float, default=50.0)
    args = parser.parse_args()

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    samples = payload["samples"]
    source_time = np.asarray([float(sample["t"]) for sample in samples])
    source_position = np.asarray([sample["mocap_pos"] for sample in samples])
    source_quaternion = np.asarray([sample["mocap_quat"] for sample in samples])
    position_step = np.max(
        np.linalg.norm(np.diff(source_position, axis=0), axis=2), axis=1
    )
    quaternion_dot = np.min(
        np.abs(np.sum(
            source_quaternion[1:] * source_quaternion[:-1], axis=2
        )),
        axis=1,
    )
    orientation_step = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
    retimed_dt = np.maximum.reduce([
        np.diff(source_time),
        position_step / args.linear_speed,
        orientation_step / args.angular_speed,
    ])
    retimed_source_time = np.concatenate(([0.0], np.cumsum(retimed_dt)))
    output_time = np.arange(
        0.0, retimed_source_time[-1] + 0.5 / args.hz, 1.0 / args.hz
    )
    hi = np.minimum(
        np.searchsorted(retimed_source_time, output_time, side="right"),
        len(retimed_source_time) - 1,
    )
    lo = np.maximum(0, hi - 1)
    span = np.maximum(retimed_source_time[hi] - retimed_source_time[lo], 1e-12)
    alpha = np.clip(
        (output_time - retimed_source_time[lo]) / span, 0.0, 1.0
    )
    template = copy.deepcopy(samples[0])
    template.pop("debug", None)
    output_samples = []
    for index, target_time in enumerate(output_time):
        sample = copy.deepcopy(template)
        sample["t"] = float(target_time)
        sample["mocap_pos"] = (
            (1.0 - alpha[index]) * source_position[lo[index]]
            + alpha[index] * source_position[hi[index]]
        ).tolist()
        sample["mocap_quat"] = [
            _nlerp(
                source_quaternion[lo[index], side],
                source_quaternion[hi[index], side],
                float(alpha[index]),
            ).tolist()
            for side in range(source_quaternion.shape[1])
        ]
        output_samples.append(sample)
    metadata = dict(payload.get("metadata", {}))
    metadata["retiming"] = {
        "source": str(args.trace.resolve()),
        "linear_speed_m_s": args.linear_speed,
        "angular_speed_rad_s": args.angular_speed,
        "sample_hz": args.hz,
        "source_duration_s": float(source_time[-1] - source_time[0]),
        "retimed_duration_s": float(output_time[-1]),
        "path_geometry_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "format": payload.get("format", "dual_mocap_controller_trace_v2"),
                "metadata": metadata,
                "samples": output_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.plan is not None:
        if args.plan_output is None:
            raise ValueError("--plan-output is required with --plan")
        plan = np.load(args.plan)
        plan_source_time = plan["time"]
        plan_retimed_time = np.interp(
            plan_source_time,
            source_time,
            retimed_source_time,
        )
        arrays = {key: plan[key] for key in plan.files}
        arrays["time"] = plan_retimed_time
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.plan_output, **arrays)

    report = {
        "source_samples": len(samples),
        "retimed_samples": len(output_samples),
        "source_duration_s": float(source_time[-1] - source_time[0]),
        "retimed_duration_s": float(output_time[-1]),
        "maximum_linear_speed_m_s": args.linear_speed,
        "maximum_angular_speed_rad_s": args.angular_speed,
        "path_geometry_changed": False,
    }
    args.output.with_suffix(".retime.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
