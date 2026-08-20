"""Quantify target jumps, tracking error and joint-search jitter in a trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    samples = payload["samples"]
    time_s = np.asarray([sample["t"] for sample in samples], dtype=np.float64)
    dt = np.maximum(np.diff(time_s), 1e-6)
    mocap = np.asarray([sample["mocap_pos"] for sample in samples])
    result: dict[str, object] = {
        "trace": str(args.trace.resolve()),
        "samples": len(samples),
        "duration_s": float(time_s[-1] - time_s[0]),
        "sample_rate_hz": float((len(samples) - 1) / (time_s[-1] - time_s[0])),
        "metadata": payload.get("metadata", {}),
        "arms": {},
    }
    for arm_index, side in enumerate(("left", "right")):
        q = np.asarray([sample["arms"][side]["q"] for sample in samples])
        tcp = np.asarray([sample["arms"][side]["tcp_pos"] for sample in samples])
        target = mocap[:, arm_index]
        position_error = np.linalg.norm(target - tcp, axis=1)
        target_step = np.linalg.norm(np.diff(target, axis=0), axis=1)
        target_speed = target_step / dt
        q_step = np.diff(q, axis=0)
        q_speed = np.linalg.norm(q_step / dt[:, None], axis=1)
        stationary = target_speed < 0.002
        searching = stationary & (q_speed > 0.15)
        debug = [sample["arms"][side].get("debug", {}) for sample in samples]
        sigma = np.asarray([entry.get("sigma_min", np.nan) for entry in debug])
        worst_error_indices = np.argsort(position_error)[-8:][::-1]
        largest_target_jumps = np.argsort(target_step)[-8:][::-1]
        result["arms"][side] = {
            "target_translation_range_m": np.ptp(target, axis=0).tolist(),
            "position_rmse_mm": 1000.0 * float(np.sqrt(np.mean(position_error**2))),
            "position_p95_mm": 1000.0 * percentile(position_error, 95),
            "position_max_mm": 1000.0 * float(np.max(position_error)),
            "target_step_p95_mm": 1000.0 * percentile(target_step, 95),
            "target_step_max_mm": 1000.0 * float(np.max(target_step)),
            "target_steps_over_20mm": int(np.sum(target_step > 0.020)),
            "joint_step_p95_deg": float(np.degrees(percentile(np.abs(q_step), 95))),
            "joint_step_max_deg": float(np.degrees(np.max(np.abs(q_step)))),
            "joint_speed_p95_rad_s": percentile(q_speed, 95),
            "stationary_search_frame_percent": 100.0 * float(np.mean(searching)),
            "minimum_sigma": float(np.nanmin(sigma)),
            "worst_position_error_events": [
                {
                    "time_s": float(time_s[index]),
                    "error_mm": 1000.0 * float(position_error[index]),
                }
                for index in worst_error_indices
            ],
            "largest_target_jump_events": [
                {
                    "time_s": float(time_s[index + 1]),
                    "jump_mm": 1000.0 * float(target_step[index]),
                }
                for index in largest_target_jumps
            ],
        }
    output = args.output or args.trace.with_name(args.trace.stem + "_analysis.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[analysis] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
