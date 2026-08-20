"""Replay a recorded target through both analytic M0609 wrist branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.easy_ik import DualArmEasyIKPVT
from doosan_teleop.mpc_pvt import MPCConfig


def wrist_flip(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    """Exact spherical-wrist alternate decomposition, fitted to joint ranges."""
    result = q.copy()
    result[3] += np.pi
    result[4] *= -1.0
    result[5] += np.pi
    for index in (3, 5):
        alternatives = result[index] + 2.0 * np.pi * np.arange(-2, 3)
        valid = alternatives[
            (alternatives >= q_min[index]) & (alternatives <= q_max[index])
        ]
        if len(valid):
            result[index] = valid[np.argmin(np.abs(valid - q[index]))]
    return np.clip(result, q_min, q_max)


def replay(scene: Path, samples: list[dict], flip_right: bool) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    solver = DualArmEasyIKPVT(
        model, data, MPCConfig(collision_detection_enabled=False), robot_kind="doosan"
    )
    solver.sub_iters = 18
    solver.alpha = 0.22
    solver.dls = 0.02
    first = samples[0]
    for side, arm in solver.arms.items():
        q = np.asarray(first["arms"][side]["q"], dtype=float)
        if flip_right and side == "right":
            q = wrist_flip(q, arm.q_min, arm.q_max)
        data.qpos[arm.qpos_ids] = q
        data.qvel[arm.dof_ids] = 0.0
    mujoco.mj_forward(model, data)

    q_log, pos_log, rot_log, sigma_log = [], [], [], []
    for sample in samples:
        data.mocap_pos[:] = np.asarray(sample["mocap_pos"])
        data.mocap_quat[:] = np.asarray(sample["mocap_quat"])
        solver.step()
        q_log.append(np.concatenate([
            data.qpos[solver.arms[s].qpos_ids].copy() for s in ("left", "right")
        ]))
        pos_log.append([solver.arms[s].last_debug["pos_err"] for s in ("left", "right")])
        rot_log.append([solver.arms[s].last_debug["rot_err"] for s in ("left", "right")])
        sigma_log.append([solver.arms[s].last_debug["sigma_min"] for s in ("left", "right")])
    q_log = np.asarray(q_log)
    pos_log = np.asarray(pos_log)
    rot_log = np.asarray(rot_log)
    sigma_log = np.asarray(sigma_log)
    return {
        "q": q_log,
        "position_error": pos_log,
        "orientation_error": rot_log,
        "sigma": sigma_log,
        "summary": {
            "position_rmse_mm": (1000.0 * np.sqrt(np.mean(pos_log**2, axis=0))).tolist(),
            "orientation_rmse_deg": np.degrees(
                np.sqrt(np.mean(rot_log**2, axis=0))
            ).tolist(),
            "minimum_sigma": np.min(sigma_log, axis=0).tolist(),
            "right_sigma_p05": float(np.percentile(sigma_log[:, 1], 5)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    samples = payload["samples"]
    scene = Path(payload.get("metadata", {}).get(
        "scene", ROOT / "models/dual_m0609_2f85_spacing084_50hz_scene.xml"
    ))
    normal = replay(scene, samples, False)
    flipped = replay(scene, samples, True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        time=np.asarray([sample["t"] for sample in samples]),
        normal_q=normal["q"],
        flipped_q=flipped["q"],
        normal_sigma=normal["sigma"],
        flipped_sigma=flipped["sigma"],
    )
    report = {
        "trace": str(args.trace.resolve()),
        "analytic_identity": "q4'=q4+pi, q5'=-q5, q6'=q6+pi",
        "normal": normal["summary"],
        "flipped": flipped["summary"],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
