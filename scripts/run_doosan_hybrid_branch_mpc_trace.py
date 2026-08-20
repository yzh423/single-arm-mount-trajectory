"""View tune_01 with the tuned M0609 branch-front-end + 50 Hz MPC."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_doosan_realtime_scene import DEFAULT_OUTPUT, build_realtime_scene
from doosan_teleop.sim_app import main


if __name__ == "__main__":
    retimed_trace = ROOT / "traces/tune_01_retimed_080.json"
    retimed_plan = ROOT / "tuning/tune01_branch_full_retimed_080.npz"
    if "--trace" not in sys.argv:
        sys.argv.extend([
            "--trace",
            str(
                retimed_trace
                if retimed_trace.exists()
                else ROOT / "traces/tune_01.json"
            ),
        ])
    if "--branch-plan" not in sys.argv:
        sys.argv.extend([
            "--branch-plan",
            str(
                retimed_plan
                if retimed_plan.exists()
                else ROOT / "tuning/tune01_branch_full.npz"
            ),
        ])
    print(
        "[doosan hybrid] analytic/numerical branch plan + 1 s lookahead + "
        "rate-limited MPC; collision enabled; control=50 Hz"
    )
    raise SystemExit(main(
        "mpc_pvt",
        scene_builder=build_realtime_scene,
        scene_path=DEFAULT_OUTPUT,
        robot_label="M0609 hybrid branch MPC",
        log_tag="doosan_hybrid_branch_50hz",
        config_overrides={
            "task_tau": 0.032,
            "damping": 0.01,
            "singular_value_soft": 0.04,
            "singularity_cost": 0.5,
            "collision_cost": 800.0,
            "collision_margin": 0.015,
            "target_position_tau": 0.05,
            "target_rotation_tau": 0.06,
            "mocap_rate_limit_enabled": True,
            "target_max_speed": 2.0,
            "target_max_angular_speed": 5.0,
            "branch_reference_cost": 0.003,
            "candidate_switch_cost": 0.000001,
            "branch_reference_candidate_enabled": True,
            "settle_position_deadband": 0.0005,
            "settle_orientation_deadband": 0.00872664626,
        },
    ))
