"""One-command entry for the audited Doosan global-plan + local-MPC pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "COFFAIL/benchmark/coffee_dual_active_set2_70s.csv"
DEFAULT_PLAN = ROOT / "tuning/doosan_global_branch_plan_continuous_set2_z055.npz"
DEFAULT_RESULT = ROOT / "offline_results/doosan_recommended_planned_mpc.npz"


def run(arguments: list[str]) -> None:
    print("+", sys.executable, *arguments, flush=True)
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--replan",
        action="store_true",
        help="Recompute the expensive global branch plan before MPC.",
    )
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    plan = args.plan.resolve()
    output = args.output.resolve()
    if args.replan or not plan.exists():
        run(
            [
                "scripts/plan_collision_free_ik_branches.py",
                "--robot",
                "doosan",
                "--dataset",
                str(dataset),
                "--z-scale",
                "0.55",
                "--waypoint-dt",
                "1",
                "--beam-width",
                "10",
                "--ik-iterations",
                "70",
                "--random-restarts",
                "24",
                "--max-waypoint-jump-deg",
                "60",
                "--output",
                str(plan),
            ]
        )
        run(
            [
                "scripts/validate_branch_plan.py",
                str(plan),
                "--edge-samples",
                "51",
            ]
        )
    run(
        [
            "scripts/run_offline_mpc_consequence.py",
            "--robots",
            "doosan",
            "--dataset",
            str(dataset),
            "--z-scale",
            "0.55",
            "--branch-plan",
            str(plan),
            "--output",
            str(output),
        ]
    )
    if args.view:
        run(["scripts/view_mpc_consequence.py", "--result", str(output)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
