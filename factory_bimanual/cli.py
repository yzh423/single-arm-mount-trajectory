"""Command line entry point for safe factory-bimanual experiment staging."""
from __future__ import annotations

import argparse
from pathlib import Path

from .executors import ProductionExecutors
from .preflight import run_preflight
from .run_experiment import ExperimentConfig, FactoryBimanualExperiment


def _parser():
    parser = argparse.ArgumentParser(prog="python -m factory_bimanual.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("dry-run")
    short = sub.add_parser("short-prefix"); short.add_argument("--rows", type=int, default=2)
    full = sub.add_parser("full"); full.add_argument("--confirm-full", action="store_true")
    return parser


def main(argv=None, *, root: Path | None = None):
    args = _parser().parse_args(argv)
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    output = (root / "reports/factory_bimanual").resolve()
    executors = ProductionExecutors(root, output_root=output)
    if args.command == "preflight":
        return run_preflight(root, executors=executors.mapping())
    if args.command == "full" and not args.confirm_full:
        raise SystemExit("full experiment requires explicit --confirm-full")
    if args.command == "full":
        report = run_preflight(root, executors=executors.mapping())
        if not report["ready"]:
            raise SystemExit("preflight is not ready: " + "; ".join(report["blockers"]))
    config = ExperimentConfig(output, project_root=root, dry_run=args.command == "dry-run",
                              short_prefix_rows=args.rows if args.command == "short-prefix" else None)
    return FactoryBimanualExperiment(config, ik_executor=executors.ik,
                                     mpc_executor=executors.mpc).run()


if __name__ == "__main__":
    main()
