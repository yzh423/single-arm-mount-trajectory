"""Prepare the canonical local single-arm pose benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from design_optimization.local_pose_dataset import prepare_local_pose_benchmark


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cleaning = payload.get("cleaning", {})
    manifest = prepare_local_pose_benchmark(
        _project_path(payload["source_root"]),
        _project_path(payload["output_root"]),
        maximum_internal_repair_samples=int(
            cleaning.get("maximum_internal_repair_samples", 3)
        ),
        exclude_invalid_fraction=float(cleaning.get("exclude_invalid_fraction", 0.20)),
    )
    if payload.get("audit_output"):
        dispositions: dict[str, int] = {}
        for record in manifest["source_records"]:
            key = record["disposition"]
            dispositions[key] = dispositions.get(key, 0) + 1
        splits: dict[str, int] = {}
        for episode in manifest["episodes"]:
            key = episode["split"]
            splits[key] = splits.get(key, 0) + 1
        audit = {
            "source_file_count": manifest["source_file_count"],
            "episode_count": len(manifest["episodes"]),
            "source_dispositions": dispositions,
            "episode_splits": splits,
        }
        audit_path = _project_path(payload["audit_output"])
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/local_pose_benchmark.yaml"
    )
    args = parser.parse_args()
    manifest = run(args.config)
    print(
        json.dumps(
            {
                "source_file_count": manifest["source_file_count"],
                "episode_count": len(manifest["episodes"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
