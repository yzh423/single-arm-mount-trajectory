"""Quarantine CSV episodes whose recorded duration is below a threshold."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    task: str
    rows: int
    first_t: float
    last_t: float

    @property
    def duration_s(self) -> float:
        return self.last_t - self.first_t


def inspect_episode(path: Path) -> EpisodeInfo:
    """Read timing metadata without loading the full episode into memory."""
    first_t: float | None = None
    last_t: float | None = None
    rows = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "t" not in reader.fieldnames:
            raise ValueError(f"missing required 't' column: {path}")
        for row in reader:
            try:
                value = float(row["t"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid t value at data row {rows + 1}: {path}") from exc
            if first_t is None:
                first_t = value
            last_t = value
            rows += 1
    if first_t is None or last_t is None:
        raise ValueError(f"episode has no data rows: {path}")
    return EpisodeInfo(Path(path), Path(path).parent.name, rows, first_t, last_t)


def scan_episodes(root: Path) -> list[EpisodeInfo]:
    root = Path(root)
    paths = sorted(
        path
        for path in root.rglob("*.csv")
        if path.parent != root and "_rejected_short" not in path.parts
    )
    return [inspect_episode(path) for path in paths]


def is_short(info: EpisodeInfo, threshold_s: float) -> bool:
    return info.duration_s < threshold_s


def quarantine_path(info: EpisodeInfo, root: Path) -> Path:
    directory = Path(root) / "_rejected_short" / info.task
    candidate = directory / info.path.name
    suffix = 2
    while candidate.exists():
        candidate = directory / f"{info.path.stem}_{suffix}{info.path.suffix}"
        suffix += 1
    return candidate


def _write_manifests(records: list[dict], prefix: Path) -> None:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "destination",
        "task",
        "rows",
        "duration_s",
        "threshold_s",
        "action",
        "recorded_at_utc",
    ]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def clean(
    root: Path,
    threshold_s: float = 5.0,
    apply: bool = False,
    manifest_prefix: Path | None = None,
) -> list[dict]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"episode root does not exist: {root}")
    if threshold_s < 0:
        raise ValueError("threshold must be non-negative")

    recorded_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    for info in scan_episodes(root):
        if not is_short(info, threshold_s):
            continue
        destination = quarantine_path(info, root)
        record = {
            "source": info.path.resolve().as_posix(),
            "destination": destination.resolve().as_posix(),
            "task": info.task,
            "rows": info.rows,
            "duration_s": round(info.duration_s, 9),
            "threshold_s": threshold_s,
            "action": "quarantined" if apply else "would_quarantine",
            "recorded_at_utc": recorded_at,
        }
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(info.path), str(destination))
        records.append(record)

    if manifest_prefix is not None:
        _write_manifests(records, Path(manifest_prefix))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root directory containing task CSV files")
    parser.add_argument("--threshold", type=float, default=5.0, help="Reject durations below this value")
    parser.add_argument("--apply", action="store_true", help="Move rejected files; default is dry-run")
    parser.add_argument("--manifest-prefix", type=Path, help="Output path without .csv/.json suffix")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = clean(args.root, args.threshold, args.apply, args.manifest_prefix)
    for record in records:
        print(
            f"{record['action']}: {record['source']} "
            f"({record['duration_s']:.3f}s) -> {record['destination']}"
        )
    print(f"Selected {len(records)} episode(s) below {args.threshold:g}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
