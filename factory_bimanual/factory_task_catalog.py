"""Deterministic catalog of representative factory bimanual trajectories."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class FactoryEpisode:
    task_name: str
    csv_path: Path
    relative_path: str
    source_rows: int
    sha256: str


@dataclass(frozen=True)
class TaskRepresentative(FactoryEpisode):
    pass


@dataclass(frozen=True)
class ExcludedFactoryInput:
    relative_path: str
    reason: str


@dataclass(frozen=True)
class FactoryTaskCatalog:
    root: Path
    episodes: tuple[FactoryEpisode, ...]
    representatives: tuple[TaskRepresentative, ...]
    excluded: tuple[ExcludedFactoryInput, ...]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _source_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        count = sum(1 for _ in csv.reader(stream)) - 1
    if count < 1:
        raise ValueError(f"trajectory has no source rows: {path}")
    return count


def build_factory_task_catalog(root: Path) -> FactoryTaskCatalog:
    root = Path(root).resolve()
    episodes = []
    excluded = []
    for path in sorted(root.rglob("*.csv"), key=lambda p: _relative(p, root)):
        relative = _relative(path, root)
        if "_rejected_short" in path.parts:
            excluded.append(ExcludedFactoryInput(relative, "rejected_short"))
            continue
        if path.name.startswith("cleaning_manifest"):
            excluded.append(ExcludedFactoryInput(relative, "metadata_manifest"))
            continue
        episodes.append(FactoryEpisode(
            task_name=path.parent.name,
            csv_path=path.resolve(),
            relative_path=relative,
            source_rows=_source_rows(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ))
    grouped = {}
    for episode in episodes:
        grouped.setdefault(episode.task_name, []).append(episode)
    representatives = []
    for task_name in sorted(grouped):
        selected = sorted(grouped[task_name], key=lambda item: (
            -item.source_rows, item.relative_path))[0]
        representatives.append(TaskRepresentative(**asdict(selected)))
    return FactoryTaskCatalog(
        root=root, episodes=tuple(episodes),
        representatives=tuple(representatives), excluded=tuple(excluded))


def write_dataset_manifest(catalog: FactoryTaskCatalog, output: Path) -> None:
    def encode(item):
        value = asdict(item)
        if "csv_path" in value:
            value["csv_path"] = str(value["csv_path"])
        return value
    payload = {
        "schema": "piperx-factory-representatives-v1",
        "factory_root": str(catalog.root),
        "episodes": [encode(item) for item in catalog.episodes],
        "representatives": [encode(item) for item in catalog.representatives],
        "excluded": [encode(item) for item in catalog.excluded],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
