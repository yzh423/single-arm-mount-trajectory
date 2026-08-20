"""Stable date/task identity for factory bimanual trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class TaskFamily:
    """A factory task family whose date is part of its identity."""

    date: str
    task: str

    def __post_init__(self) -> None:
        for name, value in (("date", self.date), ("task", self.task)):
            if not value or "/" in value or "\\" in value:
                raise ValueError(f"{name} must be one non-empty path component")

    @property
    def key(self) -> str:
        return f"{self.date}/{self.task}"

    @classmethod
    def parse(cls, value: str) -> "TaskFamily":
        parts = str(value).replace("\\", "/").split("/")
        if len(parts) != 2:
            raise ValueError("task family must be formatted as <date>/<task>")
        return cls(parts[0], parts[1])


def family_from_path(path: Path, factory_root: Path) -> TaskFamily:
    """Read ``<date>/<task>`` from a CSV below a factory data root."""

    try:
        relative = Path(path).resolve().relative_to(Path(factory_root).resolve())
    except ValueError as error:
        raise ValueError("factory trajectory must be below the factory root") from error
    if len(relative.parts) < 3:
        raise ValueError("factory path must contain date, task, and file")
    return TaskFamily(relative.parts[0], relative.parts[1])
