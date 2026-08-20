"""Isolated, full-timeline experiment artifact persistence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FrameDiagnostics:
    source_row_index: int
    source_time_s: float
    left_failure_reason: str
    right_failure_reason: str
    collision: bool
    rollback: bool
    source_path: str
    execution_index: int | None = None
    execution_time_s: float | None = None
    execution_state: str | None = None
    # Execution collision evidence has two different domains.  A knot can be
    # collision-free even when the incoming interpolated edge is not.
    state_collision: bool | None = None
    incoming_transition_collision: bool | None = None


@dataclass(frozen=True)
class RunArtifactPaths:
    json_path: Path
    npz_path: Path


def _safe_relative_key(key: str) -> Path:
    pure = PurePosixPath(key.replace("\\", "/"))
    if pure.is_absolute() or not pure.parts or any(p in ("", ".", "..") for p in pure.parts):
        raise ValueError(f"unsafe run key: {key!r}")
    return Path(*pure.parts)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


class RunArtifactWriter:
    """Writes only beneath an explicitly supplied factory-bimanual root."""

    def __init__(self, root: Path, *, project_root: Path | None = None):
        self.root = Path(root).resolve()
        project = Path(project_root).resolve() if project_root is not None else self.root.parents[1]
        try:
            relative = self.root.relative_to(project)
        except ValueError as exc:
            raise ValueError("artifact root must be inside project root") from exc
        if tuple(part.lower() for part in relative.parts[-2:]) != ("reports", "factory_bimanual"):
            raise ValueError("artifact root must end with reports/factory_bimanual")

    def write_run(
        self,
        run_key: str,
        frames: Sequence[FrameDiagnostics],
        arrays: Mapping[str, np.ndarray],
        summary: Mapping[str, Any],
    ) -> RunArtifactPaths:
        directory = self.root / "runs" / _safe_relative_key(run_key)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "summary.json"
        npz_path = directory / "arrays.npz"
        payload = {"schema_version": 1, "summary": dict(summary),
                   "frames": [asdict(frame) for frame in frames]}
        _atomic_json(json_path, payload)
        # np.savez needs a filename suffix; write beside target and atomically replace.
        temp = directory / "arrays.tmp.npz"
        np.savez_compressed(temp, **{key: np.asarray(value) for key, value in arrays.items()})
        os.replace(temp, npz_path)
        return RunArtifactPaths(json_path, npz_path)
