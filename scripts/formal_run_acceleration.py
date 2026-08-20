"""Deterministic cache validation helpers for formal experiment runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


def input_fingerprint(paths: Iterable[Path], parameters: Mapping[str, object]) -> str:
    """Hash file contents and result-affecting parameters, independent of path order."""
    digest = hashlib.sha256()
    for path in sorted((Path(item).resolve() for item in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    digest.update(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _namespaced_fingerprint(
    namespace: str, paths: Iterable[Path], parameters: Mapping[str, object]
) -> str:
    if "namespace" in parameters:
        raise ValueError("namespace is reserved by the fingerprint helper")
    return input_fingerprint(paths, {"namespace": namespace, **parameters})


def search_fingerprint(paths: Iterable[Path], parameters: Mapping[str, object]) -> str:
    return _namespaced_fingerprint("strict-search-v2", paths, parameters)


def solve_fingerprint(paths: Iterable[Path], parameters: Mapping[str, object]) -> str:
    return _namespaced_fingerprint("strict-solve-v2", paths, parameters)


def render_fingerprint(paths: Iterable[Path], parameters: Mapping[str, object]) -> str:
    return _namespaced_fingerprint("strict-render-v2", paths, parameters)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish a complete JSON document without exposing a partial main file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def artifact_is_current(cache: Path, audit: Path, fingerprint: str) -> bool:
    """Return true only for a complete artifact produced from identical inputs."""
    cache, audit = Path(cache), Path(audit)
    if not cache.is_file() or not audit.is_file():
        return False
    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("input_fingerprint") == fingerprint


def json_fingerprint_matches(path: Path, fingerprint: str) -> bool:
    """Check a self-describing JSON result without trusting timestamps."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("input_fingerprint") == fingerprint
