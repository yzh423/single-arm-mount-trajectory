"""Protect the established single-arm pipeline from bimanual experiment changes."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path


_EXPLICIT = (
    "scripts/search_strict_urdf_mount.py",
    "scripts/strict_mujoco_ik.py",
)
_TREES = ("reports/single_arm", "videos/single_arm")


def protected_paths(root: Path) -> tuple[Path, ...]:
    root = Path(root)
    relative: set[Path] = set()
    for name in _EXPLICIT:
        path = root / name
        if path.is_file():
            relative.add(Path(name))
    tests = root / "tests"
    if tests.is_dir():
        for path in tests.glob("test_*.py"):
            relative.add(path.relative_to(root))
    for tree_name in _TREES:
        tree = root / tree_name
        if tree.is_dir():
            for path in tree.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    relative.add(path.relative_to(root))
    return tuple(sorted(relative, key=lambda path: path.as_posix()))


def snapshot_protected_files(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        path.as_posix(): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in protected_paths(root)
    }


def assert_protected_files_unchanged(
    root: Path, snapshot: Mapping[str, str]
) -> None:
    current = snapshot_protected_files(root)
    changed = sorted(
        key for key in set(snapshot) | set(current)
        if snapshot.get(key) != current.get(key)
    )
    if changed:
        raise RuntimeError(f"protected single-arm files changed: {', '.join(changed)}")
