from pathlib import Path

import pytest

from factory_bimanual.isolation import (
    assert_protected_files_unchanged,
    protected_paths,
    snapshot_protected_files,
)


ROOT = Path(__file__).parents[2]


def test_single_arm_sources_and_outputs_are_protected():
    paths = {path.as_posix() for path in protected_paths(ROOT)}
    assert "scripts/search_strict_urdf_mount.py" in paths
    assert "scripts/strict_mujoco_ik.py" in paths
    assert any(path.startswith("tests/test_") for path in paths)
    assert any(path.startswith("reports/single_arm/") for path in paths)
    assert any(path.startswith("videos/single_arm/") for path in paths)
    assert not any(path.startswith("tests/factory_bimanual/") for path in paths)


def test_hash_verifier_detects_a_changed_protected_file(tmp_path):
    protected = tmp_path / "scripts" / "search_strict_urdf_mount.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("before", encoding="utf-8")
    snapshot = snapshot_protected_files(tmp_path)
    protected.write_text("after", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        assert_protected_files_unchanged(tmp_path, snapshot)
