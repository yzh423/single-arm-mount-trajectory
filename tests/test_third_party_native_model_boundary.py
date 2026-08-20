from __future__ import annotations

from pathlib import Path

import yaml

from scripts.strict_urdf_model_audit import MODELS, OFFICIAL


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROBOTS = {
    "doosan",
    "xarm6",
    "ur5",
    "kinova_gen3_lite",
    "arx_x5",
    "big_yam",
    "franka_panda",
    "franka_panda_locked_j3",
    "i2rt_yam",
    "nero",
    "openarm",
    "piperx",
    "willow",
}


def _below(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def test_all_thirteen_models_are_third_party_native_sources() -> None:
    assert set(MODELS) == EXPECTED_ROBOTS
    for name, entry in MODELS.items():
        assert entry.path.is_file(), (name, entry.path)
        assert _below(entry.path, OFFICIAL), (name, entry.path)
        if entry.mesh_dir is not None:
            assert entry.mesh_dir.is_dir(), (name, entry.mesh_dir)
            assert _below(entry.mesh_dir, OFFICIAL), (name, entry.mesh_dir)
        for package_root in (entry.package_roots or {}).values():
            assert package_root.is_dir(), (name, package_root)
            assert _below(package_root, OFFICIAL), (name, package_root)


def test_registry_has_no_assets_or_scalable_segments() -> None:
    path = ROOT / "configs/robot_registry_13.yaml"
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(registry["robots"]) == EXPECTED_ROBOTS
    for name, row in registry["robots"].items():
        source = ROOT / row["source_model"]
        assert source.is_file(), (name, source)
        assert _below(source, OFFICIAL), (name, source)
        assert "scalable_segments" not in row, name


def test_active_model_pipeline_has_no_assets_or_normalization_execution() -> None:
    active_files = (
        "scripts/strict_urdf_model_audit.py",
        "scripts/solve_strict_urdf_task_cache.py",
        "scripts/render_model_assembly_qa.py",
        "scripts/run_twelve_arm_two_single_tasks.py",
        "factory_bimanual/robot_contracts.py",
        "factory_bimanual/scene_builder.py",
    )
    forbidden = (
        "Assets/canonical",
        '"Assets" / "canonical"',
        "apply_uniform_scale",
        "normalization_scale",
        "LEGACY_SOURCE_MODELS",
        "LEGACY_MODELS",
    )
    for relative in active_files:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, (relative, token)


def test_all_runtime_python_is_free_of_legacy_model_sources_and_scalers() -> None:
    roots = ("scripts", "design_optimization", "factory_bimanual")
    forbidden = (
        '"Assets"',
        "'Assets'",
        "Assets/canonical",
        "Assets\\canonical",
        "normalize_design_exact",
        "normalize_chain_to_reach",
        "apply_uniform_scale",
        "uniformly_scale_robot",
        "normalization_scale",
        "scalable_segments",
    )
    for directory in roots:
        for path in (ROOT / directory).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in source, (path.relative_to(ROOT), token)
