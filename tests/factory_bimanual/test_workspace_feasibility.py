import json
from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.source_data import load_factory_task
from factory_bimanual.workspace_feasibility import (
    audit_workspace_feasibility,
    trajectory_geometry,
)


ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "screw_cap": ROOT / "data/factory/8-11/Screw_Cap/handheld_20260811_162854.csv",
    "pour_raw_material": ROOT / "data/factory/8-12/PourRawMaterial/handheld_20260812_111542.csv",
}


def test_geometry_is_rigid_transform_invariant_and_encloses_every_point():
    points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    a = trajectory_geometry(points)
    b = trajectory_geometry(points @ rotation.T + [9., -3., 4.])

    assert b.diameter_m == pytest.approx(a.diameter_m)
    assert b.enclosing_radius_m == pytest.approx(a.enclosing_radius_m)
    assert np.linalg.norm(points - a.enclosing_center_m, axis=1).max() <= a.enclosing_radius_m + 1e-12


def test_impossibility_has_machine_checkable_diameter_proof(tmp_path: Path):
    scene = tmp_path / "xarm6.xml"
    build_same_model_scene(ROBOT_CONTRACTS["xarm6"], 0.8, scene)
    points = np.array([[0., 0., 0.], [100., 0., 0.]])
    report = audit_workspace_feasibility("synthetic", {"left": points}, {"xarm6": scene}, sample_count=32)
    row = report["results"][0]

    assert row["status"] == "impossible"
    assert row["proof"]["inequality"] == "trajectory_diameter_m > 2 * reachable_radius_upper_bound_m"
    assert row["trajectory_diameter_m"] > 2 * row["reachable_radius_upper_bound_m"]
    json.dumps(report)


def test_reach_bound_excludes_world_to_mount_translation(tmp_path: Path):
    contract = ROBOT_CONTRACTS["xarm6"]
    low, high = tmp_path / "low.xml", tmp_path / "high.xml"
    build_same_model_scene(contract, 0.8, low, table_height_m=.75)
    build_same_model_scene(contract, 0.8, high, table_height_m=9.0)
    points = np.array([[0., 0., 0.], [3., 0., 0.]])
    rows = [audit_workspace_feasibility(
        "mount_invariant", {"left": points}, {"xarm6": scene}, sample_count=32
    )["results"][0] for scene in (low, high)]
    assert rows[1]["reachable_radius_upper_bound_m"] == pytest.approx(
        rows[0]["reachable_radius_upper_bound_m"])
    assert all(row["status"] == "impossible" for row in rows)


def test_above_table_vertical_span_larger_than_reach_is_impossible(tmp_path: Path):
    scene = tmp_path / "xarm6.xml"
    build_same_model_scene(ROBOT_CONTRACTS["xarm6"], 0.8, scene)
    probe = audit_workspace_feasibility(
        "probe", {"right": np.zeros((2, 3))}, {"xarm6": scene},
        sample_count=32,
    )["results"][0]
    span = probe["reachable_radius_upper_bound_m"] + .1
    points = np.array([[0., 0., 0.], [0., 0., span]])
    row = audit_workspace_feasibility(
        "vertical", {"right": points}, {"xarm6": scene}, sample_count=32,
        require_targets_above_mount=True,
    )["results"][0]
    assert row["status"] == "impossible"
    assert row["proof"]["inequality"] == "trajectory_z_span_m > reachable_radius_upper_bound_m"


@pytest.mark.parametrize("task_name,csv_path", TASKS.items())
def test_complete_exact_csvs_and_all_supported_scenes_are_audited(
        tmp_path: Path, task_name: str, csv_path: Path):
    task = load_factory_task(csv_path, task_name)
    scenes = {}
    for robot, contract in ROBOT_CONTRACTS.items():
        scene = tmp_path / f"{robot}.xml"
        build_same_model_scene(contract, 0.8, scene)
        scenes[robot] = scene

    report = audit_workspace_feasibility(
        task.name,
        {"left": task.left_position_m, "right": task.right_position_m},
        scenes,
        sample_count=64,
    )

    assert report["constraints"] == {
        "full_trajectory": True, "crop": False, "scale": False,
        "shared_rigid_transform": True,
    }
    assert len(report["results"]) == 2 * len(ROBOT_CONTRACTS)
    assert {r["robot"] for r in report["results"]} == set(ROBOT_CONTRACTS)
    assert {r["side"] for r in report["results"]} == {"left", "right"}
    assert all(r["trajectory_point_count"] == len(task.time_s) for r in report["results"])
    assert all(r["status"] in {"feasible", "impossible", "unknown"} for r in report["results"])
    assert all(r["reachable_radius_upper_bound_m"] >= r["empirical_max_radius_m"] for r in report["results"])
    json.dumps(report, allow_nan=False)
