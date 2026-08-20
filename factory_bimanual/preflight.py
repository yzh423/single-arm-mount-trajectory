"""Truthful preflight checks; this module never runs an experiment."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .controller_adapter import make_easyik, make_mpc
from .defaults import CONTROLLER_PROFILES_PATH, OUTPUT_ROOT, SELECTED_SPACING_M, TASK_SPECS
from .robot_contracts import ROBOT_CONTRACTS
from .scene_builder import build_same_model_scene
from .source_data import load_factory_task
from .workspace_feasibility import audit_workspace_feasibility


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_spacing_evidence(evidence_root: Path, selected: dict[str, float]):
    """Validate measured selection evidence, not merely configuration keys."""
    required_tasks = {spec.name for spec in TASK_SPECS}
    required_metrics = {"synchronous_coverage", "longest_failure_s",
                        "cross_arm_collision_frames", "table_base_collision_frames",
                        "aggregate_tcp_error"}
    errors = []
    for robot in ROBOT_CONTRACTS:
        path = Path(evidence_root) / f"{robot}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{robot}: missing/unreadable spacing evidence: {exc}")
            continue
        status = payload.get("status")
        if status != "valid_selection":
            errors.append(f"{robot}: spacing status {status!r} is not valid_selection")
        if robot not in selected or not np.isclose(payload.get("selected_spacing_m", np.nan), selected.get(robot, np.nan)):
            errors.append(f"{robot}: selected spacing does not match evidence")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"{robot}: candidates missing")
            continue
        tasks = {row.get("task") for row in candidates if isinstance(row, dict)}
        if tasks != required_tasks:
            errors.append(f"{robot}: evidence must contain exactly both tasks")
        if any(not required_metrics.issubset(row) or "spacing_m" not in row
               for row in candidates if isinstance(row, dict)):
            errors.append(f"{robot}: collision/coverage/error metrics missing")
        coverage = [row.get("synchronous_coverage") for row in candidates if isinstance(row, dict)]
        if not coverage or not any(isinstance(value, (int, float)) and np.isfinite(value) and value > 0
                                   for value in coverage):
            errors.append(f"{robot}: invalid_all_zero coverage")
    return not errors, errors


def run_preflight(root: Path, *, output_path: Path | None = None, executors=None) -> dict:
    root = Path(root).resolve()
    output = Path(output_path or OUTPUT_ROOT / "preflight.json").resolve()
    allowed = root / "reports/factory_bimanual"
    if not _inside(output, allowed):
        raise ValueError("preflight output must stay inside reports/factory_bimanual")
    checks: dict[str, bool] = {}
    blockers: list[str] = []
    tasks = []
    try:
        for spec in TASK_SPECS:
            task = load_factory_task(spec.csv_path, spec.name)
            tasks.append({"name": spec.name, "csv": str(spec.csv_path), "rows": len(task.time_s),
                          "registration": spec.registration(task).matrix.tolist()})
        checks["full_source_rows"] = True
    except Exception as exc:
        checks["full_source_rows"] = False
        blockers.append(f"CSV/full-row validation failed: {exc}")
    checks["assets"] = all(c.source_urdf.is_file() for c in ROBOT_CONTRACTS.values())
    if not checks["assets"]:
        blockers.append("robot assets missing")
    profiles = json.loads(CONTROLLER_PROFILES_PATH.read_text(encoding="utf-8"))["profiles"]
    scene_rows = []
    scenes_ok = controllers_ok = True
    scene_root = allowed / "preflight_scenes"
    for name, contract in ROBOT_CONTRACTS.items():
        try:
            spacing = SELECTED_SPACING_M.get(name, 0.8)  # compile-only provisional geometry
            xml = scene_root / f"{name}.xml"
            manifest = build_same_model_scene(contract, spacing, xml)
            model, data = mujoco.MjModel.from_xml_path(str(xml)), None
            data = mujoco.MjData(model)
            name_map = {side: {"joints": getattr(manifest, f"{side}_joint_names"),
                               "site": f"{side}_tcp", "target": f"{side}_target",
                               "gripper_prefix": f"{side}_"} for side in ("left", "right")}
            velocity = {side: np.asarray(profiles[name]["joint_velocity_rad_s"], float)
                        for side in ("left", "right")}
            make_easyik(model, data, contract, name_map=name_map, velocity_limits=velocity)
            make_mpc(model, data, contract, name_map=name_map, velocity_limits=velocity)
            scene_rows.append({"robot": name, "xml": str(xml), "compile_spacing_m": spacing})
        except Exception as exc:
            scenes_ok = controllers_ok = False
            blockers.append(f"{name} scene/controller validation failed: {exc}")
    checks["scenes_compile"] = scenes_ok
    checks["controllers_construct"] = controllers_ok
    workspace_audits = []
    impossible_rows = []
    if scenes_ok and checks["full_source_rows"]:
        scene_paths = {row["robot"]: Path(row["xml"]) for row in scene_rows}
        for spec in TASK_SPECS:
            task = load_factory_task(spec.csv_path, spec.name)
            audit = audit_workspace_feasibility(
                spec.name,
                {"left": task.left_position_m, "right": task.right_position_m},
                scene_paths,
                sample_count=64,
                require_targets_above_mount=True,
            )
            rows = [row for row in audit["results"] if row["status"] == "impossible"]
            impossible_rows.extend(rows)
            workspace_audits.append({
                "task": spec.name, "impossible": rows,
                "full_report": str((allowed / f"{spec.name}_tabletop_feasibility.json").resolve()),
            })
            Path(workspace_audits[-1]["full_report"]).write_text(
                json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    checks["workspace_feasible"] = not impossible_rows
    for row in impossible_rows:
        blockers.append(
            f"mathematically impossible: {row['task']} {row['robot']} {row['side']} "
            f"({row['proof']['inequality']})"
        )
    spacing_ok, spacing_errors = validate_spacing_evidence(
        allowed / "spacing", SELECTED_SPACING_M)
    checks["spacing_finalized"] = spacing_ok
    if not checks["spacing_finalized"]:
        blockers.extend(spacing_errors)
    checks["executors_available"] = bool(executors and executors.get("ik") and executors.get("mpc"))
    if not checks["executors_available"]:
        blockers.append("IK and MPC executors are not configured")
    checks["output_isolated"] = True
    report = {"schema_version": 1, "ready": all(checks.values()), "checks": checks,
              "blockers": blockers, "tasks": tasks, "robots": scene_rows,
              "workspace_audits": workspace_audits}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
