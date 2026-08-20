"""Production wiring for isolated, resumable factory-bimanual jobs."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import mujoco
import numpy as np

from .artifacts import FrameDiagnostics, RunArtifactWriter
from .branch_equivalence import compare_initial_branches
from .defaults import CONTROLLER_PROFILES_PATH, SELECTED_SPACING_M, TASK_SPECS
from .easyik_runner import EasyIKBudget, EasyIKScene, run_easyik_task
from .mpc_runner import MPCScene, run_mpc_task
from .mujoco_candidate_generator import MuJoCoCandidateGenerator
from .mujoco_collision_adapter import MuJoCoPairedCollisionChecker
from .native_dof_mpc import MPCConfig
from .registration import register_task
from .robot_contracts import ROBOT_CONTRACTS
from .scene_builder import build_same_model_scene
from .source_data import load_factory_task
from .strict_bimanual_ik import BimanualIKConfig, solve_strict_bimanual_path


def _prefix(task, rows):
    count = len(task.time_s) if rows is None else min(int(rows), len(task.time_s))
    values = {}
    for key, value in task.__dict__.items():
        if isinstance(value, np.ndarray) and value.ndim and len(value) == len(task.time_s):
            values[key] = value[:count].copy()
        else:
            values[key] = value
    # Both historical runner spellings are supported without changing source modules.
    values["source_row_indices"] = values["source_row_index"]
    return SimpleNamespace(**values)


class ProductionExecutors:
    def __init__(self, root: Path, *, spacing_by_robot: Mapping[str, float] | None = None,
                 output_root: Path | None = None, easyik_iterations: int = 40,
                 mpc_config: MPCConfig | None = None):
        self.root = Path(root).resolve()
        self.output_root = Path(output_root or self.root / "reports/factory_bimanual").resolve()
        try:
            relative = self.output_root.relative_to(self.root)
        except ValueError:
            raise ValueError("executor output must be reports/factory_bimanual inside project root")
        if tuple(part.lower() for part in relative.parts[-2:]) != ("reports", "factory_bimanual"):
            raise ValueError("executor output must be reports/factory_bimanual inside project root")
        self.spacing = dict(SELECTED_SPACING_M if spacing_by_robot is None else spacing_by_robot)
        self.writer = RunArtifactWriter(self.output_root, project_root=self.root)
        self.easyik_budget = EasyIKBudget(easyik_iterations)
        self.mpc_config = mpc_config or MPCConfig()
        self.profiles = json.loads(CONTROLLER_PROFILES_PATH.read_text(encoding="utf-8"))["profiles"]

    def mapping(self):
        return {"ik": self.ik, "mpc": self.mpc}

    def load_task(self, name: str, prefix_rows: int | None = None):
        try:
            spec = next(item for item in TASK_SPECS if item.name == name)
        except StopIteration as exc:
            raise ValueError(f"unknown task: {name}") from exc
        source = load_factory_task(spec.csv_path, spec.name)
        return _prefix(register_task(source, spec.registration(source)), prefix_rows)

    def _scene(self, robot):
        if robot not in self.spacing:
            raise RuntimeError(f"selected spacing is missing for {robot}")
        contract = ROBOT_CONTRACTS[robot]
        xml = self.output_root / "scenes" / f"{robot}.xml"
        manifest = build_same_model_scene(contract, float(self.spacing[robot]), xml)
        model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
        names = {side: {"joints": getattr(manifest, f"{side}_joint_names"),
                        "site": f"{side}_tcp", "target": f"{side}_target",
                        "gripper_prefix": f"{side}_"} for side in ("left", "right")}
        limits = {side: np.asarray(self.profiles[robot]["joint_velocity_rad_s"], float)
                  for side in ("left", "right")}
        collision = MuJoCoPairedCollisionChecker(model, data, names)
        return contract, model, data, names, limits, collision

    @staticmethod
    def _frames(task, result):
        failures = getattr(result, "failures", [None] * len(task.time_s))
        collisions = getattr(result, "collision_classes", [()] * len(task.time_s))
        rollback = getattr(result, "rollback", np.zeros(len(task.time_s), bool))
        return [FrameDiagnostics(int(task.source_row_indices[i]), float(task.time_s[i]),
            str(failures[i] or ""), str(failures[i] or ""), bool(collisions[i]),
            bool(rollback[i]), str(task.source_path)) for i in range(len(task.time_s))]

    def _write_ik(self, job, task, result):
        arrays = {name: np.asarray(getattr(result, name)) for name in (
            "left_q", "right_q", "left_actual_tcp", "right_actual_tcp",
            "left_position_error_m", "right_position_error_m",
            "left_orientation_error_rad", "right_orientation_error_rad", "success")}
        row_zero_success = bool(len(result.success) and result.success[0]
                                and int(task.source_row_indices[0]) == 0)
        summary = {"initializer_valid": row_zero_success, "successful_rows": int(result.success.sum()),
                   "total_rows": len(result.success), "source_rows_preserved": True}
        if row_zero_success:
            summary["initializer_row"] = 0
            summary["initializer_source_index"] = 0
        paths = self.writer.write_run(f"{job.robot}/{job.task}/{job.mode}",
                                      self._frames(task, result), arrays, summary)
        return summary | {"artifact_json": str(paths.json_path), "artifact_npz": str(paths.npz_path)}

    @staticmethod
    def _common_initialization_row(first, second):
        """Return the shared row-0 initializer, rejecting stale legacy metadata."""
        rows = []
        for artifact in (first, second):
            summary = artifact["summary"]
            if (not summary.get("initializer_valid")
                    or int(summary.get("initializer_row", -1)) != 0
                    or int(summary.get("initializer_source_index", -1)) != 0):
                raise RuntimeError("both IK initializers must be successful at source row 0")
            rows.append(int(summary["initializer_row"]))
        if rows[0] != rows[1]:
            raise RuntimeError("IK branches must compare the same source row 0")
        return rows[0]

    def ik(self, job):
        task = self.load_task(job.task, job.short_prefix_rows)
        contract, model, data, names, limits, collision = self._scene(job.robot)
        if job.mode == "strict_a":
            generator = MuJoCoCandidateGenerator(model, data, contract, name_map=names)
            result = solve_strict_bimanual_path(model, contract, task,
                BimanualIKConfig(generator, collision, beam_width=64, max_velocity_rad_s=limits["left"]))
        elif job.mode == "easyik":
            result = run_easyik_task(EasyIKScene(model, data, contract, names,
                                     velocity_limits=limits, collision_checker=collision),
                                     task, self.easyik_budget)
        else:
            raise ValueError(f"unknown IK mode: {job.mode}")
        output = self._write_ik(job, task, result)
        if job.mode == "easyik":
            a = self._load_ik(job.robot, job.task, "strict_a")
            if a is not None and output["initializer_valid"]:
                easy_artifact = {"summary": output}
                row = self._common_initialization_row(a, easy_artifact)
                decision = compare_initial_branches((a["arrays"]["left_q"][row], a["arrays"]["right_q"][row]),
                    (result.left_q[row], result.right_q[row]))
                output.update(left_delta_deg=float(np.rad2deg(decision.left_max_delta_rad)),
                              right_delta_deg=float(np.rad2deg(decision.right_max_delta_rad)),
                              comparison_row=row, comparison_source_index=0)
                artifact_path = Path(output["artifact_json"])
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                artifact["summary"].update({key: output[key] for key in (
                    "left_delta_deg", "right_delta_deg", "comparison_row",
                    "comparison_source_index")})
                artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return output

    def _load_ik(self, robot, task, mode):
        base = self.output_root / "runs" / robot / task / mode
        if not (base / "summary.json").is_file() or not (base / "arrays.npz").is_file(): return None
        meta = json.loads((base / "summary.json").read_text(encoding="utf-8"))["summary"]
        with np.load(base / "arrays.npz") as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        return {"summary": meta, "arrays": arrays}

    def mpc(self, job):
        source_mode = "easyik" if job.mode == "mpc_easyik" else "strict_a"
        saved = self._load_ik(job.robot, job.task, source_mode)
        if saved is None or not saved["summary"].get("initializer_valid"):
            raise RuntimeError(f"valid {source_mode} initializer artifact is required")
        summary = saved["summary"]
        if (int(summary.get("initializer_row", -1)) != 0
                or int(summary.get("initializer_source_index", -1)) != 0):
            raise RuntimeError(f"valid {source_mode} initializer at source row 0 is required")
        row = 0
        initial = (saved["arrays"]["left_q"][row], saved["arrays"]["right_q"][row])
        task = self.load_task(job.task, job.short_prefix_rows)
        contract, model, data, names, limits, collision = self._scene(job.robot)
        result = run_mpc_task(MPCScene(model, data, contract, names, limits, collision),
                              task, initial, self.mpc_config, job.mode)
        arrays = {name: np.asarray(getattr(result, name)) for name in (
            "left_q", "right_q", "left_actual_tcp", "right_actual_tcp",
            "left_target_lag_m", "right_target_lag_m", "rollback",
            "control_steps_per_interval", "step_wall_time_s")}
        summary = {"initializer_valid": True, "initializer_source": source_mode,
                   "initializer_row": 0, "initializer_source_index": 0,
                   "total_rows": len(task.time_s), "rollback_rows": int(result.rollback.sum())}
        paths = self.writer.write_run(f"{job.robot}/{job.task}/{job.mode}",
                                      self._frames(task, result), arrays, summary)
        return summary | {"artifact_json": str(paths.json_path), "artifact_npz": str(paths.npz_path)}


__all__ = ["ProductionExecutors"]
