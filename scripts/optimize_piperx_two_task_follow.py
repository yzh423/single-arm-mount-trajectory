"""Deterministic probe-and-refine mount search for strict PiperX following."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from factory_bimanual.complete_follow import CompleteFollowRunner
from factory_bimanual.piperx_recommended import (
    WorldMount,
    load_recommended_config,
    world_mount_for_family,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.task_family import TaskFamily
from factory_bimanual.tool_frame_calibration import CalibrationArtifact
from scripts.render_factory_dual_piperx_fixed_time import (
    CALIBRATION_PATH,
    TABLE_HEIGHT_M,
)
from scripts.run_piperx_recommended_v31 import (
    _registered_source,
    _source_path,
    condition_complete_follow_targets,
    resample_task_60hz,
    scene_mount_kwargs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".tmp/piperx_two_task_mount_search"
LOG_SCHEMA = "piperx-two-task-search-v1"


@dataclass(frozen=True)
class MountCandidate:
    name: str
    mode: str
    left_xyz_m: tuple[float, float, float]
    right_xyz_m: tuple[float, float, float]
    left_yaw_deg: float
    right_yaw_deg: float
    origin: str

    @property
    def key(self) -> str:
        payload = asdict(self).copy()
        payload.pop("name")
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def as_world_mount(self, base: WorldMount) -> WorldMount:
        return WorldMount(
            family=base.family,
            morphology=base.morphology,
            mode=self.mode,
            left_xyz_m=np.asarray(self.left_xyz_m, dtype=float),
            right_xyz_m=np.asarray(self.right_xyz_m, dtype=float),
            shared_base_z_m=float(self.left_xyz_m[2]),
            source_take=base.source_take,
            left_yaw_deg=float(self.left_yaw_deg),
            right_yaw_deg=float(self.right_yaw_deg),
            coordinate_domain="registered_world",
            selection_method=f"deterministic search: {self.origin}",
        )


def _replace(candidate, *, name, origin, left=None, right=None,
             left_yaw=None, right_yaw=None):
    return MountCandidate(
        name=name,
        mode=candidate.mode,
        left_xyz_m=tuple(candidate.left_xyz_m if left is None else left),
        right_xyz_m=tuple(candidate.right_xyz_m if right is None else right),
        left_yaw_deg=(
            candidate.left_yaw_deg if left_yaw is None else float(left_yaw)),
        right_yaw_deg=(
            candidate.right_yaw_deg if right_yaw is None else float(right_yaw)),
        origin=origin,
    )


def local_mount_candidates(
        incumbent: MountCandidate, *, xy_step_m: float, z_step_m: float,
        yaw_step_deg: float, round_index: int):
    """Return deterministic one-coordinate neighbours around an incumbent."""

    origin = f"local_round_{int(round_index)}"
    candidates = []
    for side in ("left", "right"):
        for axis, axis_name in ((0, "x"), (1, "y")):
            for direction in (-1.0, 1.0):
                values = list(getattr(incumbent, f"{side}_xyz_m"))
                values[axis] = round(values[axis] + direction * xy_step_m, 9)
                kwargs = {side: values}
                candidates.append(_replace(
                    incumbent,
                    name=(f"r{round_index}_{side}_{axis_name}_"
                          f"{direction:+.0f}"),
                    origin=origin,
                    **kwargs,
                ))
    for direction in (-1.0, 1.0):
        z = round(incumbent.left_xyz_m[2] + direction * z_step_m, 9)
        left = (*incumbent.left_xyz_m[:2], z)
        right = (*incumbent.right_xyz_m[:2], z)
        candidates.append(_replace(
            incumbent, name=f"r{round_index}_shared_z_{direction:+.0f}",
            origin=origin, left=left, right=right))
    for side in ("left", "right"):
        for direction in (-1.0, 1.0):
            value = getattr(incumbent, f"{side}_yaw_deg") + (
                direction * yaw_step_deg)
            candidates.append(_replace(
                incumbent,
                name=f"r{round_index}_{side}_yaw_{direction:+.0f}",
                origin=origin,
                **{f"{side}_yaw": value},
            ))
    unique = {candidate.key: candidate for candidate in candidates}
    return [unique[key] for key in sorted(unique)]


def _read_log(path: Path):
    path = Path(path)
    if not path.is_file():
        return {"schema": LOG_SCHEMA, "records": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != LOG_SCHEMA:
        raise ValueError("unexpected two-task search log schema")
    return payload


def append_experiment_record(path: Path, record):
    path = Path(path)
    payload = _read_log(path)
    payload["records"].append(dict(record))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pending_candidates(
        candidates, log_path: Path, *, resume: bool, probe_signature=None):
    if not resume:
        return list(candidates)
    completed = {
        str(record["candidate_key"])
        for record in _read_log(log_path)["records"]
        if record.get("status") in {"probe_complete", "full_complete"}
        and (probe_signature is None
             or record.get("probe_signature") == probe_signature)
    }
    return [candidate for candidate in candidates
            if candidate.key not in completed]


def _candidate_from_world_mount(name, mount, origin):
    return MountCandidate(
        name=name,
        mode=mount.mode,
        left_xyz_m=tuple(float(value) for value in mount.left_xyz_m),
        right_xyz_m=tuple(float(value) for value in mount.right_xyz_m),
        left_yaw_deg=mount.left_yaw_deg,
        right_yaw_deg=mount.right_yaw_deg,
        origin=origin,
    )


def seal_bag_seed_candidates(configured_mount):
    """Known report candidates plus the registered PDF recommendation."""

    return [
        _candidate_from_world_mount(
            "pdf_horizontal_forward", configured_mount, "pdf_v31"),
        MountCandidate(
            "historical_upright", "upright_table",
            (-0.37, 0.18, 0.85), (-0.20, -0.47, 0.85),
            0.0, 30.0, "historical_fixed_time"),
        MountCandidate(
            "v4_upright_best", "upright_table",
            (-0.35, 0.25, 0.81), (-0.30, -0.45, 0.81),
            15.0, 15.0, "historical_v4"),
        MountCandidate(
            "orientation_shortlist_upright", "upright_table",
            (-0.012438139, 0.468367208, 0.81),
            (-0.284546709, -0.345352057, 0.81),
            -60.0, 15.0, "three_orientation_shortlist"),
    ]


def _load_context(family_text, source_take, rate_hz):
    config = load_recommended_config()
    family = TaskFamily.parse(family_text)
    source, registration = _registered_source(
        _source_path(family, source_take), family)
    task = resample_task_60hz(source, rate_hz=rate_hz)
    configured = world_mount_for_family(
        config, family, registration.rotation_world_from_vr,
        registration.translation_world_m)
    calibration = CalibrationArtifact.read(CALIBRATION_PATH)
    spec = config.mounts[family.key]
    offsets = {
        "left": (spec.left_tool_offset_quaternion_wxyz or
                 calibration.left_offset_quaternion_wxyz),
        "right": (spec.right_tool_offset_quaternion_wxyz or
                  calibration.right_offset_quaternion_wxyz),
    }
    task, mapped, _ = condition_complete_follow_targets(
        task, offsets, apply_conditioning=False)
    return config, configured, task, mapped


def probe_candidate(candidate, *, base_mount, task, mapped, config,
                    output_dir, rows_by_side, maximum_candidates):
    mount = candidate.as_world_mount(base_mount)
    scene = Path(output_dir) / "scenes" / f"{candidate.key}.scene.xml"
    kwargs, orientation = scene_mount_kwargs(
        mount, task, table_height_m=TABLE_HEIGHT_M)
    build_same_model_scene(
        ROBOT_CONTRACTS["piperx"], mount.base_distance_m, scene, **kwargs)
    model = mujoco.MjModel.from_xml_path(str(scene))
    runner = CompleteFollowRunner(
        model, task, mapped, config,
        maximum_candidates_per_side=maximum_candidates)
    details = []
    for side in ("left", "right"):
        for row in rows_by_side[side]:
            runner.generator.reset()
            candidates = runner._global(side, int(row))
            details.append({
                "side": side,
                "row": int(row),
                "candidate_count": len(candidates),
                "strict": bool(candidates),
            })
    probe_signature = hashlib.sha256(json.dumps(
        {side: [int(row) for row in rows_by_side[side]]
         for side in ("left", "right")},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()[:16]
    return {
        "candidate_key": candidate.key,
        "candidate": asdict(candidate),
        "status": "probe_complete",
        "probe_signature": probe_signature,
        "strict_hits": sum(item["strict"] for item in details),
        "strict_total": len(details),
        "candidate_count_sum": sum(item["candidate_count"] for item in details),
        "details": details,
        "orientation": orientation,
        "scene": str(scene.resolve()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="8-11/Seal_Bag")
    parser.add_argument("--source-take", default="161504")
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--incumbent-key",
        help="candidate key already present in the experiment log",
    )
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--maximum-candidates", type=int, default=8)
    parser.add_argument("--probe-left-rows", type=int, nargs="*", default=(0,))
    parser.add_argument("--probe-right-rows", type=int, nargs="*", default=(0, 735))
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args(argv)


def run(options):
    config, base_mount, task, mapped = _load_context(
        options.family, options.source_take, options.rate_hz)
    if TaskFamily.parse(options.family).task.lower() != "seal_bag":
        seeds = [_candidate_from_world_mount(
            "configured_mount", base_mount, "configured")]
    else:
        seeds = seal_bag_seed_candidates(base_mount)
    log_path = Path(options.output_dir) / "experiment.json"
    if options.incumbent_key:
        matching = [
            record for record in _read_log(log_path)["records"]
            if record.get("candidate_key") == options.incumbent_key
        ]
        if not matching:
            raise ValueError("incumbent key is absent from the experiment log")
        incumbent = MountCandidate(**matching[-1]["candidate"])
        candidates = [incumbent]
    else:
        incumbent = seeds[2] if len(seeds) > 2 else seeds[0]
        candidates = list(seeds)
    for round_index in range(1, options.max_rounds + 1):
        scale = 0.5 ** (round_index - 1)
        candidates.extend(local_mount_candidates(
            incumbent, xy_step_m=0.025 * scale,
            z_step_m=0.02 * scale, yaw_step_deg=7.5 * scale,
            round_index=round_index))
    rows = {
        "left": tuple(options.probe_left_rows),
        "right": tuple(options.probe_right_rows),
    }
    probe_signature = hashlib.sha256(json.dumps(
        {side: list(rows[side]) for side in ("left", "right")},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()[:16]
    candidates = pending_candidates(
        candidates, log_path, resume=options.resume,
        probe_signature=probe_signature)
    records = []
    for candidate in candidates:
        record = probe_candidate(
            candidate, base_mount=base_mount, task=task, mapped=mapped,
            config=config, output_dir=options.output_dir,
            rows_by_side=rows,
            maximum_candidates=options.maximum_candidates)
        append_experiment_record(log_path, record)
        records.append(record)
        print(json.dumps({
            "name": candidate.name,
            "key": candidate.key,
            "strict": f"{record['strict_hits']}/{record['strict_total']}",
            "candidate_count_sum": record["candidate_count_sum"],
        }), flush=True)
    all_records = [
        record for record in _read_log(log_path)["records"]
        if record.get("probe_signature") == probe_signature
    ]
    best = max(all_records, key=lambda item: (
        int(item.get("strict_hits", -1)) / max(
            1, int(item.get("strict_total", 0))),
        int(item.get("candidate_count_sum", -1)) / max(
            1, int(item.get("strict_total", 0))),
        str(item.get("candidate_key", "")),
    ))
    return {"log": log_path, "best": best, "new_records": len(records)}


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
