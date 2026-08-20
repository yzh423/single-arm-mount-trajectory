"""Validate orientation shortlists on full source time and render winners."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.render_factory_dual_piperx_fixed_time import (
    render_saved_run, run_task,
)
from scripts.run_piperx_seal_bag_orientation_comparison import OUTPUT, RESULTS, _atomic_json
from factory_bimanual.orientation_mount_search import mount_fingerprint


RUNS = OUTPUT / "runs"
CAMERA = (180.0, -18.0, 2.35)


def select_collision_free_attempt(attempts):
    """Select by tracking quality, but never trade away physical safety."""
    safe = [item for item in attempts
            if int(item["summary"]["collision_frame_count"]) == 0]
    if not safe:
        raise RuntimeError(
            "no collision-free finalist; expand mount spacing/search bounds")
    return max(safe, key=lambda item: float(
        item["summary"]["synchronous_strict_coverage"]))


def attempt_stem(root, mode, index, mount):
    return Path(root) / (
        f"{mode}_candidate_{index}_{mount_fingerprint(mount)[:12]}")


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    selected = {}
    for mode in ("upright_table", "horizontal_wall", "inverted"):
        search = json.loads((RESULTS / f"{mode}.json").read_text(encoding="utf-8"))
        accepted = None
        attempts = []
        for index, mount in enumerate(search["execution_shortlist"], 1):
            stem = attempt_stem(RUNS, mode, index, mount)
            summary_path = stem.with_suffix(".summary.json")
            trajectory = stem.with_suffix(".trajectory.npz")
            if summary_path.exists() and trajectory.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                solver = summary.get("solver", {})
                if (solver.get("method"), solver.get("profile")) != (
                        "xarm6-style", "comparison"):
                    summary = run_task(
                        "seal_bag", mount, output=stem.with_suffix(".mp4"),
                        render_video=False, solver_method="xarm6-style",
                        camera_override=CAMERA, solver_profile="comparison")
            else:
                summary = run_task(
                    "seal_bag", mount, output=stem.with_suffix(".mp4"),
                    render_video=False, solver_method="xarm6-style",
                    camera_override=CAMERA, solver_profile="comparison")
            attempts.append({"candidate": index, "mount": mount,
                             "summary": summary,
                             "trajectory": str(trajectory.resolve())})
            if int(summary["collision_frame_count"]) == 0:
                accepted = attempts[-1]; break
        accepted = select_collision_free_attempt(attempts)
        accepted["safety_status"] = "valid_collision_free"
        video = OUTPUT / "videos" / f"{mode}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        render_saved_run("seal_bag", accepted["mount"],
                         accepted["trajectory"], video,
                         camera_override=CAMERA)
        accepted["video"] = str(video.resolve())
        accepted["video_summary"] = str(video.with_suffix(".summary.json").resolve())
        selected[mode] = {"accepted": accepted, "attempts": attempts}
        _atomic_json(RESULTS / "execution_manifest.json", selected)
    _atomic_json(RESULTS / "execution_manifest.json",
                 {"status": "complete", "camera": CAMERA, "modes": selected})


if __name__ == "__main__":
    main()
