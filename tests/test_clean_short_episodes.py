import csv
import json
from pathlib import Path

from scripts.clean_short_episodes import (
    clean,
    inspect_episode,
    is_short,
    quarantine_path,
    scan_episodes,
)


def write_episode(path: Path, times=(0.0, 1.0, 4.9)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "right_tcp_pos_x", "right_tcp_pos_y", "right_tcp_pos_z"])
        for value in times:
            writer.writerow([value, value, value + 1, value + 2])
    return path


def test_inspect_episode_uses_first_and_last_t(tmp_path):
    info = inspect_episode(write_episode(tmp_path / "task" / "episode.csv", (2.0, 3.0, 7.25)))
    assert info.rows == 3
    assert info.duration_s == 5.25
    assert info.task == "task"


def test_short_threshold_is_strict(tmp_path):
    below = inspect_episode(write_episode(tmp_path / "a" / "below.csv", (0.0, 4.999)))
    equal = inspect_episode(write_episode(tmp_path / "a" / "equal.csv", (0.0, 5.0)))
    assert is_short(below, 5.0)
    assert not is_short(equal, 5.0)


def test_scan_excludes_quarantine(tmp_path):
    root = tmp_path / "8-11"
    active = write_episode(root / "Fold_Box" / "active.csv")
    write_episode(root / "_rejected_short" / "Fold_Box" / "rejected.csv")
    assert [item.path for item in scan_episodes(root)] == [active]


def test_quarantine_path_never_overwrites(tmp_path):
    root = tmp_path / "8-11"
    info = inspect_episode(write_episode(root / "Fold_Box" / "episode.csv"))
    first = root / "_rejected_short" / "Fold_Box" / "episode.csv"
    write_episode(first)
    assert quarantine_path(info, root).name == "episode_2.csv"


def test_clean_dry_run_and_apply_write_equivalent_manifests(tmp_path):
    root = tmp_path / "8-11"
    short = write_episode(root / "Fold_Box" / "short.csv", (0.0, 4.0))
    long = write_episode(root / "Fold_Box" / "long.csv", (0.0, 5.0))
    dry_prefix = root / "dry_manifest"

    dry_records = clean(root, threshold_s=5.0, apply=False, manifest_prefix=dry_prefix)

    assert short.exists() and long.exists()
    assert [record["action"] for record in dry_records] == ["would_quarantine"]
    with dry_prefix.with_suffix(".json").open(encoding="utf-8") as handle:
        dry_json = json.load(handle)
    with dry_prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        dry_csv = list(csv.DictReader(handle))
    assert len(dry_json) == len(dry_csv) == 1
    assert dry_json[0]["source"].endswith("Fold_Box/short.csv")

    apply_prefix = root / "apply_manifest"
    applied = clean(root, threshold_s=5.0, apply=True, manifest_prefix=apply_prefix)

    assert [record["action"] for record in applied] == ["quarantined"]
    destination = Path(applied[0]["destination"])
    assert not short.exists()
    assert destination.exists()
    assert long.exists()
    with apply_prefix.with_suffix(".json").open(encoding="utf-8") as handle:
        apply_json = json.load(handle)
    assert apply_json[0]["duration_s"] == dry_json[0]["duration_s"]
    assert apply_json[0]["threshold_s"] == 5.0

