import hashlib
from pathlib import Path

from factory_bimanual.factory_task_catalog import build_factory_task_catalog


def _csv(path: Path, rows: int, payload="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("a,b\n" + "\n".join(f"{i},{payload}" for i in range(rows)) + "\n",
                    encoding="utf-8")


def test_catalog_excludes_metadata_and_rejected_and_selects_longest(tmp_path):
    _csv(tmp_path / "day" / "TaskA" / "short.csv", 2)
    winner = tmp_path / "day" / "TaskA" / "long.csv"
    _csv(winner, 4)
    _csv(tmp_path / "day" / "_rejected_short" / "TaskB" / "bad.csv", 9)
    _csv(tmp_path / "day" / "cleaning_manifest_20260811.csv", 8)

    catalog = build_factory_task_catalog(tmp_path)

    assert [item.task_name for item in catalog.representatives] == ["TaskA"]
    selected = catalog.representatives[0]
    assert selected.source_rows == 4
    assert selected.csv_path == winner.resolve()
    assert selected.sha256 == hashlib.sha256(winner.read_bytes()).hexdigest()
    assert len(catalog.excluded) == 2


def test_catalog_breaks_equal_row_tie_by_normalized_path(tmp_path):
    first = tmp_path / "a" / "TaskA" / "a.csv"
    second = tmp_path / "b" / "TaskA" / "b.csv"
    _csv(second, 3); _csv(first, 3)
    selected = build_factory_task_catalog(tmp_path).representatives[0]
    assert selected.csv_path == first.resolve()


def test_real_factory_catalog_has_eleven_representatives():
    root = Path(__file__).resolve().parents[2] / "data" / "factory"
    catalog = build_factory_task_catalog(root)
    assert len(catalog.representatives) == 11
    assert len(catalog.episodes) == 27
    assert {item.task_name for item in catalog.representatives} == {
        "AluminumFoilPouch_BoxPackaging", "Bag_BoxPacking", "Fold_Box",
        "InsertIntoBottle", "InsertSmallPackageIntoMachine", "PackIntoBox",
        "PourRawMaterial", "PutIntoBox", "Screw_Cap", "Seal_Bag",
        "WaterSoluble_AluminumPouch",
    }
    assert next(item for item in catalog.representatives
                if item.task_name == "Seal_Bag").source_rows == 2439
