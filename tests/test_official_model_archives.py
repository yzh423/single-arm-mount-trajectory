from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def test_all_published_official_model_archives_are_readable():
    archives = sorted((
        ROOT / "third_party" / "official_robot_models").glob("*.zip"))
    assert archives
    for archive in archives:
        with ZipFile(archive) as handle:
            assert handle.testzip() is None, f"corrupt member in {archive.name}"
