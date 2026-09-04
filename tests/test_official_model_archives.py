from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "third_party" / "official_robot_models"


def test_all_published_official_model_archives_are_readable():
    archives = sorted(MODEL_ROOT.glob("*.zip"))
    assert archives
    for archive in archives:
        with ZipFile(archive) as handle:
            assert handle.testzip() is None, f"corrupt member in {archive.name}"


def test_failed_partial_download_cache_is_not_published():
    assert not (MODEL_ROOT / "incomplete").exists()
