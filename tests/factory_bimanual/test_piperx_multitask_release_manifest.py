import pytest
from types import SimpleNamespace

from scripts import build_piperx_multitask_release_manifest as release


def test_release_file_record_detects_content_drift(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    record = release._file_record(artifact)

    assert release._validate_file_record(record) == artifact.resolve()

    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="release artifact drift"):
        release._validate_file_record(record)


def test_composite_provenance_must_bind_the_published_video():
    check = SimpleNamespace(
        frame_count=3, fps=30.0, duration_s=0.1, width=1280, height=720)
    record = {"path": "video.mp4", "sha256": "a" * 64}
    provenance = {
        "schema": "piperx-four-mount-composite-v1",
        "trajectory": "8-11/Fold_Box/161044",
        "output_sha256": "b" * 64,
        "frame_count": 3,
        "fps": 30.0,
        "duration_s": 0.1,
        "width": 1280,
        "height": 720,
        "panels": {mode: {} for mode in release.bundle.STUDY_MODES},
    }

    with pytest.raises(ValueError, match="provenance drift"):
        release._validate_video_provenance(provenance, record, check)


def test_embedded_panel_evidence_must_match_published_shards():
    expected = {
        mode: {
            "schema": "piperx-four-mount-panel-v2-content-addressed",
            "summary_sha256": mode * 4,
            "trajectory_sha256": mode * 4,
            "scene_sha256": mode * 4,
        }
        for mode in release.bundle.STUDY_MODES
    }
    provenance = {
        "panels": {
            mode: {"evidence": dict(evidence)}
            for mode, evidence in expected.items()
        }
    }
    provenance["panels"]["baseline"]["evidence"]["scene_sha256"] = "wrong"

    with pytest.raises(ValueError, match="panel evidence drift"):
        release._validate_published_panel_evidence(provenance, expected)
