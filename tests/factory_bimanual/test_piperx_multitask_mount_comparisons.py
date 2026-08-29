from types import SimpleNamespace

from scripts import render_piperx_multitask_mount_comparisons as renderer


def test_interrupted_cached_panel_is_not_reused(monkeypatch, tmp_path):
    panel = tmp_path / "partial.mp4"
    panel.write_bytes(b"not-an-mp4")
    monkeypatch.setattr(
        renderer, "decode_check_mp4",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken")))

    assert not renderer._cached_panel_is_valid(panel, tmp_path / "summary.json")


def test_valid_30fps_cached_panel_is_reused(monkeypatch, tmp_path):
    panel = tmp_path / "complete.mp4"
    panel.write_bytes(b"encoded")
    summary = tmp_path / "summary.json"
    trajectory = tmp_path / "trajectory.npz"
    scene = tmp_path / "scene.xml"
    trajectory.write_bytes(b"trajectory-v1")
    scene.write_bytes(b"scene-v1")
    summary.write_text(
        '{"artifacts":{"trajectory_npz":{"path":"%s"},'
        '"scene_xml":{"path":"%s"}}}'
        % (trajectory.as_posix(), scene.as_posix()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        renderer, "decode_check_mp4",
        lambda *args, **kwargs: SimpleNamespace(fps=30.0))
    renderer._write_panel_provenance(panel, summary)

    assert renderer._cached_panel_is_valid(panel, summary)

    trajectory.write_bytes(b"trajectory-v2")
    assert not renderer._cached_panel_is_valid(panel, summary)
