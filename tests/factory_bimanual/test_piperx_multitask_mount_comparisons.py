from types import SimpleNamespace

from scripts import render_piperx_multitask_mount_comparisons as renderer


def test_interrupted_cached_panel_is_not_reused(monkeypatch, tmp_path):
    panel = tmp_path / "partial.mp4"
    panel.write_bytes(b"not-an-mp4")
    monkeypatch.setattr(
        renderer, "decode_check_mp4",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken")))

    assert not renderer._cached_panel_is_valid(panel)


def test_valid_30fps_cached_panel_is_reused(monkeypatch, tmp_path):
    panel = tmp_path / "complete.mp4"
    panel.write_bytes(b"encoded")
    monkeypatch.setattr(
        renderer, "decode_check_mp4",
        lambda *args, **kwargs: SimpleNamespace(fps=30.0))

    assert renderer._cached_panel_is_valid(panel)
