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
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
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
        % (trajectory.name, scene.name),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        renderer, "decode_check_mp4",
        lambda *args, **kwargs: SimpleNamespace(fps=30.0))
    renderer._write_panel_provenance(panel, summary)

    assert renderer._cached_panel_is_valid(panel, summary)

    trajectory.write_bytes(b"trajectory-v2")
    assert not renderer._cached_panel_is_valid(panel, summary)


def test_composite_provenance_binds_output_and_all_panels(tmp_path):
    output = tmp_path / "comparison.mp4"
    output.write_bytes(b"comparison")
    panels = {}
    for mode in renderer.STUDY_PANEL_ORDER:
        panel = tmp_path / f"{mode}.mp4"
        panel.write_bytes(mode.encode("utf-8"))
        panel.with_suffix(".provenance.json").write_text(
            '{"schema":"panel","summary_sha256":"%s"}' % mode,
            encoding="utf-8")
        panels[mode] = panel

    sidecar = renderer._write_composite_provenance(
        output, panels, "task/example",
        SimpleNamespace(frame_count=42, fps=30.0, width=1280, height=720))

    import json
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["output_sha256"] == renderer._sha256(output)
    assert payload["trajectory"] == "task/example"
    assert payload["frame_count"] == 42
    assert set(payload["panels"]) == set(renderer.STUDY_PANEL_ORDER)
    assert payload["panels"]["baseline"]["evidence"] == {
        "schema": "panel", "summary_sha256": "baseline"}
