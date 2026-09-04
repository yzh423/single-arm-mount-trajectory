import argparse
import json
from pathlib import Path

from scripts import render_model_assembly_qa


def test_assembly_qa_loads_native_geometry_without_rescaling():
    source = (Path(__file__).parents[1] / "scripts" / "render_model_assembly_qa.py").read_text()
    assert "uniformly_scale_robot" not in source
    assert "load_native_spec" in source


def test_assembly_qa_writes_json_report_when_robot_list_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(robots=[], output=tmp_path),
    )

    render_model_assembly_qa.main()

    assert json.loads((tmp_path / "qa.json").read_text(encoding="utf-8")) == {
        "robots": {},
        "status": "pass",
    }
