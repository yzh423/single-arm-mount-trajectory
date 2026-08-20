from pathlib import Path


def test_assembly_qa_loads_native_geometry_without_rescaling():
    source = (Path(__file__).parents[1] / "scripts" / "render_model_assembly_qa.py").read_text()
    assert "uniformly_scale_robot" not in source
    assert "load_native_spec" in source
