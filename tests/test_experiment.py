from pathlib import Path

import torch

from design_optimization.experiment import ExperimentConfig, load_checkpoint, save_checkpoint_atomic


def test_config_roundtrip_and_atomic_checkpoint(tmp_path: Path):
    config = ExperimentConfig(topology="kinova")
    yaml_path = tmp_path / "run.yaml"; config.write(yaml_path)
    loaded = ExperimentConfig.from_yaml(yaml_path)
    assert loaded.topology == "kinova"
    assert loaded.constraints.kinova_vendor_fraction == 0.45
    checkpoint = tmp_path / "state.pt"
    save_checkpoint_atomic(checkpoint, {"generation": 7, "mean": torch.arange(3)})
    state = load_checkpoint(checkpoint)
    assert state["generation"] == 7
    torch.testing.assert_close(state["mean"], torch.arange(3))
    assert not checkpoint.with_suffix(".pt.tmp").exists()
