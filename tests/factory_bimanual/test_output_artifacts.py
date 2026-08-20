import json
from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.artifacts import FrameDiagnostics, RunArtifactWriter


def test_writer_retains_failed_frames_and_only_writes_factory_namespace(tmp_path: Path):
    root = tmp_path / "reports" / "factory_bimanual"
    writer = RunArtifactWriter(root)
    frames = [
        FrameDiagnostics(10, 0.0, "ok", "ok", False, False, "source.csv"),
        FrameDiagnostics(11, 0.12, "branch_lost", "ok", True, False, "source.csv"),
    ]
    result = writer.write_run(
        "xarm6/screw_cap/strict_a", frames,
        arrays={"source_row_indices": np.array([10, 11]), "time_s": np.array([0.0, 0.12])},
        summary={"planning_success": False, "execution_success": None},
    )
    assert result.json_path.is_relative_to(root)
    assert result.npz_path.is_relative_to(root)
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert len(payload["frames"]) == 2
    assert payload["frames"][1]["left_failure_reason"] == "branch_lost"
    npz = np.load(result.npz_path)
    assert npz["source_row_indices"].tolist() == [10, 11]

    with pytest.raises(ValueError, match="unsafe run key"):
        writer.write_run("../single_arm/oops", frames, {}, {})

