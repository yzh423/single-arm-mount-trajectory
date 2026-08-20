import json
from pathlib import Path

import pytest

from factory_bimanual import cli


ROOT = Path(__file__).resolve().parents[2]


def test_full_requires_explicit_confirmation():
    with pytest.raises(SystemExit, match="confirm-full"):
        cli.main(["full"], root=ROOT)


def test_full_requires_ready_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_preflight", lambda *a, **k: {"ready": False, "blockers": ["spacing"]})
    with pytest.raises(SystemExit, match="preflight is not ready"):
        cli.main(["full", "--confirm-full"], root=ROOT)


def test_dry_run_only_plans_jobs(monkeypatch):
    class FakeExperiment:
        def __init__(self, config, **kwargs): self.config = config
        def run(self): return {"dry_run": self.config.dry_run, "jobs": [{"status": "planned"}]}
    monkeypatch.setattr(cli, "FactoryBimanualExperiment", FakeExperiment)
    result = cli.main(["dry-run"], root=ROOT)
    assert result["dry_run"] is True
    assert all(row["status"] == "planned" for row in result["jobs"])


def test_cli_uses_supplied_root_for_output(monkeypatch, tmp_path):
    captured = {}
    class FakeExecutors:
        def __init__(self, root, output_root): captured.update(root=root, output=output_root)
        def mapping(self): return {"ik": object(), "mpc": object()}
        def ik(self, job): pass
        def mpc(self, job): pass
    monkeypatch.setattr(cli, "ProductionExecutors", FakeExecutors)
    monkeypatch.setattr(cli, "FactoryBimanualExperiment", lambda config, **kw: type("X", (), {"run": lambda s: {}})())
    cli.main(["dry-run"], root=tmp_path)
    assert captured["output"] == (tmp_path / "reports/factory_bimanual").resolve()
