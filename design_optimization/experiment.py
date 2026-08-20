from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import hashlib
import os
import platform
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class DataConfig:
    path: str = r"C:\Users\KelvinLM\Documents\Codex\Doosan_Dual_Quest3\data\EgoDex\pose_only_test\egodex_pose_only_test.npz"
    train_frames: int = 192
    validation_frames: int = 256
    test_frames: int = 512
    minimum_confidence: float = 0.8
    seed: int = 750


@dataclass
class IKConfig:
    seeds: int = 24
    iterations: int = 90
    position_tolerance_m: float = 0.0025
    orientation_tolerance_deg: float = 1.5


@dataclass
class SearchConfig:
    population: int = 32
    generations: int = 50
    elite_fraction: float = 0.2
    checkpoint_every: int = 1
    snapshot_interval_minutes: float = 60.0
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class ConstraintConfig:
    topology_tolerance_mm: float = 0.1
    xarm_ur_motor_offset_m: float = 0.057
    kinova_vendor_fraction: float = 0.45
    collision_margin_m: float = 0.015
    table_height_m: float = 0.0
    collision_aware_branch_selection: bool = True
    collision_hard_constraint: bool = False
    optimize_outer_elbow: bool = False
    outer_elbow_branch_weight: float = 0.0


@dataclass
class ExperimentConfig:
    topology: str = "doosan"
    run_name: str = "egodex_pareto"
    data: DataConfig = field(default_factory=DataConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            topology=payload.get("topology", cls.topology),
            run_name=payload.get("run_name", cls.run_name),
            data=DataConfig(**payload.get("data", {})),
            ik=IKConfig(**payload.get("ik", {})),
            search=SearchConfig(**payload.get("search", {})),
            constraints=ConstraintConfig(**payload.get("constraints", {})),
        )

    def write(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def save_checkpoint_atomic(path: str | Path, state: dict[str, Any]) -> None:
    """Crash-safe checkpoint: write a sibling temporary file then replace."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def reproducibility_manifest(config: ExperimentConfig, dataset_path: str | Path) -> dict[str, Any]:
    """Capture enough immutable context to reproduce a research run."""
    path = Path(dataset_path).resolve()
    cuda_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "config": asdict(config),
        "dataset": {"path": str(path), "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path)},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "cuda_runtime": torch.version.cuda},
        "hardware": {"cuda_available": torch.cuda.is_available(), "cuda_device": cuda_device},
        "determinism": {"seed": config.data.seed,
                        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
    }
