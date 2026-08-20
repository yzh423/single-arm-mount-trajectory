from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import torch


@dataclass(frozen=True)
class EgoDexSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


class EgoDexPoseDataset:
    """Memory-mapped access to the 108 MB EgoDex pose-only aggregate."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self.offsets = self.data["episode_offsets"]
        self.tasks = self.data["episode_task"]
        self.sources = self.data["episode_source"]

    def split(self, train=0.70, validation=0.15) -> EgoDexSplit:
        buckets = [[], [], []]
        for index, source in enumerate(self.sources):
            value = int.from_bytes(hashlib.sha256(str(source).encode()).digest()[:8], "little") / 2**64
            bucket = 0 if value < train else (1 if value < train + validation else 2)
            buckets[bucket].append(index)
        return EgoDexSplit(*(np.asarray(row, dtype=np.int64) for row in buckets))

    def sample_frames(self, episodes: np.ndarray, count: int, seed: int = 750) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed)
        episode_ids = rng.choice(episodes, size=count, replace=True)
        indices = np.asarray([rng.integers(self.offsets[e], self.offsets[e + 1]) for e in episode_ids])
        return {
            key: torch.from_numpy(np.asarray(self.data[key][indices]).copy())
            for key in ("left_relative_xyz", "left_relative_quat_wxyz",
                        "right_relative_xyz", "right_relative_quat_wxyz")
        }

    def sample_indices(self, episodes: np.ndarray, count: int, seed: int = 750,
                       minimum_confidence: float = 0.8) -> np.ndarray:
        """Deterministically sample valid bimanual frames from episode IDs."""
        rng = np.random.default_rng(seed)
        accepted: list[int] = []
        attempts = 0
        while len(accepted) < count and attempts < count * 30:
            episode = int(rng.choice(episodes))
            index = int(rng.integers(self.offsets[episode], self.offsets[episode + 1]))
            if (self.data["left_confidence"][index] >= minimum_confidence and
                    self.data["right_confidence"][index] >= minimum_confidence):
                accepted.append(index)
            attempts += 1
        if len(accepted) != count:
            raise RuntimeError(f"only sampled {len(accepted)}/{count} confidence-qualified frames")
        return np.asarray(accepted, dtype=np.int64)

    def episode(self, episode_id: int) -> dict[str, np.ndarray]:
        start, stop = int(self.offsets[episode_id]), int(self.offsets[episode_id + 1])
        keys = ("time_s", "left_confidence", "right_confidence", "left_relative_xyz",
                "left_relative_quat_wxyz", "right_relative_xyz", "right_relative_quat_wxyz")
        return {key: np.asarray(self.data[key][start:stop]).copy() for key in keys}

    def statistics(self) -> dict[str, object]:
        frames = int(self.offsets[-1])
        return {"episodes": len(self.tasks), "frames": frames, "hours": frames / 30.0 / 3600.0,
                "task_classes": len(np.unique(self.tasks))}
