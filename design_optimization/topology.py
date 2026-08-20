from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .robot_contract import COMMON_TOOL_LENGTH_M


@dataclass(frozen=True)
class TopologyTemplate:
    name: str
    axes: torch.Tensor
    deltas: torch.Tensor
    home_rotation: torch.Tensor
    q_min: torch.Tensor
    q_max: torch.Tensor
    tool_length_m: float = COMMON_TOOL_LENGTH_M
    tool_transform: torch.Tensor | None = None
    first_joint_origin_m: torch.Tensor | None = None
    source_model: Path | None = None
    tcp_authority: str = "legacy_unspecified"

    def __post_init__(self) -> None:
        dof = int(self.axes.shape[0])
        if self.axes.shape != (dof, 3) or self.deltas.shape != (dof, 3):
            raise ValueError("axes and deltas must have matching [dof,3] shapes")
        if self.q_min.shape != (dof,) or self.q_max.shape != (dof,):
            raise ValueError("joint bounds must have shape [dof]")
        if self.tool_transform is not None and self.tool_transform.shape != (4, 4):
            raise ValueError("tool_transform must have shape [4,4]")
        if self.first_joint_origin_m is not None and self.first_joint_origin_m.shape != (3,):
            raise ValueError("first_joint_origin_m must have shape [3]")

    @property
    def dof(self) -> int:
        return int(self.axes.shape[0])

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "TopologyTemplate":
        return TopologyTemplate(
            self.name,
            self.axes.to(device=device, dtype=dtype),
            self.deltas.to(device=device, dtype=dtype),
            self.home_rotation.to(device=device, dtype=dtype),
            self.q_min.to(device=device, dtype=dtype),
            self.q_max.to(device=device, dtype=dtype),
            self.tool_length_m,
            None if self.tool_transform is None else self.tool_transform.to(device=device, dtype=dtype),
            None if self.first_joint_origin_m is None else self.first_joint_origin_m.to(device=device, dtype=dtype),
            self.source_model,
            self.tcp_authority,
        )


@dataclass(frozen=True)
class DesignBatch:
    """A batch of topology-preserving designs sharing joint axes."""

    template: TopologyTemplate
    deltas: torch.Tensor  # [design, dof, 3]
    points: torch.Tensor  # [design, dof, 3]
    home: torch.Tensor  # [design, 4, 4], flange frame
    reach_scale: torch.Tensor  # [design]

    @property
    def count(self) -> int:
        return int(self.deltas.shape[0])


def load_templates(
    audit_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, TopologyTemplate]:
    payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    payload = payload.get("robots", payload)
    templates: dict[str, TopologyTemplate] = {}
    for name, row in payload.items():
        deltas_key = next(
            (key for key in (
                "native_deltas_m", "source_deltas_m", "flange_deltas_m",
            ) if key in row),
            None,
        )
        if deltas_key is None:
            raise ValueError(f"{name}: audit row has no joint delta field")
        deltas = torch.tensor(row[deltas_key], dtype=dtype, device=device)
        tool_translation = row.get("flange_tcp_translation_m", (0.0, 0.0, COMMON_TOOL_LENGTH_M))
        audited_tool_length = float(sum(float(value) ** 2 for value in tool_translation) ** 0.5)
        tool_transform = torch.eye(4, dtype=dtype, device=device)
        tool_transform[:3, 3] = torch.tensor(
            tool_translation,
            dtype=dtype, device=device
        )
        templates[name] = TopologyTemplate(
            name=name,
            axes=torch.tensor(row["axes"], dtype=dtype, device=device),
            deltas=deltas,
            home_rotation=torch.tensor(row["flange_home_rotation"], dtype=dtype, device=device),
            q_min=torch.tensor(row["q_min_rad"], dtype=dtype, device=device),
            q_max=torch.tensor(row["q_max_rad"], dtype=dtype, device=device),
            # Use the audited physical TCP transform. Most arms retain the
            # common 130 mm tool; models whose real grasp center is farther
            # out (currently Nero) must not be screened with a shorter proxy.
            tool_length_m=audited_tool_length,
            tool_transform=tool_transform,
            first_joint_origin_m=torch.tensor(
                row.get("first_joint_origin_m", (0.0, 0.0, 0.0)), dtype=dtype, device=device
            ),
            source_model=Path(row["source_model"]).resolve() if row.get("source_model") else None,
            tcp_authority=str(row.get("tcp_authority", "legacy_unspecified")),
        )
        dof = templates[name].dof
        if deltas.shape != (dof, 3):
            raise ValueError(f"{name}: expected {dof} 3-D joint points/deltas")
    return templates
