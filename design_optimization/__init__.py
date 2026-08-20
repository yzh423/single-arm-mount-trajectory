"""Task-driven robot design optimization.

Torch-backed exports are loaded lazily so data preparation and model parsing
do not initialize a GPU/OpenMP runtime merely by importing a lightweight
submodule.
"""

from __future__ import annotations

from typing import Any


__all__ = ["DesignBatch", "TopologyTemplate", "load_templates"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .topology import DesignBatch, TopologyTemplate, load_templates

        return {
            "DesignBatch": DesignBatch,
            "TopologyTemplate": TopologyTemplate,
            "load_templates": load_templates,
        }[name]
    raise AttributeError(name)
