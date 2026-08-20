"""Build the model/TCP provenance gate for every registered native arm."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.official_model_manifest import MODEL_SOURCES
from scripts.strict_urdf_model_audit import MODELS, OFFICIAL


def build_gate() -> dict:
    rows = []
    unqualified = []
    for name, entry in MODELS.items():
        source = entry.path.resolve()
        official_source = source.is_relative_to(OFFICIAL.resolve())
        ranking_eligible = (
            official_source
            and entry.tcp_authority in {
                "official_frame", "official_xacro_default_83p5mm",
                "model_ee_frame_115mm", "fallback_130mm", "specified_139mm"}
            and entry.joint_limit_authority != "official_urdf_unverified_placeholder"
        )
        if not ranking_eligible:
            unqualified.append(name)
        manifest = MODEL_SOURCES[name]
        rows.append({
            **manifest,
            "robot": name,
            "runtime_model": str(source.relative_to(ROOT)),
            "runtime_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "official_directory_source": official_source,
            "tcp_parent": entry.tcp_parent,
            "tcp_authority": entry.tcp_authority,
            "tool_offset_m": entry.tool_offset_m,
            "tool_axis": list(entry.tool_axis),
            "joint_limit_authority": entry.joint_limit_authority,
            "active_joints": list(entry.joints),
            "ranking_eligible": ranking_eligible,
        })
    return {
        "schema_version": 3,
        "status": "pass" if not unqualified else "pass_with_limit_warning",
        "experiment_execution_allowed": all(row["official_directory_source"] for row in rows),
        "scientific_ranking_requires_disclosure": bool(unqualified),
        "unqualified_ranking_robots": unqualified,
        "tcp_policy": ("model-defined TCP frame when supplied (including PiPER-X ee_frame "
                       "and OpenArm hand_tcp); Doosan uses the specified 0.139 m tool "
                       "extension; otherwise fixed 0.130 m along the audited tool axis"),
        "geometry_policy": "pinned vendor-native dimensions with no morphology rescaling",
        "robots": rows,
    }


def main() -> None:
    payload = build_gate()
    output = ROOT / "reports/single_arm/official_model_provenance_gate.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
