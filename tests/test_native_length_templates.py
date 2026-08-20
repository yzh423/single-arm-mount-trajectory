from design_optimization.topology import load_templates
from scripts.solve_strict_urdf_task_cache import build_model
from scripts.strict_mujoco_model import sampled_maximum_tcp_reach_m
from scripts.strict_urdf_model_audit import MODELS


def test_templates_load_official_native_audit_directly():
    audit = "reports/single_arm/model_audit.json"
    native = load_templates(audit)
    assert native["ur5"].tool_length_m == 0.0


def test_real_urdf_build_restores_native_flange_reach():
    model = build_model("ur5", [0.0, 0.0, 0.0], 0.0)

    reach = sampled_maximum_tcp_reach_m(model, MODELS["ur5"].joints, "strict_flange", sample_count=2048)

    assert 1.0 < reach < 1.1
