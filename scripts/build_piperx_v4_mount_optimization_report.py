"""Build the Fold_Box fixed-time mount and IK optimization addendum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
)

from factory_bimanual.mount_topology import (
    MountTopologyConfig, MuJoCoMountTopologyChecker,
)
from factory_bimanual.mujoco_collision_adapter import (
    MuJoCoPairedCollisionChecker,
)
from factory_bimanual.multitask_fixed_time_study import (
    discover_dual_hand_trajectories,
)
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.video import (
    build_realtime_timing, interpolation_sample,
)
from scripts.build_piperx_controller_event_manifest import (
    validate_archive_summary,
)
from scripts.audit_piperx_v4_target_clearance import (
    audit_target_clearance,
)
from scripts.build_piperx_literature_optimization_report import (
    _register_font, _styles, _table,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = (ROOT / "reports/piperx_controller_event_v4/shards"
                 / "8-11/Fold_Box/161044/baseline")
TRIAL_ROOT = (ROOT / "reports/piperx_controller_event_v4_mount_trials"
              / "right_shift_10cm_3cm")
OPTIMIZED_ROOT = (TRIAL_ROOT / "full/shards/8-11/Fold_Box/161044/baseline")
DEFAULT_OUTPUT = (ROOT / "output/pdf"
                  / "PiperX_Fold_Box_Fixed_Time_mount_IK_优化核验报告.pdf")


TARGET_KEYS = (
    "source_time_s", "source_poll_row_index",
    "left_target_position_m", "right_target_position_m",
    "left_target_quaternion_wxyz", "right_target_quaternion_wxyz",
)


def validate_identical_targets(baseline, optimized):
    """Reject a mount comparison if any raw target or source time changed."""
    for key in TARGET_KEYS:
        if not np.array_equal(np.asarray(baseline[key]),
                              np.asarray(optimized[key])):
            raise ValueError(f"mount comparison target/time drift: {key}")
    return True


def mount_comparison(baseline, optimized):
    """Compare accepted event counts only for the same complete recording."""
    if (baseline["source_sha256"] != optimized["source_sha256"]
            or baseline["controller_event_rows"]
            != optimized["controller_event_rows"]):
        raise ValueError("mount comparisons require the same source and window")
    count = int(baseline["controller_event_rows"])
    if count < 1:
        raise ValueError("comparison requires at least one controller event")
    old = int(baseline["both_accept_frames"])
    new = int(optimized["both_accept_frames"])
    if not (0 <= old <= count and 0 <= new <= count):
        raise ValueError("accepted event counts are outside the source window")
    return {
        "event_count": count,
        "baseline_accepted": old,
        "optimized_accepted": new,
        "accepted_gain": new - old,
        "coverage_gain_percentage_points": 100.0 * (new - old) / count,
    }


def _read_shard(directory):
    summary_path = next(Path(directory).glob("*.summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scene = ROOT / summary["artifacts"]["scene_xml"]["path"]
    trajectory = ROOT / summary["artifacts"]["trajectory_npz"]["path"]
    with np.load(trajectory, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    spec = next(candidate for candidate in discover_dual_hand_trajectories(
        ROOT / "data/factory") if candidate.key == summary["trajectory"])
    validate_archive_summary(summary, arrays, spec=spec)
    return summary, arrays, scene, trajectory


def audit_video_interpolation_safety(scene, arrays, *, fps=30):
    """Check every encoded-time kinematic pose, not just saved events."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    names = {side: {
        "joints": ROBOT_CONTRACTS["piperx"].prefixed_joint_names(side),
        "site": f"{side}_tcp",
    } for side in ("left", "right")}
    qids = {side: np.asarray([
        model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in names[side]["joints"]], dtype=int)
        for side in ("left", "right")}
    collision = MuJoCoPairedCollisionChecker(
        model, data, names, transition_steps=5, clearance_margin_m=.015)
    topology = MuJoCoMountTopologyChecker(
        model, data, names,
        config=MountTopologyConfig(transition_steps=5))
    source_time = np.asarray(arrays["source_time_s"], dtype=float)
    qpos = np.asarray(arrays["qpos"], dtype=float)
    timing = build_realtime_timing(source_time, fps)
    invalid_collision = invalid_topology = 0
    minimum_clearance = float("inf")
    for time in timing.encoded_time_s:
        lower, upper, alpha = interpolation_sample(
            source_time, float(time))
        state = (1 - alpha) * qpos[lower] + alpha * qpos[upper]
        pair = tuple(state[qids[side]] for side in ("left", "right"))
        report = collision.state(*pair)
        invalid_collision += int(not report.valid)
        invalid_topology += int(not topology.state(*pair).valid)
        minimum_clearance = min(
            minimum_clearance, collision.clearance(*pair).minimum_m)
    return {
        "frame_count": len(timing.encoded_time_s),
        "source_duration_s": float(timing.source_duration_s),
        "invalid_collision_or_clearance_frames": invalid_collision,
        "invalid_topology_frames": invalid_topology,
        "minimum_named_clearance_m": minimum_clearance,
    }


def _make_figures(baseline, optimized, arrays, certificate):
    figure_dir = TRIAL_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    mount_image = figure_dir / "fold_box_raw_target_mount_layout.png"
    gap_image = figure_dir / "fold_box_gripper_clearance_certificate.png"

    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    ax.add_patch(plt.Rectangle(
        (-.9, -.7), 1.8, 1.4, facecolor="#EDF3F7",
        edgecolor="#91A6B5", linewidth=1.2))
    for side, color in (("left", "#1684C7"), ("right", "#E58B31")):
        track = np.asarray(arrays[f"{side}_target_position_m"])
        ax.plot(track[:, 0], track[:, 1], color=color, alpha=.34,
                linewidth=1.2, label=f"{side} raw TCP path")
        old = np.asarray(baseline["mount"]["xy"][side], dtype=float)
        new = np.asarray(optimized["mount"]["xy"][side], dtype=float)
        ax.scatter(*old, s=130, facecolors="none", edgecolors=color,
                   linewidths=2.0, marker="o")
        ax.scatter(*new, s=80, color=color, marker="o", zorder=5)
        if np.linalg.norm(new - old) > 1e-9:
            ax.annotate("", xy=new, xytext=old,
                        arrowprops={"arrowstyle": "->", "color": color,
                                    "linewidth": 1.7})
    ax.plot([], [], "o", markerfacecolor="none", markeredgecolor="#374C59",
            label="previous base")
    ax.plot([], [], "o", color="#374C59", label="optimized base")
    ax.set(xlim=(-.94, .94), ylim=(-.74, .74), xlabel="world X (m)",
           ylabel="world Y (m)", aspect="equal")
    ax.grid(alpha=.18)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.savefig(mount_image, dpi=210, facecolor="white")
    plt.close(fig)

    times = np.asarray(arrays["source_time_s"])
    gaps = 1000 * np.asarray(certificate["exact_target_gap_m"])
    gain_mm = 1000 * certificate["maximum_tolerance_clearance_gain_m"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    window = (times >= 1.3) & (times <= 2.9)
    ax.plot(times[window], gaps[window], color="#245E91", linewidth=2.0,
            label="exact raw-target gripper-base gap")
    ax.axhline(15.0, color="#D25D49", linestyle="--", linewidth=1.5,
               label="required clearance: 15 mm")
    ax.axhline(15.0 - gain_mm, color="#AA6D2A", linestyle=":",
               linewidth=1.5, label="certified infeasibility threshold")
    impossible = np.asarray(certificate["certified_infeasible_indices"], int)
    ax.scatter(times[impossible], gaps[impossible], color="#C44536",
               s=22, zorder=4, label="provably infeasible events")
    ax.set(xlim=(1.3, 2.9), ylim=(0, 65), xlabel="fixed source time (s)",
           ylabel="gripper-base gap (mm)")
    ax.grid(alpha=.22)
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(gap_image, dpi=210, facecolor="white")
    plt.close(fig)
    return mount_image, gap_image


def _failure_runs(acceptance):
    groups = []
    for index in np.flatnonzero(~np.asarray(acceptance, dtype=bool)):
        if not groups or index > groups[-1][-1] + 1:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return [(group[0], group[-1], len(group)) for group in groups]


def _footer(canvas, document):
    canvas.saveState()
    width, _height = A4
    canvas.setStrokeColor(colors.HexColor("#A8BCCB"))
    canvas.line(17 * mm, 14 * mm, width - 17 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#617584"))
    canvas.drawString(17 * mm, 10 * mm,
                      "PiperX Fold_Box | fixed controller-event time | geometric study")
    canvas.drawRightString(width - 17 * mm, 10 * mm,
                           str(document.page))
    canvas.restoreState()


def build(output=DEFAULT_OUTPUT):
    baseline, old_arrays, _old_scene, _old_npz = _read_shard(BASELINE_ROOT)
    optimized, arrays, scene, trajectory = _read_shard(OPTIMIZED_ROOT)
    validate_identical_targets(old_arrays, arrays)
    comparison = mount_comparison(baseline, optimized)
    if (optimized["retiming_applied"] or optimized["dynamics_enforced"]
            or any(int(optimized[key]) for key in (
                "collision_frames", "edge_collision_frames",
                "topology_invalid_frames"))):
        raise ValueError("optimized geometric evidence violates the report contract")
    certificate_path = TRIAL_ROOT / "target_clearance_certificate.json"
    saved_certificate = json.loads(
        certificate_path.read_text(encoding="utf-8"))
    certificate = audit_target_clearance(scene, trajectory)
    if (saved_certificate["certified_infeasible_indices"]
            != certificate["certified_infeasible_indices"]
            or not np.allclose(
                saved_certificate["exact_target_gap_m"],
                certificate["exact_target_gap_m"], rtol=0.0, atol=1e-9)
            or certificate["accepted_infeasible_overlap_count"]):
        raise ValueError("target-clearance certificate is stale or contradicted")
    portfolio = TRIAL_ROOT / "portfolio"
    video = portfolio / "fold-box-piperx-optimized-mount-complete-trajectories.mp4"
    poster = portfolio / "fold-box-piperx-optimized-mount-complete-trajectories-poster.png"
    provenance = json.loads(video.with_suffix(
        ".provenance.json").read_text(encoding="utf-8"))
    if (not video.is_file() or not poster.is_file()
            or provenance["controller_event_count"] != comparison["event_count"]
            or not np.isclose(provenance["strict_paired_coverage"],
                              optimized["both_accept_coverage"])):
        raise ValueError("optimized portfolio media does not match the shard")
    qa = audit_video_interpolation_safety(scene, arrays)
    if (qa["invalid_collision_or_clearance_frames"]
            or qa["invalid_topology_frames"]):
        raise ValueError("interpolated video frames violate safety")
    mount_image, gap_image = _make_figures(
        baseline, optimized, arrays, certificate)

    _register_font()
    styles = _styles()
    center_small = ParagraphStyle(
        "caption_center", parent=styles["small"], alignment=TA_CENTER)
    story = []

    def P(value, style="body"):
        story.append(Paragraph(value, styles[style]))

    P("PiperX Fold_Box 双手 Fixed-time 安装与 IK 优化核验", "title")
    P("完整原始轨迹 · 双手同步控制事件 · MuJoCo 运动学重放", "subtitle")
    P(f"在相同的 {comparison['event_count']} 个控制事件和相同目标下，"
      f"双手严格达标由 <b>{comparison['baseline_accepted']}/{comparison['event_count']} "
      f"({100*baseline['both_accept_coverage']:.2f}%)</b> 提升至 "
      f"<b>{comparison['optimized_accepted']}/{comparison['event_count']} "
      f"({100*optimized['both_accept_coverage']:.2f}%)</b>，多跟随 "
      f"<b>{comparison['accepted_gain']} 帧</b>；所有已发布状态、扫掠边和拓扑检查为零违规。"
      "这仍不是 100% 完全跟随，也不是硬件可执行轨迹。", "callout")
    P("1. 可比协议与结果", "h1")
    P("源数据为 Fold_Box/161044 双手记录：1,478 条主机轮询行折叠为 755 个左右同步控制器更新事件。"
      "每个事件保留原 receive_monotonic_s 时间与原始轮询行索引；无重定时、无目标平滑、无随时间翻腕，"
      "使用固定工具 SE(3) 变换。双手同时满足位置 ≤1 mm、姿态 ≤0.5°才计为跟随。", "body")
    rows = [
        ["安装/范围", "达标", "覆盖率", "失败帧", "安全违规"],
        ["原桌面基线 / 全程", "716/755", "94.83%", "39", "0 / 0 / 0"],
        ["局部候选 moveA / 前300", "260/300", "86.67%", "40", "0 / 0 / 0"],
        ["仅右底座调整 / 前300", "284/300", "94.67%", "16", "0 / 0 / 0"],
        ["仅右底座调整 / 全程", "739/755", "97.88%", "16", "0 / 0 / 0"],
    ]
    story.append(_table(rows, [55*mm, 31*mm, 27*mm, 24*mm, 39*mm], 7.6))
    story.append(Spacer(1, 3*mm))
    story.append(Image(str(mount_image), width=160*mm, height=76*mm))
    story.append(Paragraph(
        "图 1. 桌面俯视：淡线为完整原始双手 TCP 轨迹；空心圆为旧底座，实心圆为新底座。"
        "左底座不变；右底座沿 X 移动 -0.10 m、沿 Y 移动 +0.03 m。",
        center_small))
    P("新双底座中心距 0.6041 m，高度均为 0.7571 m；该设置通过项目的桌面范围、"
      "底座最小间距和支撑高度检查。此处只改变 mount，未改源轨迹、工具外参、IK 容差或碰撞间隙。", "small")

    story.append(PageBreak())
    P("2. IK 失败段与严格跟随上界", "h1")
    P("旧解在 0–22、64、98–112 号事件 HOLD。新安装恢复了起始 0–22 号事件；"
      "仍在 64、98–112 号事件执行安全 HOLD。新解已接受帧的最大误差为 "
      f"{optimized['maximum_accepted_position_error_mm']:.4f} mm、"
      f"{optimized['maximum_accepted_orientation_error_deg']:.4f}°。", "body")
    story.append(Image(str(gap_image), width=160*mm, height=74*mm))
    story.append(Paragraph(
        "图 2. 由固定原始双手 TCP 位姿直接确定的夹爪底座间隙。"
        "低于点线的事件，即便在允许的位姿误差范围内也无法达到 15 mm 间隙。",
        center_small))
    P("夹爪底座相对 TCP 为刚体，故其两侧几何间隙不随底座安装和 IK 分支改变。"
      "根据各碰撞几何到 TCP 的包围半径 0.1772 m，1 mm 平移与 0.5° 旋转容差至多能将"
      f"两侧间隙提高 {1000*certificate['maximum_tolerance_clearance_gain_m']:.3f} mm。"
      f"原始目标最小间隙仅 {1000*certificate['minimum_exact_target_gap_m']:.3f} mm；"
      "第 99–107 号共 9 个事件满足“目标间隙 + 最有利容差上界 &lt; 15 mm”，"
      "因此严格安全跟随的理论上限不超过 746/755（98.81%）。这是必要条件上界，"
      "不表示剩余 746 帧一定有连续、动力学可行的 IK。", "callout")
    P("失败类别解释", "h2")
    failure_rows = [
        ["事件", "实测候选原因", "当前处理"],
        ["0（旧安装）", "右臂 link4/link6 候选自碰撞约 2.72 mm", "新安装恢复"],
        ["64、98", "候选夹爪底座间隙不足 15 mm", "安全 HOLD"],
        ["99–107", "原始 TCP 目标几何本身与间隙门冲突", "不可标为成功"],
        ["112", "候选静态安全，但旧姿态到候选的扫掠边不安全", "安全 HOLD"],
    ]
    story.append(_table(failure_rows, [24*mm, 108*mm, 44*mm], 7.3))
    P("候选碰撞不是视频中执行的碰撞。执行序列的 755 个状态、全部扫掠边、安装拓扑均为零违规；"
      f"另外逐帧检查视频 {qa['frame_count']} 个插值画面，碰撞/间隙与拓扑违规均为 0，"
      f"最小命名间隙为 {1000*qa['minimum_named_clearance_m']:.3f} mm。", "small")

    story.append(PageBreak())
    P("3. 真实 MuJoCo 渲染与可执行性边界", "h1")
    story.append(Image(str(poster), width=160*mm, height=90*mm))
    story.append(Paragraph(
        "图 3. 新安装的完整轨迹视频截图：原始双手路径、实际双手 TCP 路径、"
        "当前姿态、误差与安全 HOLD 状态均来自同一完整时间轴。", center_small))
    P(f"视频编码为 {provenance['frame_count']} 帧、{provenance['fps']:.0f} fps、"
      f"{provenance['encoded_duration_s']:.3f} s；源事件跨度为 "
      f"{provenance['source_duration_s']:.3f} s。完整目标与实际 TCP 轨迹从首帧显示，"
      "局部高亮随源时间推进。当前 TCP 游标是 MuJoCo 对归档关节状态的正运动学结果；"
      "不是前向动力学仿真，也不是实体 PiperX 回放。", "body")
    P("固定时间的动力学检查", "h2")
    dynamics_rows = [
        ["完整轨迹", "最大关节速度", "最大关节加速度", "已强制动力学"],
        ["旧安装", f"{baseline['maximum_velocity_rad_s']:.2f} rad/s",
         f"{baseline['maximum_acceleration_rad_s2']:.1f} rad/s²", "否"],
        ["新安装", f"{optimized['maximum_velocity_rad_s']:.2f} rad/s",
         f"{optimized['maximum_acceleration_rad_s2']:.1f} rad/s²", "否"],
        ["SDK V2 可配置上限", "3.0 rad/s", "5.0 rad/s²", "需单独验收"],
    ]
    story.append(_table(dynamics_rows, [44*mm, 41*mm, 49*mm, 42*mm], 7.2))
    P("新 mount 提升的是严格几何跟随，而非真实硬件可下发性。关节导数明显超过 Piper SDK V2 "
      "文档所列可配置上限；不能通过缩短视频或重定时掩盖。若保持当前原始目标和 15 mm "
      "间隙，至少 9 帧的严格完全跟随已被几何证明不可行。空间重映射方案可单独研究，"
      "但必须与原始目标严格跟随分列指标。", "callout")
    P("复现与来源", "h2")
    P("求解入口：scripts/run_piperx_controller_event_v4.py 的 --mount-json 与隔离 --output。"
      "间隙证书：scripts/audit_piperx_v4_target_clearance.py。"
      "完整归档、证书、渲染视频和逐文件溯源位于 reports/piperx_controller_event_v4_mount_trials/right_shift_10cm_3cm。", "small")
    P("Piper SDK V2 官方接口："
      "<link href='https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD'>"
      "github.com/agilexrobotics/piper_sdk/asserts/V2/INTERFACE_V2.MD</link>。"
      "本报告是 Fold_Box 的增补核验；不替代原有多任务报告，也不将该任务结果外推至其他记录。", "small")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=17*mm,
        rightMargin=17*mm, topMargin=17*mm, bottomMargin=19*mm)
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(build(args.output))


if __name__ == "__main__":
    main()
