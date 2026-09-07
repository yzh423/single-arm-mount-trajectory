"""Build the controller-event v4 PiperX root-cause and optimization PDF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from scripts.build_piperx_literature_optimization_report import (
    FONT, MODE_ZH, _figure, _register_font, _styles, _table,
)
from scripts.build_piperx_controller_event_manifest import (
    validate_archive_summary,
)
from factory_bimanual.multitask_fixed_time_study import (
    discover_dual_hand_trajectories,
)


ROOT = Path(__file__).resolve().parents[1]
PREFIX_ROOT = ROOT / "reports/piperx_controller_event_v4_prefix300"
FULL_ROOT = ROOT / "reports/piperx_controller_event_v4"
OLD_AUDIT = (ROOT / "reports/piperx_multitask_fixed_time_mount_study"
             / "literature_audit/raw_source_shard_audit.csv")
DEFAULT_OUTPUT = ROOT / "output/pdf/PiperX双手Fixed-Time-v4根因诊断与优化报告.pdf"
TASKS = (
    "8-11/Seal_Bag/161504",
    "8-11/Fold_Box/161044",
)
MODES = ("baseline", "upright_table", "horizontal_wall", "inverted")


def validate_report_scope(prefix, full):
    """Keep full-run claims separate from the 300-event comparison window."""
    expected_prefix = {(task, mode) for task in TASKS for mode in MODES}
    if not expected_prefix.issubset(prefix):
        raise ValueError("eight prefix300 controller-event summaries are required")
    expected_full = {
        (task, mode) for task in TASKS
        for mode in ("baseline", "upright_table")}
    if not expected_full.issubset(full):
        raise ValueError(
            "complete baseline and upright controller-event summaries are required")
    for key in expected_prefix:
        value = prefix[key]
        total = int(value.get(
            "controller_event_rows_total", value["controller_event_rows"]))
        if int(value["controller_event_rows"]) != min(300, total):
            raise ValueError(f"prefix300 scope mismatch: {key}")
        if int(value.get(
                "controller_events_not_solved", total - 300)) != max(
                    0, total - 300):
            raise ValueError(f"prefix300 remainder mismatch: {key}")
    for key in expected_full:
        value = full[key]
        total = int(value.get(
            "controller_event_rows_total", value["controller_event_rows"]))
        if (int(value["controller_event_rows"]) != total
                or int(value.get("controller_events_not_solved", 0)) != 0):
            raise ValueError(f"complete-run scope mismatch: {key}")
    return True


def _read_summaries(root):
    result = {}
    specs = {
        spec.key: spec for spec in discover_dual_hand_trajectories(
            ROOT / "data/factory")}
    for path in Path(root).rglob("*.summary.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        key = (value["trajectory"], value["mode"])
        if key in result:
            raise ValueError(f"duplicate controller-event summary: {key}")
        if value["trajectory"] not in specs:
            raise ValueError(f"unknown source trajectory: {key}")
        trajectory_path = ROOT / value["trajectory_artifact"]
        with np.load(trajectory_path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        validate_archive_summary(
            value, payload, spec=specs[value["trajectory"]])
        result[key] = value
    if not result:
        raise ValueError(f"no controller-event summaries found under {root}")
    return result


def _read_old_baselines():
    with OLD_AUDIT.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        row["trajectory"]: row for row in rows
        if row["trajectory"] in TASKS and row["mode"] == "baseline"
    }


def _make_figures(prefix, full, old):
    destination = PREFIX_ROOT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    short = {TASKS[0]: "Seal_Bag", TASKS[1]: "Fold_Box"}

    x = np.arange(2)
    legacy_raw = [100 * float(old[task]["raw_pair_coverage"])
                  for task in TASKS]
    legacy_conditioned = [
        100 * float(old[task]["conditioned_pair_coverage"])
        for task in TASKS]
    event_raw = [100 * float(full[(task, "baseline")][
        "both_accept_coverage"]) for task in TASKS]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - .26, legacy_raw, .26, label="v3执行对原始目标复算", color="#AAB7C4")
    ax.bar(x, legacy_conditioned, .26, label="v3条件化目标", color="#E19A3E")
    ax.bar(x + .26, event_raw, .26, label="v4控制器事件原始目标", color="#16849B")
    ax.set_xticks(x, [short[task] for task in TASKS])
    ax.set_ylim(0, 108)
    ax.set_ylabel("双手严格覆盖率（%）")
    ax.set_title("相同基准安装下的目标与时间合同对照")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, fontsize=8)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    fig.tight_layout()
    contract = destination / "v3_v4_contract_comparison.png"
    fig.savefig(contract, dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)
    colors_ = ("#64748B", "#3478D4", "#E28B3C", "#884FC2")
    for ax, task in zip(axes, TASKS):
        values = [100 * float(prefix[(task, mode)]["both_accept_coverage"])
                  for mode in MODES]
        bars = ax.bar(range(4), values, color=colors_)
        ax.set_xticks(range(4), [MODE_ZH[mode] for mode in MODES], rotation=16)
        ax.set_title(short[task])
        ax.grid(axis="y", alpha=.22)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axes[0].set_ylabel("前300控制器事件严格覆盖率（%）")
    axes[0].set_ylim(0, 108)
    fig.suptitle("同一事件时间轴、同一安全门下的四安装对比")
    fig.tight_layout()
    mounts = destination / "event_v4_four_mount_coverage.png"
    fig.savefig(mounts, dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    poll = [int(full[(task, "baseline")]["source_poll_rows"])
            for task in TASKS]
    events = [int(full[(task, "baseline")].get(
        "controller_event_rows_total",
        full[(task, "baseline")]["controller_event_rows"]))
              for task in TASKS]
    duplicate = [100 * (1 - event / total)
                 for total, event in zip(poll, events)]
    bars = ax.bar([short[task] for task in TASKS], duplicate,
                  color=("#2A9D8F", "#D66A4A"), width=.55)
    ax.set_ylabel("重复轮询行比例（%）")
    ax.set_title("主机轮询行不等于控制器新目标帧")
    ax.grid(axis="y", alpha=.22)
    ax.bar_label(bars, labels=[
        f"{value:.1f}%  ({event}/{total})"
        for value, event, total in zip(duplicate, events, poll)],
        fontsize=9, padding=3)
    fig.tight_layout()
    timing = destination / "controller_event_duplicate_ratio.png"
    fig.savefig(timing, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return contract, mounts, timing


def _footer(canvas, document):
    canvas.saveState()
    canvas.setFont(FONT, 7)
    canvas.setFillColor(colors.HexColor("#657786"))
    canvas.drawString(17 * mm, 10 * mm, "PiperX Fixed-time v4 根因诊断与优化 · 2026-09-07")
    canvas.drawRightString(193 * mm, 10 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def build(output=DEFAULT_OUTPUT):
    _register_font()
    styles = _styles()
    styles["metric"] = ParagraphStyle(
        "metric", parent=styles["body"], fontName=FONT, fontSize=18,
        leading=23, alignment=TA_CENTER, textColor=colors.HexColor("#167D9A"))
    prefix = _read_summaries(PREFIX_ROOT)
    full = _read_summaries(FULL_ROOT)
    old = _read_old_baselines()
    validate_report_scope(prefix, full)
    contract_figure, mount_figure, timing_figure = _make_figures(
        prefix, full, old)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=16 * mm,
        title="PiperX双手Fixed-Time-v4根因诊断与优化报告",
        author="single-arm-mount evidence pipeline")
    story = []

    def P(text, style="body"):
        story.append(Paragraph(text, styles[style]))

    P("PiperX 双手 Fixed-time v4<br/>根因诊断与优化报告", "title")
    P("控制器事件时间 · 禁止重定时 · 原始固定工具目标 · 1 mm / 0.5° · 零碰撞硬门", "subtitle")
    P("结论先行", "h1")
    P("低成功率主要来自错误的时间语义、目标合同漂移和构型不可达，而不是双手不同步。v4 不改变播放速度，不平滑目标，不使用时变翻腕；它只把重复轮询观测还原为同步控制器新帧，并修正对应的原始行索引。", "callout")
    metric_rows = [[
        Paragraph("Seal_Bag 完整基准", styles["small"]),
        Paragraph("Fold_Box 完整基准", styles["small"]),
        Paragraph("安全违规", styles["small"]),
    ], [
        Paragraph(f"{100*full[(TASKS[0], 'baseline')]['both_accept_coverage']:.2f}%", styles["metric"]),
        Paragraph(f"{100*full[(TASKS[1], 'baseline')]['both_accept_coverage']:.2f}%", styles["metric"]),
        Paragraph("0 / 0 / 0", styles["metric"]),
    ], ["1872 控制器事件", "755 控制器事件", "frame / edge / topology"]]
    story.append(_table(metric_rows, [58 * mm] * 3, 8))
    story.append(Spacer(1, 4 * mm))
    P("这两个结果均使用完整事件序列。Fold_Box 的 v4 结果相对旧执行对原始目标的 10.08% 复算覆盖提高到 94.83%；Seal_Bag 从 31.65% 提高到 100%。旧条件化覆盖不能再充当原始轨迹完全跟随指标。")

    story.append(PageBreak())
    P("1. 根因一：轮询时间被误当成目标时间", "h1")
    story.append(_figure(timing_figure, 168 * mm))
    P("图 1｜双臂控制器帧号和接收时间同时更新；其余行只是主机读取同一帧的重复观测。", "small")
    timing_rows = [["任务", "主机轮询行", "控制器事件", "重复比例", "双臂异步更新"]]
    for task in TASKS:
        item = full[(task, "baseline")]
        total = int(item["source_poll_rows"])
        events = int(item["controller_event_rows"])
        timing_rows.append([
            task.split("/")[-2], str(total), str(events),
            f"{100*(1-events/total):.1f}%", "0",
        ])
    story.append(_table(timing_rows, [42 * mm, 33 * mm, 33 * mm, 31 * mm, 37 * mm], 7.8))
    P("旧流程按约 80 Hz 主机轮询 t 建立轨迹，控制器实际新帧只有约 43-67 Hz。重复保持值后接新值会形成伪造的“静止-突跳”序列。v4 仅合并左右帧号均未变化的连续重复行，事件时间取左右相同的 receive_monotonic_s，并在 NPZ 中保留每个事件对应的原始 CSV 行号。")
    P("这不是 retiming：控制器事件之间的真实时间间隔保持不变，也没有插值、拉伸或压缩任务时间。")

    story.append(PageBreak())
    P("2. 根因二：求解目标超出验收预算", "h1")
    story.append(_figure(contract_figure, 173 * mm))
    P("图 2｜v3 对条件化目标成功，不等于对原始固定工具目标成功；v4 直接对原始目标求解。", "small")
    contract_rows = [["合同项", "v3", "v4"]]
    contract_rows.extend([
        ["时间", "主机轮询 t", "成对控制器接收事件"],
        ["平移平滑", "最多 5 mm", "无"],
        ["姿态平滑", "最多 1°", "无"],
        ["Fold_Box 翻腕", "时变，最多 12.5°", "无"],
        ["验收", "条件化目标 1 mm / 0.5°", "原始固定工具目标 1 mm / 0.5°"],
        ["retiming", "无", "无"],
    ])
    story.append(_table(contract_rows, [46 * mm, 61 * mm, 69 * mm], 7.6))
    P("5 mm / 1° 的预处理幅度本身已经超过 1 mm / 0.5° 的总验收阈值，Fold_Box 的时变翻腕偏移更大。因此旧“成功”轨迹对原始目标复算时覆盖率会显著下降。v4 把目标变换限制为配置中固定不随时间变化的工具 SE(3) 映射。")

    story.append(PageBreak())
    P("3. 根因三：安装构型决定可达性", "h1")
    story.append(_figure(mount_figure, 177 * mm))
    P("图 3｜为保证画面对齐，四种安装均使用各任务前 300 个控制器事件；所有分片安全违规为 0。", "small")
    mount_rows = [["任务", "基准", "桌面", "墙装", "倒挂"]]
    for task in TASKS:
        mount_rows.append([task.split("/")[-2], *[
            f"{100*prefix[(task, mode)]['both_accept_coverage']:.2f}%"
            for mode in MODES]])
    story.append(_table(mount_rows, [44 * mm, 33 * mm, 33 * mm, 33 * mm, 33 * mm], 8))
    P("墙装和倒挂的低覆盖不是碰撞穿透，而是当前候选预算与求解配置没有找到同时满足双臂姿态、关节限位、15 mm 碰撞余量与拓扑门的连续 IK 分支。这不是不可行性证明。0% 时执行输出为安全 HOLD，不能计为成功。")
    P("现有外层 mount 搜索只有 16 个粗候选、3 个 dense 候选和 6 个局部候选，且主要按稀疏几何覆盖排序。推荐扩大连续安装搜索预算，并在外层目标中加入完整姿态可达率、奇异值、关节速度和加速度代价；不应把四种安装的平均覆盖当作“推荐安装成功率”。")

    story.append(PageBreak())
    P("4. 动力学结论：几何完全跟随仍不等于可下发", "h1")
    dyn_rows = [["完整基准", "覆盖", "最大关节速度", "最大关节加速度", "官方 3/5 上限"]]
    for task in TASKS:
        item = full[(task, "baseline")]
        dyn_rows.append([
            task.split("/")[-2],
            f"{100*item['both_accept_coverage']:.2f}%",
            f"{item['maximum_velocity_rad_s']:.2f} rad/s",
            f"{item['maximum_acceleration_rad_s2']:.1f} rad/s²",
            "未通过",
        ])
    story.append(_table(dyn_rows, [40 * mm, 27 * mm, 38 * mm, 43 * mm, 28 * mm], 7.6))
    P("v4 同时新增了固定时间速度/加速度限幅原语、加速度感知的多分支边门、关节限位制动包络和绝对链长不可达预筛。当前发布结果未启用该动力学选项；早期局部限幅试验出现很低的严格位姿覆盖，只能说明现有局部方法不足，不能单独证明原始任务不存在满足 3 rad/s、5 rad/s² 的其他连续 IK 分支。启用硬门后，求解器在没有安全动力学步时立即失败，并对最终速度、加速度再次验收。")
    P("因此项目必须保留两条独立输出：A 为几何 Fixed-time 跟随研究，用于 mount/IK 比较；B 为硬件可执行轨迹，必须在速度、加速度、碰撞和拓扑全部通过后才能下发。当前尚未产出 B；禁止重定时时是否存在 B，需要用全局连续分支与约束轨迹优化继续验证，不能由局部求解器失败直接下结论。")
    P("推荐的下一层求解器", "h2")
    solver_rows = [["优先级", "约束或目标", "实现"]]
    solver_rows.extend([
        ["P0", "双臂同步事件、关节界、碰撞、扫掠碰撞、拓扑", "硬门"],
        ["P1", "速度、加速度、必要时 jerk", "HQP / 窗口约束优化"],
        ["P2", "原始双手目标 1 mm / 0.5°", "容差管内最小误差"],
        ["P3", "奇异规避、分支连续、关节裕量", "SDLS + 多分支图"],
    ])
    story.append(_table(solver_rows, [26 * mm, 87 * mm, 63 * mm], 7.7))

    story.append(PageBreak())
    P("5. 文献依据与项目决策", "h1")
    literature = [
        ["来源", "可靠结论", "本项目对应决策"],
        ["Gao et al., RA-L 2021", "跨工作空间映射应同时考虑位置、姿态和速度，并保持连续性", "原始严格跟随与连续运动重映射分开报告"],
        ["Wen et al., IEEE RAM 2024", "双臂人类命令需通过任务空间约束优化适配为可行机器人参考", "不可行命令进入适配轨，不伪装为原轨迹成功"],
        ["Rakita et al., HRI 2017", "直接手-末端映射受机器人运动学和速度能力限制", "不再假定任意 VR 6-DoF 轨迹可被固定基座复现"],
        ["RelaxedIK, RSS 2018", "联合优化精度、连续性、碰撞、速度、加速度和 jerk", "替换逐帧独立 warm-start 的长期方向"],
        ["TOPP-RA, T-RO 2018", "路径动力学可行性与时间参数化不可分", "禁止重定时时显式报告不可行"],
    ]
    story.append(_table(literature, [42 * mm, 68 * mm, 66 * mm], 7.1))
    P("参考链接", "h2")
    refs = [
        "Gao et al. Motion Mappings for Continuous Bilateral Teleoperation. IEEE RA-L 6(3), 2021. <link href='https://doi.org/10.1109/LRA.2021.3068924'>doi:10.1109/LRA.2021.3068924</link>.",
        "Wen et al. Collaborative Bimanual Manipulation Using Optimal Motion Adaptation and Interaction Control. IEEE Robotics &amp; Automation Magazine 31(4), 2024. <link href='https://doi.org/10.1109/MRA.2023.3270222'>doi:10.1109/MRA.2023.3270222</link>.",
        "Rakita et al. A Motion Retargeting Method for Effective Mimicry-based Teleoperation of Robot Arms. HRI 2017. <link href='https://doi.org/10.1145/2909824.3020254'>doi:10.1145/2909824.3020254</link>.",
        "Rakita et al. RelaxedIK. RSS 2018. <link href='https://doi.org/10.15607/RSS.2018.XIV.043'>doi:10.15607/RSS.2018.XIV.043</link>.",
        "Pham &amp; Pham. TOPP-RA. IEEE T-RO 34, 2018. <link href='https://doi.org/10.1109/TRO.2018.2819195'>doi:10.1109/TRO.2018.2819195</link>.",
        "AgileX Robotics. <link href='https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD'>Piper SDK V2 官方接口</link>.",
    ]
    for value in refs:
        P(value, "small")

    story.append(PageBreak())
    P("6. 产物、复现与证据边界", "h1")
    artifact_rows = [["产物", "范围", "用途"]]
    artifact_rows.extend([
        ["piperx_controller_event_v4", "Seal/Fold 完整基准", "原始目标主结论"],
        ["piperx_controller_event_v4_prefix300", "2任务 × 4安装 × 300事件", "等长构型比较与视频"],
        ["bundle_manifest.json", "8个安全分片", "时间轴、摘要和渲染入口"],
        ["两条 four_mount_fixed_time.mp4", "真实 MuJoCo 四宫格", "同步、安装和 HOLD 可视核验"],
        ["本报告", "根因、改动、边界、文献", "项目交付说明"],
    ])
    story.append(_table(artifact_rows, [62 * mm, 49 * mm, 65 * mm], 7.5))
    P("复现命令", "h2")
    P("<font name='Courier'>python -m scripts.run_piperx_controller_event_v4 --trajectory 8-11/Seal_Bag/161504 --mode baseline</font><br/><font name='Courier'>python -m scripts.build_piperx_controller_event_manifest --output reports/piperx_controller_event_v4_prefix300</font><br/><font name='Courier'>python -m scripts.render_piperx_multitask_mount_comparisons --output reports/piperx_controller_event_v4_prefix300</font>", "small")
    P("证据边界", "h2")
    P("v4 证明了两个代表任务在基准安装下的原始目标几何跟随显著改善，并证明四安装结果在 300 事件窗口内零碰撞。它没有证明所有 27 条轨迹或四种安装均能完全跟随，也没有证明几何轨迹可直接下发真实 PiperX。其他任务必须按同一事件协议逐条重算；若允许空间重映射，必须单列为 motion-retargeted 指标。", "callout")

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(build(args.output))


if __name__ == "__main__":
    main()
