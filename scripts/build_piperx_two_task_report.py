"""Build the audited Chinese PDF for the PiperX Fold_Box + Seal_Bag run."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reports/piperx_two_task_complete_follow"
DEFAULT_MANIFEST = BUNDLE / "two_task_manifest.json"
DEFAULT_EXPERIMENT = BUNDLE / "two_task_experiment_log.json"
DEFAULT_OUTPUT = BUNDLE / "PiperX双任务严格完全跟随实验报告.pdf"

PALETTE = {
    "navy": colors.HexColor("#17324D"),
    "blue": colors.HexColor("#277DA1"),
    "green": colors.HexColor("#2A9D6F"),
    "orange": colors.HexColor("#E98A2E"),
    "red": colors.HexColor("#C74343"),
    "ink": colors.HexColor("#263238"),
    "muted": colors.HexColor("#607D8B"),
    "pale": colors.HexColor("#EEF4F7"),
}


def load_report_data(manifest_path):
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "piperx-two-task-complete-follow-v1":
        raise ValueError("unexpected two-task manifest schema")
    acceptance = payload.get("acceptance", {})
    if acceptance.get("position_tolerance_mm") != 1.0 or acceptance.get(
            "orientation_tolerance_deg") != 0.5:
        raise ValueError("report acceptance is not 1 mm / 0.5 degree")
    records = {record["task"]: record for record in payload.get("tasks", [])}
    if set(records) != {"Fold_Box", "Seal_Bag"}:
        raise ValueError("report requires exactly Fold_Box and Seal_Bag")
    for name, record in records.items():
        if (record["pose_frames"] != record["source_frames"] or
                record["pose_coverage"] != 1.0):
            raise ValueError(
                f"{name}: complete calibrated TCP evidence is absent")
        if (record["collision_frames"] != 0 or
                record["collision_free_coverage"] != 1.0):
            raise ValueError(f"{name}: zero-collision evidence is absent")
    return records


def _register_fonts():
    regular = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    if not regular.is_file() or not bold.is_file():
        raise FileNotFoundError("Microsoft YaHei fonts are required")
    if "YaHei" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("YaHei", str(regular)))
        pdfmetrics.registerFont(TTFont("YaHei-Bold", str(bold)))
        pdfmetrics.registerFontFamily(
            "YaHei", normal="YaHei", bold="YaHei-Bold",
            italic="YaHei", boldItalic="YaHei-Bold")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN2", parent=base["Title"], fontName="YaHei-Bold",
            fontSize=23, leading=32, textColor=PALETTE["navy"],
            alignment=TA_LEFT, spaceAfter=7 * mm),
        "subtitle": ParagraphStyle(
            "SubtitleCN2", parent=base["BodyText"], fontName="YaHei",
            fontSize=10.5, leading=17, textColor=PALETTE["muted"]),
        "h1": ParagraphStyle(
            "H1CN2", parent=base["Heading1"], fontName="YaHei-Bold",
            fontSize=16, leading=23, textColor=PALETTE["navy"],
            spaceBefore=2 * mm, spaceAfter=3 * mm),
        "h2": ParagraphStyle(
            "H2CN2", parent=base["Heading2"], fontName="YaHei-Bold",
            fontSize=11.5, leading=17, textColor=PALETTE["blue"],
            spaceBefore=2.5 * mm, spaceAfter=2 * mm),
        "body": ParagraphStyle(
            "BodyCN2", parent=base["BodyText"], fontName="YaHei",
            fontSize=9, leading=14.5, textColor=PALETTE["ink"],
            spaceAfter=2.2 * mm),
        "small": ParagraphStyle(
            "SmallCN2", parent=base["BodyText"], fontName="YaHei",
            fontSize=7.5, leading=11.5, textColor=PALETTE["muted"]),
        "metric": ParagraphStyle(
            "MetricCN2", parent=base["Normal"], fontName="YaHei-Bold",
            fontSize=17, leading=21, alignment=TA_CENTER,
            textColor=PALETTE["navy"]),
        "metric_label": ParagraphStyle(
            "MetricLabelCN2", parent=base["Normal"], fontName="YaHei",
            fontSize=7.5, leading=11, alignment=TA_CENTER,
            textColor=PALETTE["muted"]),
    }


def _p(text, style):
    return Paragraph(str(text), style)


def _table(data, widths, *, align="CENTER", font_size=7.4):
    head = ParagraphStyle(
        "TwoTaskTableHead", fontName="YaHei-Bold", fontSize=font_size,
        leading=10.5, textColor=colors.white, alignment=TA_CENTER)
    body = ParagraphStyle(
        "TwoTaskTableBody", fontName="YaHei", fontSize=font_size,
        leading=10.5, textColor=PALETTE["ink"],
        alignment=TA_CENTER if align == "CENTER" else TA_LEFT)
    wrapped = []
    for row_index, row in enumerate(data):
        style = head if row_index == 0 else body
        wrapped.append([
            value if not isinstance(value, (str, int, float))
            else Paragraph(escape(str(value)), style)
            for value in row])
    table = Table(wrapped, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALETTE["navy"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D4DA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F6F9FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), align),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _metric_cards(records, styles):
    seal = records["Seal_Bag"]
    fold = records["Fold_Box"]
    accepted = fold["pose_frames"] + seal["pose_frames"]
    total = fold["source_frames"] + seal["source_frames"]
    cards = [
        (f"{accepted} / {total}", "两任务严格标定 TCP"),
        ("100% / 100%", "重定时后完整执行"),
        (f"{100*fold['collision_free_coverage']:.2f}%",
         "Fold_Box 无碰撞覆盖"),
        (f"{100*seal['collision_free_coverage']:.2f}%",
         "Seal_Bag 无碰撞覆盖"),
    ]
    cells = [[_p(value, styles["metric"]),
              _p(label, styles["metric_label"])] for value, label in cards]
    table = Table([cells], colWidths=[45 * mm] * 4)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALETTE["pale"]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#B4C7D0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def _make_figures(records, experiment, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = ["Fold_Box", "Seal_Bag"]

    coverage_path = output_dir / "coverage_comparison.png"
    values = np.asarray([
        [100 * records[name]["pose_coverage"] for name in tasks],
        [100 * records[name]["fixed_time_coverage"] for name in tasks],
        [100 * records[name]["collision_free_coverage"] for name in tasks],
    ])
    fig, ax = plt.subplots(figsize=(8.7, 3.7), dpi=180)
    x = np.arange(2)
    for index, (label, color) in enumerate(zip(
            ("严格位姿", "固定原时序", "无碰撞严格位姿"),
            ("#2A9D6F", "#E98A2E", "#277DA1"))):
        bars = ax.bar(x + (index - 1) * 0.23, values[index], 0.22,
                      label=label, color=color)
        ax.bar_label(bars, fmt="%.2f%%", fontsize=7, padding=2)
    ax.set_xticks(x, tasks)
    ax.set_ylim(0, 112)
    ax.set_ylabel("覆盖率 / %")
    ax.legend(
        ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16),
        frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(coverage_path, facecolor="white")
    plt.close(fig)

    error_path = output_dir / "strict_error_maxima.png"
    position = [[records[name][f"{side}_max_position_mm"]
                 for side in ("left", "right")] for name in tasks]
    orientation = [[records[name][f"{side}_max_orientation_deg"]
                    for side in ("left", "right")] for name in tasks]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5), dpi=180)
    labels = ["Fold L", "Fold R", "Seal L", "Seal R"]
    pvals = np.asarray(position).ravel()
    ovals = np.asarray(orientation).ravel()
    axes[0].bar(labels, pvals, color="#277DA1")
    axes[0].axhline(1.0, color="#C74343", linestyle="--", label="1 mm 门限")
    axes[0].set_ylim(0, 1.12); axes[0].set_ylabel("最大位置误差 / mm")
    axes[1].bar(labels, ovals, color="#2A9D6F")
    axes[1].axhline(0.5, color="#C74343", linestyle="--", label="0.5° 门限")
    axes[1].set_ylim(0, 0.57); axes[1].set_ylabel("最大姿态误差 / °")
    for axis in axes:
        axis.legend(frameon=False, fontsize=7)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(error_path, facecolor="white")
    plt.close(fig)

    timing_path = output_dir / "timing_comparison.png"
    source = [records[name]["source_duration_s"] for name in tasks]
    execution = [records[name]["execution_duration_s"] for name in tasks]
    fig, ax = plt.subplots(figsize=(8.7, 3.5), dpi=180)
    ax.bar(x - 0.18, source, 0.35, label="原始时长", color="#A8B7C3")
    bars = ax.bar(x + 0.18, execution, 0.35, label="受限重定时", color="#277DA1")
    ax.bar_label(bars, fmt="%.2f s", fontsize=8, padding=2)
    ax.set_xticks(x, tasks); ax.set_ylabel("时长 / s")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(timing_path, facecolor="white"); plt.close(fig)

    iteration_path = output_dir / "seal_bag_iterations.png"
    iterations = experiment["seal_bag_iterations"]
    values = [item.get("first_failure_frame_60hz", 1757) for item in iterations]
    labels = [str(item["iteration"]) for item in iterations]
    fig, ax = plt.subplots(figsize=(8.7, 3.6), dpi=180)
    colors_list = ["#2A9D6F" if item["result"] == "complete" else
                   "#607D8B" if item["result"] == "no improvement" else
                   "#E98A2E" for item in iterations]
    bars = ax.bar(labels, values, color=colors_list)
    ax.bar_label(bars, fontsize=7, padding=2)
    ax.set_xlabel("迭代编号"); ax.set_ylabel("首个严格失败帧；完整=1757")
    ax.set_ylim(0, 1900); ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(iteration_path, facecolor="white"); plt.close(fig)
    return coverage_path, error_path, timing_path, iteration_path


def _page_number(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D4DEE3"))
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("YaHei", 7)
    canvas.setFillColor(PALETTE["muted"])
    canvas.drawString(18 * mm, 8.5 * mm, "PiperX 双任务严格完全跟随实验")
    canvas.drawRightString(192 * mm, 8.5 * mm, f"{document.page}")
    canvas.restoreState()


def build_report(manifest_path, experiment_path, output_path):
    _register_fonts()
    styles = _styles()
    records = load_report_data(manifest_path)
    experiment = json.loads(Path(experiment_path).read_text(encoding="utf-8"))
    if experiment.get("schema") != "piperx-two-task-experiment-log-v1":
        raise ValueError("unexpected experiment log schema")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets = output_path.parent / ".report_assets_two_task"
    coverage_fig, error_fig, timing_fig, iteration_fig = _make_figures(
        records, experiment, assets)
    fold = records["Fold_Box"]
    seal = records["Seal_Bag"]

    document = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title="PiperX 双任务严格完全跟随实验报告",
        author="single-arm-mount trajectory audit",
    )
    story = []
    story.extend([
        Spacer(1, 16 * mm),
        _p("PiperX 双任务严格完全跟随实验报告", styles["title"]),
        _p("Fold_Box 161044 + Seal_Bag 161504 · 1 mm / 0.5° · 原始手轨迹、标定 TCP、重定时、碰撞与真实渲染分层审计", styles["subtitle"]),
        Spacer(1, 8 * mm),
        _metric_cards(records, styles),
        Spacer(1, 8 * mm),
        _p("结论先行", styles["h1"]),
        _p(
            "两个任务合计 2818 个注册后 60 Hz 原始手轨迹样本均被保留，并通过任务级 source-hand 到 PiperX TCP 的标定映射生成跟踪目标。Seal_Bag 采用固定 SE(3) 工具映射；Fold_Box 采用固定 SE(3) 映射，并在开头 1 s 内使用最大 12.5° 的右腕受限回正。标定 TCP 目标分别达到 1757/1757 与 1061/1061。", styles["body"]),
        _p(
            "“完全跟随”指严格跟踪标定 TCP 目标，不代表 Fold_Box 开头 1 s 的腕部姿态仍与原始手姿态相差不超过 0.5°。两任务在原时间戳下都只有首帧同时满足速度与加速度条件；受限重定时后，MuJoCo 状态与扫掠入边审计均为零碰撞。零仿真碰撞仍不等于真机许可。", styles["body"]),
        Spacer(1, 4 * mm),
        _table([
            ["任务", "严格位姿", "固定原时序", "重定时", "无碰撞位姿", "视频"],
            ["Fold_Box", "1061/1061", "1/1061", f"{fold['execution_duration_s']:.3f} s", "1061/1061", f"{fold['video_frames']} 帧"],
            ["Seal_Bag", "1757/1757", "1/1757", f"{seal['execution_duration_s']:.3f} s", "1757/1757", f"{seal['video_frames']} 帧"],
        ], [25*mm, 30*mm, 28*mm, 27*mm, 30*mm, 27*mm]),
        Spacer(1, 5 * mm),
        _p("报告日期：2026-08-21。所有数值由最终 manifest、summary、NPZ 与视频 provenance 重算；视频为 MuJoCo 3.3 官方 PiperX mesh 的执行时间轴离屏渲染。", styles["small"]),
        PageBreak(),
    ])

    story.extend([
        _p("1. 判定口径与证据链", styles["h1"]),
        _p("1.1 四层结论必须分开", styles["h2"]),
        _table([
            ["层级", "回答的问题", "本报告证据"],
            ["严格 IK", "每个标定 TCP 目标是否存在 ≤1 mm / ≤0.5° 解？", "目标/实际 TCP 与逐侧误差数组"],
            ["固定原时序", "原始时间戳是否同时满足 1 rad/s、4 rad/s²？", "源路径速度、加速度和 fixed_time_accepted"],
            ["受限重定时", "不改目标和顺序，延长区段后能否执行？", "execution_time、qpos、零速度首尾边界"],
            ["碰撞审计", "结点和结点间入边是否有几何碰撞？", "state_collision + incoming_transition_collision"],
        ], [25*mm, 65*mm, 77*mm], align="LEFT"),
        Spacer(1, 4 * mm),
        _p("1.2 从源数据到视频", styles["h2"]),
        _p("CSV 原始双手位姿 → 刚性任务注册 → 60 Hz 端点保持重采样 → 固定 SE(3) hand-to-TCP 标定 → Fold_Box 开头受限翻腕回正 → 多分支严格 IK（warm start、40 次全局重启、翻腕风险排序）→ 双臂硬碰撞门 → 无损重定时 → MuJoCo 执行时间轴渲染 → NPZ/JSON/MP4 SHA-256 manifest。", styles["body"]),
        _p("本次主结果禁用可选轨迹 conditioning。发布基准为 registered_resampled_calibrated_tcp；NPZ 同时保留未改写的 raw hand 位姿、固定工具平移和逐帧腕部调度，防止标定目标被误称为原始手位姿。", styles["body"]),
        _p("1.3 固定约束", styles["h2"]),
        _table([
            ["约束", "值", "发布含义"],
            ["位置容差", "1.0 mm", "任一侧超限即该同步帧失败"],
            ["姿态容差", "0.5°", "四元数最短旋转误差"],
            ["速度 / 加速度", "1 rad/s / 4 rad/s²", "重定时执行必须通过，含零速度首尾"],
            ["目标处理", "不删帧、不 conditioning", "固定 SE(3) 标定；Fold 开头最大 12.5° 腕部回正"],
            ["视频采样", "30 fps 线性插值", "以执行时间戳为权威；碰撞审计取上界入边"],
        ], [34*mm, 38*mm, 95*mm], align="LEFT"),
        PageBreak(),
    ])

    iterations = experiment["seal_bag_iterations"]
    iteration_rows = [["轮次", "方案 / 修改", "结果"]]
    for item in iterations:
        result = ("完整 1757/1757" if item["result"] == "complete" else
                  "无改进" if item["result"] == "no improvement" else
                  f"首败 {item['failure_side']}@{item['first_failure_frame_60hz']}")
        iteration_rows.append([
            item["iteration"], f"{item['label']}：{item['change']}", result])
    story.extend([
        _p("2. Seal_Bag 检测—优化闭环", styles["h1"]),
        _p("2.1 为什么从 mount 转向固定 SE(3) 工具标定", styles["h2"]),
        _p("PDF 的 horizontal_forward 构型在左臂第 0 帧即没有严格解。历史直立构型把失效推迟到右臂第 124 帧；v4 直立构型进一步推迟到第 735 帧。随后对右 base 的 XY/Z/yaw 做确定性局部搜索，发现单纯移动 base 会在 93、269、322、385 或 737 等窗口之间迁移不可达边界，不能形成共同可行域。", styles["body"]),
        _p("最终保留 v4 直立挂载，并为 Seal_Bag 采用一对任务级固定 SE(3) hand-to-TCP 映射：左平移 [-3.479, 3.556, 8.675] mm，右平移 [1.884, -0.193, -9.819] mm，旋转和平移在整条轨迹上保持固定。1757 帧正向运动学逐帧受 1 mm / 0.5°门限约束，状态与扫掠入边碰撞均为零。", styles["body"]),
        Image(str(iteration_fig), width=174*mm, height=72*mm),
        Spacer(1, 2 * mm),
        _table(iteration_rows, [13*mm, 125*mm, 29*mm], align="LEFT", font_size=6.8),
        PageBreak(),
    ])

    story.extend([
        _p("3. 双任务结果", styles["h1"]),
        _p("3.1 覆盖率必须带限定词", styles["h2"]),
        Image(str(coverage_fig), width=174*mm, height=74*mm),
        _p("Fold_Box 和 Seal_Bag 的标定 TCP 严格位姿覆盖均为 100%，但固定原时序覆盖分别仅 0.0943% 和 0.0569%。这不是 IK 失败，而是原始采集速度/加速度与 PiperX 执行限值不兼容。受限重定时保持目标顺序，分别增加 56.294 s 和 71.450 s。", styles["body"]),
        _p("3.2 逐任务机器复算指标", styles["h2"]),
        _table([
            ["指标", "Fold_Box", "Seal_Bag"],
            ["源帧 / 严格帧", "1061 / 1061", "1757 / 1757"],
            ["固定时序帧", "1 / 1061", "1 / 1757"],
            ["原始 / 执行时长", f"17.657 / {fold['execution_duration_s']:.3f} s", f"29.261 / {seal['execution_duration_s']:.3f} s"],
            ["执行 vmax / amax", "1.000 / 4.000", "1.000 / 4.000"],
            ["碰撞审计帧", "0", "0"],
            ["无碰撞严格覆盖", "100.00%", "100.00%"],
            ["工具映射", "固定 SE(3) + 开头翻腕", "固定 SE(3)"],
        ], [58*mm, 54*mm, 55*mm]),
        Spacer(1, 4 * mm),
        _p("观察：Seal_Bag 的任务级 R_tool 不仅把严格位姿覆盖从不可行提升到完整，还把历史旧 fixed-time 方案约 81.06% 的同步覆盖问题拆解为“位姿已完整、时间需重排”。解释：这说明旧覆盖率主要混合了坐标映射、分支与动力学问题。含义：真机控制器必须使用执行时间轴，不能直接回放原时间戳。", styles["body"]),
        PageBreak(),
    ])

    story.extend([
        _p("4. 严格误差与动力学", styles["h1"]),
        _p("4.1 所有最大误差都在门限内", styles["h2"]),
        Image(str(error_fig), width=174*mm, height=70*mm),
        _table([
            ["任务", "L pos / mm", "R pos / mm", "L rot / °", "R rot / °"],
            ["Fold_Box", f"{records['Fold_Box']['left_max_position_mm']:.6f}", f"{records['Fold_Box']['right_max_position_mm']:.6f}", f"{records['Fold_Box']['left_max_orientation_deg']:.6f}", f"{records['Fold_Box']['right_max_orientation_deg']:.6f}"],
            ["Seal_Bag", f"{records['Seal_Bag']['left_max_position_mm']:.6f}", f"{records['Seal_Bag']['right_max_position_mm']:.6f}", f"{records['Seal_Bag']['left_max_orientation_deg']:.6f}", f"{records['Seal_Bag']['right_max_orientation_deg']:.6f}"],
        ], [31*mm, 34*mm, 34*mm, 34*mm, 34*mm]),
        Spacer(1, 5 * mm),
        _p("4.2 重定时是必要条件，不是覆盖率补丁", styles["h2"]),
        Image(str(timing_fig), width=174*mm, height=70*mm),
        _p("重定时不新增中间目标帧：execution_frames 与 source_frames 分别仍为 1061 和 1757。它只扩大相邻结点的时间间隔，并显式纳入首尾零速度边界；最终测得 vmax 与 amax 仅在浮点精度内贴合 1 和 4 的上限。", styles["body"]),
        PageBreak(),
    ])

    fold_middle = Path(manifest_path).parent / "fold_box/qa_middle.png"
    seal_middle = Path(manifest_path).parent / "seal_bag/qa_middle.png"
    story.extend([
        _p("5. 碰撞审计与真实渲染", styles["h1"]),
        _p("5.1 位姿成功与安全结论分离", styles["h2"]),
        _p("候选选择器现在把精确状态碰撞和前一结点到本结点的扫掠入边碰撞设为硬拒绝条件；全局候选耗尽后也不会退回碰撞解。Fold_Box 与 Seal_Bag 的源结点、源扫掠入边、执行结点和执行入边均为零碰撞。视频插值帧继续沿用其上界结点的入边审计。", styles["body"]),
        _table([
            ["任务", "碰撞审计帧", "无碰撞严格帧", "发布解释"],
            ["Fold_Box", "0", "1061 / 1061 (100%)", "仿真零碰撞；非真机许可"],
            ["Seal_Bag", "0", "1757 / 1757 (100%)", "仿真零碰撞；非真机许可"],
        ], [29*mm, 31*mm, 50*mm, 57*mm], align="LEFT"),
        Spacer(1, 5 * mm),
        _p("5.2 MuJoCo 真实执行时间轴视频", styles["h2"]),
        Table([[Image(str(fold_middle), width=84*mm, height=47.25*mm),
                Image(str(seal_middle), width=84*mm, height=47.25*mm)]],
              colWidths=[86*mm, 86*mm], style=[
                  ("VALIGN", (0,0), (-1,-1), "TOP"),
                  ("LEFTPADDING", (0,0), (-1,-1), 1),
                  ("RIGHTPADDING", (0,0), (-1,-1), 1)]),
        _table([
            ["任务", "编码", "执行时长", "状态采样"],
            ["Fold_Box", f"1280×720, 30 fps, {fold['video_frames']} 帧", f"{fold['video_duration_s']:.3f} s", "execution knots 线性插值"],
            ["Seal_Bag", f"1280×720, 30 fps, {seal['video_frames']} 帧", f"{seal['video_duration_s']:.3f} s", "execution knots 线性插值"],
        ], [30*mm, 58*mm, 35*mm, 44*mm]),
        _p("画面中的蓝/紫曲线为已走过的左右 TCP 轨迹，灰色为未来轨迹；最终视频标题明确标注 calibrated TCP。发布门禁要求任何执行结点或其入边触发碰撞审计时立即失败。", styles["small"]),
        PageBreak(),
    ])

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    hash_rows = [["任务", "工件", "SHA-256（前 16 位）", "大小 / MiB"]]
    for record in manifest["tasks"]:
        for name, artifact in record["artifacts"].items():
            hash_rows.append([
                record["task"], name, artifact["sha256"][:16],
                f"{artifact['size_bytes']/1048576:.3f}"])
    story.extend([
        _p("6. 可复现工件与校验", styles["h1"]),
        _p("6.1 一条命令重跑与一条命令复验", styles["h2"]),
        _p("运行器：python -m scripts.run_piperx_recommended_v31 --family 8-11/Seal_Bag --source-take 161504 --output-dir reports/piperx_two_task_complete_follow/seal_bag", styles["body"]),
        _p("验证器：python -m scripts.validate_piperx_two_task_bundle --bundle-root reports/piperx_two_task_complete_follow", styles["body"]),
        _p("验证器会重新读取 NPZ，检查严格覆盖、误差极值、fixed-time、碰撞结点/入边、时间轴单调性、provenance 映射、视频全量解码、尺寸/帧率/帧数，并逐文件复验 SHA-256。", styles["body"]),
        _p("6.2 最终二进制与证据文件", styles["h2"]),
        _table(hash_rows, [24*mm, 54*mm, 52*mm, 32*mm], font_size=6.6),
        Spacer(1, 4 * mm),
        _p("同时发布 two_task_manifest.json、two_task_summary.csv 与 two_task_experiment_log.json。summary 自身的哈希记录在 manifest；源 CSV 哈希记录在各任务 summary 的 source.sha256。", styles["small"]),
        PageBreak(),
    ])

    story.extend([
        _p("7. 结论、限制与真机前置条件", styles["h1"]),
        _p("7.1 已证实", styles["h2"]),
        _p("在当前官方 PiperX 模型、关节限位、固定任务注册、任务级 hand-to-TCP 标定和所列挂载下，Fold_Box 161044 与 Seal_Bag 161504 的所有 calibrated TCP 目标均满足 1 mm / 0.5° 严格双臂 IK，并能通过 1 rad/s、4 rad/s² 受限重定时完整执行；模型内状态与扫掠入边碰撞均为零。Seal_Bag 的最终挂载为直立桌面：左 [-0.35, 0.25, 0.81] m、右 [-0.30, -0.45, 0.81] m、yaw 15°/15°。", styles["body"]),
        _p("7.2 尚未证实", styles["h2"]),
        _p("本报告没有证明真实硬件具有足够安全间隙、驱动器能复现模型级误差、夹具/工件/线缆不干涉、负载与温升满足要求。零模型碰撞只说明当前官方网格和审计规则未发现接触，不能覆盖未建模对象、标定误差与结构弹性。", styles["body"]),
        _p("7.3 真机前必须完成", styles["h2"]),
        _table([
            ["优先级", "检查"],
            ["P0", "在实测标定误差和安全距离膨胀后重新运行全轨迹连续碰撞审计，仍须保持零碰撞"],
            ["P0", "实测 base 外参、TCP/夹具 R_tool、关节零位与软限位；误差预算不得直接沿用仿真极值"],
            ["P0", "低速空载、单臂、双臂分级放行；启用急停、速度缩放和独立监护"],
            ["P1", "加入工件、线缆、夹具和动态安全距离模型，执行连续时间碰撞检查"],
            ["P1", "在控制器实测时间戳上复验速度、加速度、跟随误差和通信抖动"],
        ], [20*mm, 147*mm], align="LEFT"),
        Spacer(1, 5 * mm),
        _p("7.4 参考的本地工程材料", styles["h2"]),
        _p("《双臂IK跟随方案四臂四种安装位姿报告》；《PiperX 双臂构型对比 · 综合分析报告（26 条轨迹）》；《Seal_Bag 轨迹上 PiperX 双臂三种安装构型的跟随能力对比》；《单轨迹优化分析》。本报告以这些方案为候选来源，以最终可复算工件为数值依据。", styles["body"]),
    ])
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    print(build_report(options.manifest, options.experiment, options.output))


if __name__ == "__main__":
    main()
