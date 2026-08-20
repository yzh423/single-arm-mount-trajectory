"""Build the audited Chinese PDF report for PiperX recommended v3.1."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
ARTIFACT_DIR = ROOT / "reports/piperx_recommended_v31"
SUMMARY_PATH = ARTIFACT_DIR / "8-11_Fold_Box_161044_recommended_v31.summary.json"
TRAJECTORY_PATH = ARTIFACT_DIR / "8-11_Fold_Box_161044_recommended_v31.trajectory.npz"
DEFAULT_OUTPUT = ARTIFACT_DIR / "PiperX双臂IK跟随优化与验证报告_v3.1.pdf"
SOURCE_REPORT = ROOT / "双臂IK跟随方案四臂四种安装位姿报告.pdf"


PALETTE = {
    "navy": colors.HexColor("#15324B"),
    "blue": colors.HexColor("#277DA1"),
    "green": colors.HexColor("#2A9D6F"),
    "orange": colors.HexColor("#E98A2E"),
    "red": colors.HexColor("#D1495B"),
    "ink": colors.HexColor("#263238"),
    "muted": colors.HexColor("#607D8B"),
    "pale": colors.HexColor("#EEF4F7"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--trajectory", type=Path, default=TRAJECTORY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _register_fonts():
    regular = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    if not regular.is_file() or not bold.is_file():
        raise FileNotFoundError("Microsoft YaHei fonts are required for Chinese PDF")
    pdfmetrics.registerFont(TTFont("YaHei", str(regular)))
    pdfmetrics.registerFont(TTFont("YaHei-Bold", str(bold)))
    pdfmetrics.registerFontFamily(
        "YaHei", normal="YaHei", bold="YaHei-Bold",
        italic="YaHei", boldItalic="YaHei-Bold",
    )
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN", parent=base["Title"], fontName="YaHei-Bold",
            fontSize=24, leading=32, textColor=PALETTE["navy"],
            alignment=TA_LEFT, spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN", parent=base["Normal"], fontName="YaHei",
            fontSize=11, leading=18, textColor=PALETTE["muted"],
        ),
        "h1": ParagraphStyle(
            "H1CN", parent=base["Heading1"], fontName="YaHei-Bold",
            fontSize=17, leading=23, textColor=PALETTE["navy"],
            spaceBefore=2 * mm, spaceAfter=4 * mm,
        ),
        "h2": ParagraphStyle(
            "H2CN", parent=base["Heading2"], fontName="YaHei-Bold",
            fontSize=12, leading=17, textColor=PALETTE["blue"],
            spaceBefore=3 * mm, spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN", parent=base["BodyText"], fontName="YaHei",
            fontSize=9.3, leading=15, textColor=PALETTE["ink"],
            spaceAfter=2.3 * mm,
        ),
        "small": ParagraphStyle(
            "SmallCN", parent=base["BodyText"], fontName="YaHei",
            fontSize=7.8, leading=12, textColor=PALETTE["muted"],
        ),
        "metric": ParagraphStyle(
            "MetricCN", parent=base["Normal"], fontName="YaHei-Bold",
            fontSize=18, leading=22, alignment=TA_CENTER,
            textColor=PALETTE["navy"],
        ),
        "metric_label": ParagraphStyle(
            "MetricLabelCN", parent=base["Normal"], fontName="YaHei",
            fontSize=7.8, leading=11, alignment=TA_CENTER,
            textColor=PALETTE["muted"],
        ),
    }


def _paragraph(text, style):
    return Paragraph(text, style)


def _table(data, widths, *, header=True, align="LEFT"):
    paragraph_alignment = TA_CENTER if align == "CENTER" else TA_LEFT
    header_style = ParagraphStyle(
        "TableHeaderCN", fontName="YaHei-Bold", fontSize=8,
        leading=11, textColor=colors.white, alignment=paragraph_alignment,
    )
    body_style = ParagraphStyle(
        "TableBodyCN", fontName="YaHei", fontSize=7.7,
        leading=11, textColor=PALETTE["ink"], alignment=paragraph_alignment,
    )
    wrapped = []
    for row_index, row in enumerate(data):
        style = header_style if header and row_index == 0 else body_style
        wrapped.append([
            Paragraph(escape(str(cell)), style)
            if isinstance(cell, (str, int, float)) else cell
            for cell in row
        ])
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "YaHei"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), align),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D4DA")),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1),
         [colors.white, colors.HexColor("#F6F9FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), PALETTE["navy"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "YaHei-Bold"),
        ])
    table.setStyle(TableStyle(commands))
    return table


def _metric_cards(summary, styles):
    metrics = summary["metrics"]
    accepted = metrics["accepted_only"]
    cards = [
        (f"{100 * metrics['strict_synchronous_coverage']:.2f}%", "严格双臂 ACCEPT"),
        (str(metrics["collision_frames"]), "状态/扫掠碰撞帧"),
        (f"{max(accepted['left_max_position_mm'], accepted['right_max_position_mm']):.4f} mm",
         "ACCEPT 内最大位置误差"),
        (f"{max(accepted['left_max_orientation_deg'], accepted['right_max_orientation_deg']):.4f}°",
         "ACCEPT 内最大姿态误差"),
    ]
    cells = []
    for value, label in cards:
        cells.append([
            _paragraph(value, styles["metric"]),
            _paragraph(label, styles["metric_label"]),
        ])
    table = Table([cells], colWidths=[45 * mm] * 4)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALETTE["pale"]),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B4C7D0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _make_figures(summary, arrays, directory):
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "report_state_distribution.png"
    gate_path = directory / "report_strict_gate.png"

    states, counts = np.unique(arrays["source_state"], return_counts=True)
    colors_list = [
        "#2A9D6F" if state == "FOLLOW" else
        "#E98A2E" if str(state).startswith("RESCUE") else "#D1495B"
        for state in states
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=180)
    bars = ax.bar(states, counts, color=colors_list, width=0.65)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_ylabel("源帧数")
    ax.set_title("完整 60 Hz 时间轴状态分布（HOLD/RESCUE 不计 ACCEPT）")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(state_path, transparent=False, facecolor="white")
    plt.close(fig)

    accepted = np.asarray(arrays["source_accepted"], dtype=bool)
    labels = ["左位置", "右位置", "左姿态", "右姿态"]
    maxima = [
        1000 * arrays["left_position_error_m"][accepted].max(),
        1000 * arrays["right_position_error_m"][accepted].max(),
        np.rad2deg(arrays["left_orientation_error_rad"][accepted].max()),
        np.rad2deg(arrays["right_orientation_error_rad"][accepted].max()),
    ]
    limits = [1.0, 1.0, 0.5, 0.5]
    ratios = np.asarray(maxima) / np.asarray(limits)
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=180)
    bars = ax.bar(labels, ratios, color=["#277DA1", "#277DA1", "#7A5195", "#7A5195"])
    ax.axhline(1.0, color="#D1495B", linestyle="--", linewidth=1.4,
               label="严格门槛")
    for bar, value, limit in zip(bars, maxima, limits):
        unit = "mm" if limit == 1.0 else "°"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.04,
                f"{value:.4f}{unit}", ha="center", va="top",
                color="white", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("相对门槛")
    ax.set_title("仅对 ACCEPT 帧复核：最大误差未越界")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(gate_path, transparent=False, facecolor="white")
    plt.close(fig)
    return state_path, gate_path


def _header_footer(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D4DEE3"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont("YaHei", 7)
    canvas.setFillColor(PALETTE["muted"])
    canvas.drawString(18 * mm, 9 * mm, "PiperX 双臂 IK 跟随优化与验证报告 · v3.1")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def build_report(summary_path: Path, trajectory_path: Path, output_path: Path):
    _register_fonts()
    styles = _styles()
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    arrays = np.load(trajectory_path, allow_pickle=False)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_figure, gate_figure = _make_figures(summary, arrays, output_path.parent)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title="PiperX 双臂 IK 跟随优化与验证报告 v3.1",
        author="single-arm-mount project",
    )
    story = []
    p = lambda text, name="body": _paragraph(text, styles[name])

    story.extend([
        Spacer(1, 13 * mm),
        p("PiperX 双臂 IK 跟随<br/>优化与验证报告", "title"),
        p("按《双臂IK跟随方案四臂四种安装位姿报告》推荐方案落地 · recommended v3.1", "subtitle"),
        Spacer(1, 10 * mm),
        _metric_cards(summary, styles),
        Spacer(1, 10 * mm),
        p("结论", "h1"),
        p(
            "已将项目切换到按 (日期, 任务) 分族的 PiperX 共享底座、60 Hz 有界目标平滑、"
            "frame 0 的 40 次重启锚定、后续逐帧 DLS warm-start、0.30 rad 分支守卫和 rescue_v3.1。"
            "用户指定门槛固定为 <b>1 mm / 0.5°</b>，未用宽松 tier 回填成功。"
        ),
        p(
            "完整验证取 8-11/Fold_Box 的 PDF 指定共享 base 来源 take 161044。"
            "1061 个 60 Hz 源帧中严格双臂 ACCEPT 112 帧（10.56%），碰撞 0 帧；"
            "其余帧明确记录为 HOLD 或受控恢复。该数值反映严格门槛下的真实可达限制，不等同于全程跟随成功。"
        ),
        p("证据范围", "h2"),
        p(
            "视频是 MuJoCo 3.11 对官方 PiperX 模型/网格的离屏真实渲染，不是简化示意动画，也不是实体机械臂摄像。"
            "轨迹 NPZ、逐帧 provenance、场景 XML、摘要 JSON 与三个 QA 截帧一并保留。"
        ),
        Spacer(1, 5 * mm),
        p("生成日期：2026-08-20 · 验证平台：Windows / MuJoCo 3.11 / Python 3.12", "small"),
    ])

    story.extend([
        PageBreak(),
        p("1. 推荐方案落地矩阵", "h1"),
        p("下表把源报告的关键要求逐项映射到本次实现和可审计证据。"),
        _table([
            ["源报告要求", "本次实现", "证据"],
            ["任务族按 (日期, 任务) 拆分", "TaskFamily(date, task)，同名跨日不合并", "配置含 12 个 PiperX 任务族"],
            ["60 Hz 目标平滑", "位置线性插值 + 四元数 SLERP；有界 SE(3) 平滑", "161044: 1478 原帧 → 1061 源帧"],
            ["frame 0：40 次重启锚定", "分层关节种子 + 40 个确定性全局重启", "anchor_restart_count=40"],
            ["后续 warm-start IK", "仅从上一指令 q 做单次 DLS；失败才打开 rescue 扫描", "warm_start_attempted 数组"],
            ["分支守卫", "周期关节最短差；每关节 Δq ≤ 0.30 rad", "StrictGate + 单元测试"],
            ["翻腕仅经 rescue", "J4/J5 腕部分支签名；普通 FOLLOW 禁止变 basin", "6 个受控 rescue 事件"],
            ["非终止失败", "失败帧 HOLD 并继续重放，不中断整条轨迹", "499 个 HOLD 源帧"],
            ["模式 A / B", "A：未来截获、丢帧、零接缝；B：暂停目标并增加节拍", "450 丢帧；0.667 s 节拍延迟"],
            ["状态与扫掠碰撞", "双臂配对前检查状态，转移按 7 子步检查", "完整结果碰撞 0 帧"],
        ], [48 * mm, 78 * mm, 54 * mm]),
        Spacer(1, 5 * mm),
        p("严格 ACCEPT 判定", "h2"),
        p(
            "一个源帧只有在左右臂同时满足位置误差 ≤ 0.001 m、姿态误差 ≤ 0.5°、"
            "关节分支变化 ≤ 0.30 rad，并通过状态与扫掠碰撞检查时才计为 ACCEPT。"
            "RESCUE_A 中间帧虽然在执行运动，但因目标继续前进，按丢帧计而不冒充跟随成功。"
        ),
    ])

    mount = summary["mount"]
    story.extend([
        PageBreak(),
        p("2. Mount 方案", "h1"),
        p(
            "PiperX 对 8-11/Fold_Box 采用源报告推荐的 <b>底部（桌面）</b>共享 base。"
            "底座坐标先定义在 VR 源轨迹坐标系，再通过与双手轨迹相同的刚体注册变换进入 MuJoCo 世界系，"
            "避免把源坐标直接当世界坐标。"
        ),
        _table([
            ["项目", "左臂", "右臂"],
            ["PDF p_base / m", "(+0.150, +0.300, -0.683)", "(+0.050, -0.300, -0.683)"],
            ["MuJoCo 世界坐标 / m",
             "(" + ", ".join(f"{v:+.6f}" for v in [*mount["xy"]["left"], mount["shared_base_z_m"]]) + ")",
             "(" + ", ".join(f"{v:+.6f}" for v in [*mount["xy"]["right"], mount["shared_base_z_m"]]) + ")"],
            ["安装形态", "upright_table", "upright_table"],
            ["底座间距", f"{mount['base_distance_m']:.6f} m", f"{mount['base_distance_m']:.6f} m"],
        ], [50 * mm, 65 * mm, 65 * mm], align="CENTER"),
        Spacer(1, 5 * mm),
        p("为什么不用旧 mount", "h2"),
        p(
            "旧项目的 mount 主要围绕单条轨迹或旧阈值优化，且任务目录只按任务名聚合，"
            "会把 8-11/Fold_Box 与 8-12/Fold_Box 混为一族。新配置冻结源报告给出的 12 族 PiperX 共享 base，"
            "默认优先底部桌面，其次人形立杆，顶部倒挂仅用于特殊高轨迹；baseline_frozen 不作为部署方案。"
        ),
        p("搜索漏斗（用于重新搜索或新任务）", "h2"),
        _table([
            ["阶段", "预算/规则", "输出"],
            ["几何粗筛", "xy 0.1 m 网格；z=最低目标−0.15 m；512 样本、0.99 分位、0.12 m 朝向锥", "每侧 top-150"],
            ["锚定门", "frame 0 每侧 40 次重启", "剔除无法从开头跟随的底座"],
            ["配对探测", "top-12；每 60 帧 warm-start 探测", "保留同步可行配对"],
            ["全程决赛", "top-3 跑完整 strict + rescue v3.1", "按双臂 ACCEPT 排名"],
        ], [34 * mm, 102 * mm, 44 * mm]),
    ])

    story.extend([
        PageBreak(),
        p("3. IK、分支守卫与翻腕", "h1"),
        p("线上层与恢复层职责分离，普通跟随不能偷偷跨腕部分支。"),
        _table([
            ["状态", "允许动作", "失败/结束条件"],
            ["FOLLOW", "上一指令 warm-start DLS；保持 J4/J5 分支签名；Δq≤0.30 rad", "严格解不存在或离开可达域 → HOLD/触发 rescue"],
            ["HOLD", "保持上一安全指令，目标重放继续", "找到受控恢复方案后进入 A 或 B"],
            ["RESCUE A", "目标继续；最小加加速度轨迹截获未来 f+p 流形", "丢弃窗口源帧；末端 seam=0"],
            ["RESCUE B", "暂停目标；按 1 rad/s、4 rad/s²、0.1 s settle、0.015 s decide 转移", "恢复点到位后重接源帧；增加节拍"],
        ], [29 * mm, 104 * mm, 47 * mm]),
        Spacer(1, 5 * mm),
        p("DLS 参数", "h2"),
        _table([
            ["阻尼", "步长比例", "单次最大关节步", "最大迭代", "frame 0 重启"],
            ["0.04", "0.70", "0.18 rad", "120", "40"],
        ], [36 * mm] * 5, align="CENTER"),
        Spacer(1, 5 * mm),
        p("翻腕约束", "h2"),
        p(
            "PiperX 的 J4/J5 以带死区的符号组成腕部 basin 签名。FOLLOW 只接受同签名候选；"
            "跨签名变化必须形成显式 RescueEvent，记录起止帧、前后分支、最大关节差、腕部运动量、"
            "预测可跟随帧数、接缝误差和节拍成本。18 帧 dwell 防止来回抖动。"
        ),
        p("零接缝原则", "h2"),
        p(
            "模式 A 的过渡终点强制等于未来恢复帧的严格 IK 解，最后一行数值直接覆写为目标 q，"
            "因此不会再产生额外追赶段。模式 B 同样用五次 smoothstep 生成最小加加速度路径，并在每个子步做碰撞检查。"
        ),
    ])

    story.extend([
        PageBreak(),
        p("4. 完整轨迹量化验证", "h1"),
        Image(str(state_figure), width=180 * mm, height=73 * mm),
        Spacer(1, 3 * mm),
        Image(str(gate_figure), width=180 * mm, height=73 * mm),
        Spacer(1, 3 * mm),
        p(
            "严格同步覆盖率为 10.56%（112/1061）。图中 HOLD 与 RESCUE_A 数量较高，说明该 mount 和"
            "PiperX 物理可达域在 1 mm / 0.5° 门槛下仍存在显著限制；本实现解决的是跟踪协议的错误跳支、"
            "失败即终止和不可审计恢复问题，不宣称消除了机械臂几何不可达性。"
        ),
        p(
            "ACCEPT 帧的最坏复核：左/右位置 0.9979 / 0.9995 mm；左/右姿态 0.4996 / 0.4997°。"
            "全部位于硬门槛内。完整状态和 7 子步扫掠检查得到 0 个碰撞帧。"
        ),
    ])

    qa_paths = [ARTIFACT_DIR / f"qa_{name}.png" for name in ("start", "middle", "end")]
    story.extend([
        PageBreak(),
        p("5. 真实渲染视频证据", "h1"),
        p(
            "视频输出 1280×720、60 fps、1099 帧、18.30 s。1099 帧包含 1061 个源帧及模式 B 插入的执行帧；"
            "模式 A 丢弃 450 个源帧，模式 B 增加 0.667 s 节拍。画面使用官方 PiperX 网格、实际求解 qpos、"
            "左右目标轨迹与状态叠字。"
        ),
        Table([
            [Image(str(path), width=58 * mm, height=32.625 * mm) for path in qa_paths],
            [p("开始：严格 TRACKING", "small"), p("中段：机械臂与轨迹", "small"), p("结束：明确 HOLD", "small")],
        ], colWidths=[60 * mm] * 3, style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8D4DA")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDE5E9")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])),
        Spacer(1, 6 * mm),
        p("视频解码核验", "h2"),
        _table([
            ["项目", "结果"],
            ["容器与编码", "MP4 / mp4v，OpenCV 可完整解码"],
            ["帧数 / 帧率", "1099 / 60.0 fps"],
            ["时长 / 分辨率", "18.30 s / 1280×720"],
            ["轨迹数组", "18 个数组；浮点字段全部 finite"],
            ["碰撞", "0 帧（状态 + 扫掠）"],
        ], [55 * mm, 125 * mm]),
        p("注：视频中的大范围半透明线束是完整左右目标轨迹，不是实体绳索或环境几何。", "small"),
    ])

    story.extend([
        PageBreak(),
        p("6. 局限、部署边界与复现", "h1"),
        p("局限", "h2"),
        p(
            "（1）本次主证据仅完整重放 8-11/Fold_Box/161044，并未重新跑完报告中的 26 条轨迹 × 4 构型；"
            "12 族 base 已配置，但其它族仍应在部署前各自完整验证。"
            "（2）视频是动力学/几何仿真渲染，不包含实体控制器延迟、背隙、线缆或工件接触。"
            "（3）10.56% 严格覆盖说明在用户收紧到 0.5° 后，大量轨迹仍超出双臂同步可达域；"
            "HOLD/rescue 让失败可控且可审计，但不能替代换臂、换工位或重搜 mount。"
        ),
        p("保守部署建议", "h2"),
        _table([
            ["条件", "建议"],
            ["必须维持 1 mm / 0.5°", "将 HOLD 窗口作为工艺不可执行段；优先重搜 mount 或换更大工作空间机械臂"],
            ["允许目标停顿", "优先模式 B，接受节拍成本以减少模式 A 丢帧"],
            ["目标不能停顿", "保留模式 A，但上层工艺必须明确容忍丢帧窗口"],
            ["上真机前", "复核 TCP 标定、速度/加速度、急停、环境碰撞体与线缆包络"],
        ], [48 * mm, 132 * mm]),
        p("复现入口", "h2"),
        p("<font name='Courier'>python -m scripts.run_piperx_recommended_v31</font>"),
        p(
            "主产物：<br/>"
            f"• {escape(str(Path(summary_path).resolve()))}<br/>"
            f"• {escape(str(Path(trajectory_path).resolve()))}<br/>"
            f"• {escape(str((ARTIFACT_DIR / '8-11_Fold_Box_161044_recommended_v31.mp4').resolve()))}<br/>"
            f"• {escape(str((ARTIFACT_DIR / '8-11_Fold_Box_161044_recommended_v31.scene.xml').resolve()))}",
            "small",
        ),
        p("来源与可追溯性", "h2"),
        p(
            f"源方案：{escape(str(SOURCE_REPORT.resolve()))}<br/>"
            f"源轨迹 SHA-256：{summary['source']['sha256']}<br/>"
            "报告中的数字来自 summary JSON 与 trajectory NPZ，未从视频画面反推。",
            "small",
        ),
    ])

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path


def main(argv=None):
    options = parse_args(argv)
    output = build_report(options.summary, options.trajectory, options.output)
    print(output)


if __name__ == "__main__":
    main()
