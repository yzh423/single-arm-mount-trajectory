"""Build the literature-grounded PiperX Fixed-time optimization audit PDF."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "reports/piperx_multitask_fixed_time_mount_study/literature_audit"
DEFAULT_OUTPUT = ROOT / "output/pdf/PiperX双手多任务Fixed-Time项目优化与文献审计报告.pdf"
FONT = "OptimizationReport"
MODES = ("baseline", "upright_table", "horizontal_wall", "inverted")
MODE_ZH = {
    "baseline": "基线",
    "upright_table": "桌面直立",
    "horizontal_wall": "墙面横装",
    "inverted": "倒挂",
}


def _register_font():
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not path.is_file():
        raise RuntimeError("Microsoft YaHei is required to build this report")
    pdfmetrics.registerFont(TTFont(FONT, str(path)))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title_zh", parent=base["Title"], fontName=FONT, fontSize=23,
            leading=31, alignment=TA_CENTER, textColor=colors.HexColor("#14283D"),
            spaceAfter=7 * mm),
        "subtitle": ParagraphStyle(
            "subtitle_zh", parent=base["BodyText"], fontName=FONT, fontSize=10,
            leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#53677A"),
            spaceAfter=7 * mm),
        "h1": ParagraphStyle(
            "h1_zh", parent=base["Heading1"], fontName=FONT, fontSize=16,
            leading=22, textColor=colors.HexColor("#194E77"), spaceBefore=2 * mm,
            spaceAfter=3 * mm),
        "h2": ParagraphStyle(
            "h2_zh", parent=base["Heading2"], fontName=FONT, fontSize=12,
            leading=17, textColor=colors.HexColor("#2E688F"), spaceBefore=2 * mm,
            spaceAfter=2 * mm),
        "body": ParagraphStyle(
            "body_zh", parent=base["BodyText"], fontName=FONT, fontSize=9,
            leading=15, textColor=colors.HexColor("#202B36"), spaceAfter=2.5 * mm),
        "small": ParagraphStyle(
            "small_zh", parent=base["BodyText"], fontName=FONT, fontSize=7.2,
            leading=10.5, textColor=colors.HexColor("#526270"), spaceAfter=1.5 * mm),
        "callout": ParagraphStyle(
            "callout_zh", parent=base["BodyText"], fontName=FONT, fontSize=10,
            leading=16, textColor=colors.HexColor("#17324D"),
            backColor=colors.HexColor("#EAF4FA"), borderColor=colors.HexColor("#7BB1CF"),
            borderWidth=.8, borderPadding=8, spaceBefore=2 * mm, spaceAfter=4 * mm),
    }


def _table(rows, widths, font_size=7.4):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3.3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#194E77")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F6F8")]),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#AEBEC9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _figure(path, width=175 * mm):
    image = Image(str(path))
    original_width = image.imageWidth
    original_height = image.imageHeight
    image.drawWidth = width
    image.drawHeight = width * original_height / original_width
    return image


def _load():
    summary = json.loads(
        (AUDIT_ROOT / "raw_source_audit_summary.json").read_text(encoding="utf-8"))
    with (AUDIT_ROOT / "raw_source_shard_audit.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = {
        "conditioned_pair_coverage", "raw_pair_coverage",
        "solver_target_max_position_deviation_mm",
        "solver_target_max_orientation_deviation_deg",
        "smoothing_max_position_deviation_mm",
        "smoothing_max_orientation_deviation_deg",
        "wrist_adaptation_max_orientation_deg", "source_interval_median_ms",
        "source_tcp_linear_speed_max_m_s", "source_tcp_angular_speed_max_rad_s",
        "maximum_joint_velocity_rad_s", "maximum_joint_acceleration_rad_s2",
    }
    for row in rows:
        for name in numeric:
            row[name] = float(row[name])
    return summary, rows


def _make_figures(summary, rows):
    output = AUDIT_ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)

    raw = [100 * summary["by_mode"][mode]["raw_pair_coverage"] for mode in MODES]
    conditioned = [
        100 * summary["by_mode"][mode]["conditioned_pair_coverage"] for mode in MODES]
    x = np.arange(len(MODES))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - .18, raw, .36, label="未平滑源目标", color="#167D9A")
    ax.bar(x + .18, conditioned, .36, label="条件化求解目标", color="#E0993E")
    ax.set_xticks(x, [MODE_ZH[mode] for mode in MODES])
    ax.set_ylabel("双手严格覆盖率（%）")
    ax.set_title("同一正式轨迹对两种目标口径的覆盖率")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    fig.tight_layout()
    coverage = output / "raw_vs_conditioned_coverage.png"
    fig.savefig(coverage, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    palette = dict(zip(MODES, ("#4063D8", "#2A9D8F", "#E76F51", "#8E5EA2")))
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        ax.scatter(
            [row["maximum_joint_acceleration_rad_s2"] for row in selected],
            [100 * row["raw_pair_coverage"] for row in selected],
            s=34, alpha=.75, label=MODE_ZH[mode], color=palette[mode])
    ax.axvline(4.0, color="#253746", ls="--", lw=1.2, label="研究限制 4 rad/s²")
    ax.axvline(5.0, color="#A63838", ls=":", lw=1.4, label="官方可配置上限 5 rad/s²")
    ax.set_xscale("log")
    ax.set_xlabel("最大关节加速度（rad/s²，对数轴）")
    ax.set_ylabel("未平滑源目标严格覆盖率（%）")
    ax.set_title("精度覆盖与固定时间动力学没有形成可部署交集")
    ax.grid(alpha=.22)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    dynamics = output / "coverage_vs_dynamics.png"
    fig.savefig(dynamics, dpi=180, bbox_inches="tight")
    plt.close(fig)

    unique = {}
    for row in rows:
        unique.setdefault(row["trajectory"], row)
    fastest = sorted(
        unique.values(), key=lambda row: row["source_tcp_linear_speed_max_m_s"],
        reverse=True)[:10]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    labels = [row["trajectory"].replace("8-11/", "").replace("8-12/", "")
              for row in fastest][::-1]
    axes[0].barh(labels, [row["source_tcp_linear_speed_max_m_s"]
                          for row in fastest][::-1], color="#167D9A")
    axes[0].set_xlabel("峰值线速度（m/s）")
    axes[0].set_title("源 TCP 平移跳变")
    angular = sorted(unique.values(), key=lambda row:
                     row["source_tcp_angular_speed_max_rad_s"], reverse=True)[:10]
    labels_a = [row["trajectory"].replace("8-11/", "").replace("8-12/", "")
                for row in angular][::-1]
    axes[1].barh(labels_a, [row["source_tcp_angular_speed_max_rad_s"]
                            for row in angular][::-1], color="#D77A2B")
    axes[1].set_xlabel("峰值角速度（rad/s）")
    axes[1].set_title("源 TCP 姿态跳变")
    for ax in axes:
        ax.grid(axis="x", alpha=.22)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    source_rates = output / "source_tcp_rate_outliers.png"
    fig.savefig(source_rates, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return coverage, dynamics, source_rates


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 7)
    canvas.setFillColor(colors.HexColor("#657786"))
    canvas.drawString(18 * mm, 10 * mm, "PiperX Fixed-time 文献与证据审计 · 2026-09-07")
    canvas.drawRightString(192 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build(output=DEFAULT_OUTPUT):
    _register_font()
    styles = _styles()
    summary, rows = _load()
    coverage_figure, dynamics_figure, rates_figure = _make_figures(summary, rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=16 * mm,
        title="PiperX双手多任务Fixed-Time项目优化与文献审计报告",
        author="single-arm-mount evidence pipeline")
    story = []
    P = lambda text, style="body": story.append(Paragraph(text, styles[style]))

    P("PiperX 双手多任务 Fixed-time<br/>项目优化与文献审计报告", "title")
    P("27 条双手轨迹 × 4 种安装 · 原始时间戳 · 禁止重定时 · 1 mm / 0.5° · 正式安全硬门", "subtitle")
    P("审计结论", "h1")
    P("现有 108 个正式分片在安全方面保持自洽：帧碰撞、扫掠边碰撞、安装拓扑违规均为 0。低覆盖的主因并非单一 mount 选择，而是目标口径和固定时间动力学。", "callout")
    key_rows = [
        ["审计项", "结果", "解释"],
        ["条件化目标覆盖", f"{100*summary['weighted_conditioned_pair_coverage']:.2f}%", "旧报告口径；目标允许空间调整"],
        ["未平滑源目标覆盖", f"{100*summary['weighted_raw_pair_coverage']:.2f}%", "本报告主口径；固定工具映射、无时变翻腕"],
        ["研究动力学通过", f"{summary['study_dynamics_pass_shards']}/108", "1 rad/s、4 rad/s²"],
        ["官方上限通过", f"{summary['official_ceiling_dynamics_pass_shards']}/108", "3 rad/s、5 rad/s²；非零覆盖交集仍为 0"],
        ["安全违规", "0 / 0 / 0", "frame / swept edge / topology"],
        ["源 TCP 峰值", f"{summary['maximum_source_tcp_linear_speed_m_s']:.2f} m/s；{summary['maximum_source_tcp_angular_speed_rad_s']:.2f} rad/s", "采样时间约 12.4 ms"],
    ]
    story.append(_table(key_rows, [46 * mm, 43 * mm, 87 * mm], 8))
    story.append(Spacer(1, 4 * mm))
    P("发布判断", "h2")
    P("当前项目可以发布为“零碰撞的 Fixed-time mount/IK 对比研究”，但不能把条件化目标覆盖解释为对原始双手记录的严格完全跟随，也不能把动力学超限的 MuJoCo 回放称为可直接下发 PiperX 的轨迹。")

    story.append(PageBreak())
    P("1. 为什么旧成功率偏乐观", "h1")
    story.append(_figure(coverage_figure))
    P("图 1｜同一组正式 TCP 实际位姿，用两套目标重算。条件化目标是求解器使用的平滑目标；未平滑源目标保留原始空间记录，仅应用固定工具坐标变换。", "small")
    P("求解前的 9 帧 Savitzky–Golay SE(3) 条件化允许最多 5 mm 平移和 1° 姿态改变，而验收阈值是 1 mm / 0.5°。108/108 个分片的条件化偏移都越过严格阈值。Fold_Box 右臂还使用最多 12.5° 的时变翻腕，因此本轮把时变翻腕也从源目标口径中剥离。")
    mode_rows = [["安装", "源目标覆盖", "条件化覆盖", "动力学通过"]]
    for mode in MODES:
        item = summary["by_mode"][mode]
        mode_rows.append([
            MODE_ZH[mode], f"{100*item['raw_pair_coverage']:.2f}%",
            f"{100*item['conditioned_pair_coverage']:.2f}%",
            f"{item['study_dynamics_pass_shards']}/27",
        ])
    story.append(_table(mode_rows, [45 * mm, 43 * mm, 43 * mm, 43 * mm], 8))

    story.append(PageBreak())
    P("2. Fixed-time 下的物理矛盾", "h1")
    story.append(_figure(dynamics_figure))
    P("图 2｜每个点为一个正式分片。位于 4 rad/s² 左侧的 35 个分片全部是 0 覆盖；非零源目标覆盖全部伴随动力学超限。", "small")
    P("TOPP-RA 把时间参数化视为满足速度、加速度和力矩约束的关键自由度。项目明确禁止重定时后，这个自由度不存在；正确行为是报告不可行窗口，或改变空间轨迹/硬件/精度要求，而不是继续增加逐帧 IK 重启。")
    P("Piper SDK V2 官方接口给出 J1–J6 最大关节速度可配置范围 0–3.0 rad/s、最大关节加速度 0–5.0 rad/s²。把审计上限从研究值 1/4 提到官方可配置上限 3/5 并未产生任何“非零严格覆盖且动力学通过”的分片。")
    story.append(_figure(rates_figure, 178 * mm))
    P("图 3｜源记录中的任务空间段速度峰值。它们是按实际相邻时间戳直接计算的诊断值，不是经过重采样得到。", "small")

    story.append(PageBreak())
    P("3. 文献对现有算法的直接启示", "h1")
    literature = [
        ["来源", "经过同行评审的核心结论", "项目改造"],
        ["TRAC-IK\nBeeson & Ames, 2015", "并行改进 Jacobian 与 SQP；支持有界关节和笛卡尔维度容差", "多启动 SQP + 容差归一化残差，替换单一固定 DLS"],
        ["RelaxedIK\nRakita et al., 2018", "逐帧 IK 会产生不连续、碰撞和奇异；应联合优化速度、加速度、jerk、自碰撞", "时间窗口内联合优化双臂，而不是每帧独立 warm-start"],
        ["HQP\nEscande et al., 2014", "用严格层级处理等式与不等式约束", "安全、关节界、速度、加速度放第一层；精度和姿势次级"],
        ["TrajOpt\nSchulman et al., 2014", "序列凸优化并进行连续碰撞检查", "保持 MuJoCo swept-edge 复核，并把过渡约束纳入窗口求解"],
        ["TOPP-RA\nPham & Pham, 2018", "速度/加速度可行性与路径时间参数化不可分", "禁止重定时时，显式输出不可行窗口"],
        ["SDLS\nBuss & Kim, 2005", "按奇异方向选择阻尼", "用奇异值自适应阻尼替换固定 0.3"],
    ]
    story.append(_table(literature, [37 * mm, 70 * mm, 69 * mm], 7.2))
    story.append(Spacer(1, 4 * mm))
    P("固定阻尼的具体问题", "h2")
    P("当前正式求解器使用 damping=0.3 和 maximum_step=0.3 rad，候选选择时未把 1 rad/s、4 rad/s² 作为硬约束；关节导数只在整条轨迹完成后计算。这会让一个姿态上严格通过的跳变在动力学上不可执行。文献支持的方向不是简单调小 step，而是在候选与分支选择阶段就使用时间间隔和上一帧速度。")

    story.append(PageBreak())
    P("4. 推荐架构：先准入，再联合求解", "h1")
    architecture = [
        ["层级", "硬约束 / 目标", "输出"],
        ["P0 数据准入", "时间严格递增；双手有效；源 TCP 跳变审计；不删帧", "可行性诊断与不可行窗口"],
        ["P0 验收合同", "未平滑源目标 1 mm / 0.5°；双手同帧", "主覆盖率；条件化覆盖仅诊断"],
        ["P1 HQP 第一层", "关节界、速度、加速度、碰撞、边碰撞、拓扑", "固定时间可执行候选"],
        ["P1 HQP 第二层", "双手源目标 tolerance tube", "严格跟随/明确失败"],
        ["P1 平滑层", "最小速度、加速度、jerk、奇异风险", "连续分支与低冲击轨迹"],
        ["P2 mount 外层", "只比较通过安全与动力学门的源目标覆盖", "桌面/墙面/倒挂的真实可部署排名"],
    ]
    story.append(_table(architecture, [34 * mm, 91 * mm, 51 * mm], 7.5))
    P("推荐状态转移", "h2")
    P("每个时间节点保存多个左右臂 IK 分支；边代价包含关节位移、速度变化、加速度、jerk 和奇异风险；边只有在速度/加速度、碰撞、扫掠碰撞和拓扑全部通过时才存在。若图不连通，保留安全 HOLD 作为执行输出，但该帧在源目标覆盖中必须失败。")
    P("关于目标平滑", "h2")
    P("允许在固定时间戳上做空间滤波，但平滑偏移必须占用总误差预算，并最终回到未平滑目标验收。推荐先为预处理分配不超过 0.5 mm / 0.25°，为 IK 与控制误差保留另一半；如果源跳变远大于该 tube，应判定数据/任务合同不可行，而不是扩大平滑上限。")

    story.append(PageBreak())
    P("5. 逐任务异常与优化优先级", "h1")
    fastest = {}
    for row in rows:
        fastest.setdefault(row["trajectory"], row)
    fastest = sorted(fastest.values(), key=lambda row:
                     row["source_tcp_linear_speed_max_m_s"], reverse=True)[:10]
    speed_rows = [["轨迹", "线速度峰值", "角速度峰值", "优先处理"]]
    for row in fastest:
        speed_rows.append([
            row["trajectory"], f"{row['source_tcp_linear_speed_max_m_s']:.2f} m/s",
            f"{row['source_tcp_angular_speed_max_rad_s']:.2f} rad/s",
            "采集/位姿跳变审计",
        ])
    story.append(_table(speed_rows, [76 * mm, 32 * mm, 34 * mm, 34 * mm], 7.1))
    P("最高源目标覆盖分片", "h2")
    best = sorted(rows, key=lambda row: row["raw_pair_coverage"], reverse=True)[:8]
    best_rows = [["轨迹", "安装", "源目标", "条件化目标"]]
    for row in best:
        best_rows.append([
            row["trajectory"], MODE_ZH[row["mode"]],
            f"{100*row['raw_pair_coverage']:.2f}%",
            f"{100*row['conditioned_pair_coverage']:.2f}%",
        ])
    story.append(_table(best_rows, [82 * mm, 31 * mm, 31 * mm, 32 * mm], 7.2))
    P("Seal_Bag 在条件化目标上可达到 94.82–100%，但对未平滑源目标只有 24.25–31.65%；这正是应优先用新口径复核的任务。Fold_Box 同样受平滑与 12.5° 时变翻腕影响。")

    story.append(PageBreak())
    P("6. 本轮代码与证据改进", "h1")
    implemented = [
        ["产物", "作用"],
        ["factory_bimanual/fixed_time_evidence_audit.py", "求解器无关的四元数、源时间、目标误差和双手通过计算"],
        ["scripts/audit_piperx_fixed_time_evidence.py", "重算 108 个分片的源目标覆盖、条件化偏移、动力学与安全"],
        ["raw_source_shard_audit.csv", "逐分片可复算结果，不只给平均数"],
        ["raw_source_audit_summary.json", "合同、全局指标和分安装汇总"],
        ["report-source.md / claim-source-ledger.md", "报告主源与逐项证据追踪"],
        ["README", "明确条件化研究与源目标严格跟随的边界"],
    ]
    story.append(_table(implemented, [65 * mm, 111 * mm], 7.8))
    P("复现命令", "h2")
    P("<font name='Courier'>python -m scripts.audit_piperx_fixed_time_evidence</font><br/><font name='Courier'>python -m scripts.build_piperx_literature_optimization_report</font><br/><font name='Courier'>python -m pytest tests/factory_bimanual/test_fixed_time_evidence_audit.py -q</font>", "small")
    P("不应做的优化", "h2")
    P("不要通过排除高动态帧、扩大 source-invalid 掩码、把安全 HOLD 记为成功、只放宽条件化上限、只增加随机 IK 重启，或在视频中改变播放时间来提高成功率。这些方法都会改变问题或指标，而不是解决 Fixed-time 完全跟随。")

    story.append(PageBreak())
    P("7. 参考文献与官方资料", "h1")
    references = [
        "1. Beeson, P. &amp; Ames, B. TRAC-IK: An Open-Source Library for Improved Solving of Generic Inverse Kinematics. IEEE-RAS Humanoids (2015). <link href='https://doi.org/10.1109/HUMANOIDS.2015.7363472'>doi:10.1109/HUMANOIDS.2015.7363472</link>.",
        "2. Rakita, D., Mutlu, B. &amp; Gleicher, M. RelaxedIK: Real-time Synthesis of Accurate and Feasible Robot Arm Motion. RSS (2018). <link href='https://doi.org/10.15607/RSS.2018.XIV.043'>doi:10.15607/RSS.2018.XIV.043</link>.",
        "3. Escande, A., Mansard, N. &amp; Wieber, P.-B. Hierarchical quadratic programming: Fast online humanoid-robot motion generation. IJRR 33 (2014). <link href='https://doi.org/10.1177/0278364914521306'>doi:10.1177/0278364914521306</link>.",
        "4. Schulman, J. et al. Motion planning with sequential convex optimization and convex collision checking. IJRR 33 (2014). <link href='https://doi.org/10.1177/0278364914528132'>doi:10.1177/0278364914528132</link>.",
        "5. Pham, H. &amp; Pham, Q.-C. A New Approach to Time-Optimal Path Parameterization Based on Reachability Analysis. IEEE T-RO 34 (2018). <link href='https://doi.org/10.1109/TRO.2018.2819195'>doi:10.1109/TRO.2018.2819195</link>.",
        "6. Buss, S. R. &amp; Kim, J.-S. Selectively Damped Least Squares for Inverse Kinematics. Journal of Graphics Tools 10 (2005). <link href='https://doi.org/10.1080/2151237X.2005.10129202'>doi:10.1080/2151237X.2005.10129202</link>.",
        "7. Nakamura, Y. &amp; Hanafusa, H. Inverse Kinematic Solutions With Singularity Robustness for Robot Manipulator Control. JDSMC 108 (1986). <link href='https://doi.org/10.1115/1.3143764'>doi:10.1115/1.3143764</link>.",
        "8. AgileX Robotics. Piper SDK V2 Interface Documentation. <link href='https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD'>官方接口文档</link>（访问日期 2026-09-07）。",
    ]
    for reference in references:
        P(reference, "small")
    P("证据边界", "h2")
    P("论文用于支持算法设计原则；项目中的覆盖率、速度、加速度、碰撞和目标偏移全部由本仓库正式 NPZ、源 CSV 和新增审计脚本重算。报告没有用论文替代实验，也没有把仿真结果外推为真实硬件成功率。")

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(build(args.output))


if __name__ == "__main__":
    main()
