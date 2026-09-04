"""Build the evidence-gated multi-task PiperX four-mount PDF report."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
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

from factory_bimanual.mount_comparison_visuals import (
    MOUNT_COLORS,
    MOUNT_LABELS,
    STUDY_PANEL_ORDER,
    audited_mount_rank,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "reports/piperx_multitask_fixed_time_mount_study"
DEFAULT_MANIFEST = DEFAULT_ROOT / "bundle_manifest.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "PiperX多任务Fixed-Time四构型对比报告.pdf"
REFERENCE_SOURCES = (
    ("AgileX Piper SDK V2 接口与关节/速度/加速度限制",
     "https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD"),
    ("AgileX Piper ROS 官方模型与固件对应的 URDF 说明",
     "https://github.com/agilexrobotics/piper_ros"),
    ("MoveIt PlanningScene 碰撞与约束检查接口",
     "https://moveit.github.io/moveit_tutorials/doc/planning_scene/planning_scene_tutorial.html"),
)
REPORT_FONT_NAME = "PiperXReportFont"


def _register_report_font():
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    )
    for path in candidates:
        if path.is_file():
            pdfmetrics.registerFont(TTFont(REPORT_FONT_NAME, str(path)))
            return REPORT_FONT_NAME
    raise RuntimeError(
        "a CJK font is required (Microsoft YaHei, Noto Sans CJK, or PingFang)")


def validate_report_manifest(manifest):
    if (manifest.get("schema") != "piperx-multitask-fixed-time-bundle-v1"
            or manifest.get("status") != "complete"
            or manifest.get("retiming_applied") is not False):
        raise ValueError("report requires a complete non-retimed bundle")
    shards = manifest.get("shards", [])
    trajectories = sorted({item.get("trajectory") for item in shards})
    pairs = {(item.get("trajectory"), item.get("mode")) for item in shards}
    expected = {(trajectory, mode) for trajectory in trajectories
                for mode in STUDY_PANEL_ORDER}
    if (manifest.get("trajectory_count") != 27
            or manifest.get("family_count") != 12
            or manifest.get("mode_count") != 4
            or len(shards) != 108 or len(trajectories) != 27
            or pairs != expected):
        raise ValueError("report requires the complete 27 x 4 matrix")
    return manifest


def _metric_shards(manifest):
    rows = []
    for shard in manifest["shards"]:
        if "metrics" in shard:
            rows.append(shard)
            continue
        summary_path = Path(shard["summary_json"])
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({**shard, "metrics": summary["metrics"],
                     "dynamics": summary["dynamics"],
                     "mount": summary["mount"]})
    return rows


def build_report_claims(manifest):
    manifest = validate_report_manifest(manifest)
    rows = _metric_shards(manifest)
    winner_counts = Counter({mode: 0 for mode in STUDY_PANEL_ORDER})
    winners = {}
    by_trajectory = defaultdict(list)
    for row in rows:
        by_trajectory[row["trajectory"]].append(row)
    for trajectory, candidates in by_trajectory.items():
        rank_row = lambda row: audited_mount_rank({
            **row["metrics"], "mode": row["mode"]})
        winner = min(candidates, key=rank_row)
        winner_counts[winner["mode"]] += 1
        winners[trajectory] = winner["mode"]
    deployable = [
        row for row in rows
        if float(row["metrics"]["both_accept_coverage"]) >= 1.0 - 1e-12
        and bool(row.get("dynamics", {}).get("limits_passed"))
        and sum(int(row["metrics"].get(name, 0)) for name in (
            "collision_frames", "edge_collision_frames",
            "topology_invalid_frames")) == 0
    ]
    invalid_by_trajectory = {
        trajectory: max(int(row["metrics"].get("invalid_source_frames", 0))
                        for row in candidates)
        for trajectory, candidates in by_trajectory.items()
    }
    return {
        "total_trajectories": len(by_trajectory),
        "total_experiments": len(rows),
        "winner_counts": dict(winner_counts),
        "winners": winners,
        "deployable_experiments": len(deployable),
        "deployable_trajectories": len({
            row["trajectory"] for row in deployable}),
        "invalid_source_frames": sum(invalid_by_trajectory.values()),
        "invalid_source_trajectories": sum(
            value > 0 for value in invalid_by_trajectory.values()),
        "rows": rows,
    }


def _styles():
    font_name = _register_report_font()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleZH", parent=base["Title"], fontName=font_name,
            fontSize=24, leading=32, alignment=TA_CENTER,
            textColor=colors.HexColor("#17243A"), spaceAfter=8 * mm),
        "subtitle": ParagraphStyle(
            "SubtitleZH", parent=base["Normal"], fontName=font_name,
            fontSize=11, leading=17, alignment=TA_CENTER,
            textColor=colors.HexColor("#526174"), spaceAfter=8 * mm),
        "h1": ParagraphStyle(
            "H1ZH", parent=base["Heading1"], fontName=font_name,
            fontSize=17, leading=23, textColor=colors.HexColor("#183B63"),
            spaceBefore=3 * mm, spaceAfter=3 * mm),
        "h2": ParagraphStyle(
            "H2ZH", parent=base["Heading2"], fontName=font_name,
            fontSize=12, leading=17, textColor=colors.HexColor("#2D5E8C"),
            spaceBefore=2 * mm, spaceAfter=2 * mm),
        "body": ParagraphStyle(
            "BodyZH", parent=base["BodyText"], fontName=font_name,
            fontSize=9, leading=15, textColor=colors.HexColor("#202936"),
            spaceAfter=2.5 * mm),
        "small": ParagraphStyle(
            "SmallZH", parent=base["BodyText"], fontName=font_name,
            fontSize=7.3, leading=10, textColor=colors.HexColor("#526174")),
    }


def _table(rows, widths, *, font_size=8):
    table = Table(rows, colWidths=widths, repeatRows=1)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), REPORT_FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#BAC7D5")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244F73")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ]
    for index in range(2, len(rows), 2):
        commands.append(("BACKGROUND", (0, index), (-1, index),
                         colors.HexColor("#EFF4F8")))
    table.setStyle(TableStyle(commands))
    return table


def _footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#738094"))
    canvas.drawRightString(A4[0] - 14 * mm, 8 * mm, str(document.page))
    canvas.restoreState()


def _middle_frame(video_path, output_path):
    capture = cv2.VideoCapture(str(video_path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode report frame: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"cannot write report frame: {output_path}")
    return output_path


def build_report(manifest_path=DEFAULT_MANIFEST, output_path=DEFAULT_OUTPUT):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    claims = build_report_claims(manifest)
    rows = claims["rows"]
    styles = _styles()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title="PiperX Multi-Task Fixed-Time Four-Mount Comparison")
    winner_rows = [["构型", "严格位姿覆盖胜出轨迹", "占比"]]
    for mode in STUDY_PANEL_ORDER:
        count = claims["winner_counts"][mode]
        winner_rows.append([
            MOUNT_LABELS[mode], str(count), f"{100 * count / 27:.1f}%"])
    story = [
        Spacer(1, 10 * mm),
        Paragraph("PiperX 多任务 Fixed-Time 四构型对比报告", styles["title"]),
        Paragraph(
            "27 条双手轨迹 × 4 种安装方式 = 108 组严格 fixed-time IK 实验",
            styles["subtitle"]),
        Paragraph("研究结论", styles["h1"]),
        Paragraph(
            "本报告比较现有基线、桌面正立、墙面或立杆横装、顶部倒挂。"
            "所有实验直接使用原始时间戳，不重定时、不插帧，位置与姿态接受门"
            "固定为 1 mm / 0.5°，并相对于发布的条件化 TCP 目标计算。"
            "构型不可行、HOLD、碰撞和动力学超限均保留。",
            styles["body"]),
        Paragraph(
            f"严格位姿覆盖率胜者统计不包含动力学门；同时达到 100% 双臂覆盖、"
            f"动力学通过和三类安全计数为零的可部署实验为 "
            f"{claims['deployable_experiments']}/108，覆盖 "
            f"{claims['deployable_trajectories']}/27 条轨迹。",
            styles["body"]),
        _table(winner_rows, [70 * mm, 45 * mm, 45 * mm], font_size=9),
        Spacer(1, 4 * mm),
        Paragraph("重要解释边界", styles["h1"]),
        Paragraph(
            "ACCEPT 表示该源帧在严格 TCP 门限内获得 IK 解；它不等于满足 PiperX"
            " 速度、加速度或真机安全要求。原始 TCP 记录在 IK 前执行确定性的 9 帧"
            "条件化，目标相对原记录的允许变化上限为 5 mm / 1°；严格 1 mm / 0.5°"
            "门限针对条件化目标。该步骤不改变时间戳，也不属于重定时。"
            "零 MuJoCo 碰撞也不构成真机许可。",
            styles["body"]),
        Paragraph(
            f"原始双手数据有效性掩码已进入每个分片。共有 "
            f"{claims['invalid_source_trajectories']} 条轨迹包含 "
            f"{claims['invalid_source_frames']} 个双手源姿态无效帧；这些帧强制"
            "拒绝，并同时报告全源帧覆盖率与仅源姿态有效帧覆盖率。",
            styles["body"]),
        PageBreak(),
        Paragraph("1. 实验协议", styles["h1"]),
        Paragraph(
            "三种非基线构型对每条双手轨迹独立搜索；候选数量、锚点与恢复预算"
            "按该轨迹的可达性和安全漏斗自适应，而不是假设所有轨迹预算相同或"
            "复用同族安装位姿。随后全部 27 条轨迹与"
            "四种构型分别执行完整源时间轴 IK。失败帧执行 HOLD，"
            "下一帧从保持状态继续。双臂 ACCEPT、状态碰撞、扫掠边碰撞、结构拓扑、"
            "关节速度和加速度分别审计。每个分片同时保存 9 帧条件化配置及其相对"
            "原始 TCP 目标的偏移范围。", styles["body"]),
        _table([["构型", "物理含义", "报告颜色"], *[
            [MOUNT_LABELS[mode], {
                "baseline": "当前任务级推荐安装",
                "upright_table": "底座安装轴竖直向上",
                "horizontal_wall": "立杆或墙装，安装轴水平",
                "inverted": "顶装倒挂，安装轴向下",
            }[mode], MOUNT_COLORS[mode]]
            for mode in STUDY_PANEL_ORDER
        ]], [55 * mm, 90 * mm, 30 * mm], font_size=8),
        Spacer(1, 4 * mm),
        Paragraph("资料依据", styles["h2"]),
        *[
            Paragraph(
                f'{index}. <link href="{url}" color="#2D5E8C">{label}</link>',
                styles["small"])
            for index, (label, url) in enumerate(REFERENCE_SOURCES, 1)
        ],
    ]
    figure_specs = [
        ("coverage_heatmap.png", "2. 双臂同时 ACCEPT 热力图"),
        ("winner_counts.png", "3. 构型胜者分布"),
        ("longest_hold_heatmap.png", "4. 最长 HOLD 轨迹占比"),
        ("collision_topology_heatmap.png", "5. 碰撞与双臂拓扑硬门"),
        ("maximum_position_error_heatmap.png", "6. ACCEPT 帧位置精度"),
        ("maximum_orientation_error_heatmap.png", "7. ACCEPT 帧姿态精度"),
        ("velocity_heatmap.png", "8. 原始节奏速度审计"),
        ("acceleration_heatmap.png", "9. 原始节奏加速度审计"),
        ("deployability_heatmap.png", "10. 覆盖、动力学与安全联合门"),
    ]
    figure_root = Path(manifest_path).parent / "figures"
    for filename, heading in figure_specs:
        path = figure_root / filename
        if not path.exists():
            raise FileNotFoundError(f"missing report figure: {path}")
        story.extend([
            PageBreak(), Paragraph(heading, styles["h1"]),
            Image(str(path), width=178 * mm, height=230 * mm,
                  kind="proportional"),
        ])

    by_trajectory = defaultdict(list)
    for row in rows:
        by_trajectory[row["trajectory"]].append(row)
    frame_root = Path(manifest_path).parent / "report_frames"
    for trajectory in sorted(by_trajectory):
        candidates = sorted(
            by_trajectory[trajectory],
            key=lambda item: STUDY_PANEL_ORDER.index(item["mode"]))
        table_rows = [[
            "构型", "全帧/有效帧 ACCEPT", "最长 HOLD", "碰撞/边/拓扑",
            "ACCEPT 最大误差", "动力学门"]]
        for row in candidates:
            metric = row["metrics"]
            dynamics = row.get("dynamics", {})
            table_rows.append([
                MOUNT_LABELS[row["mode"]],
                f"{100 * metric['both_accept_coverage']:.2f}% / "
                f"{100 * metric['both_accept_coverage_valid_source']:.2f}%",
                f"{100 * metric['longest_hold_ratio']:.1f}%",
                f"{metric['collision_frames']}/{metric['edge_collision_frames']}/"
                f"{metric['topology_invalid_frames']}",
                ("HOLD" if metric["maximum_accepted_position_error_mm"] is None
                 else f"{metric['maximum_accepted_position_error_mm']:.2f} mm / "
                      f"{metric['maximum_accepted_orientation_error_deg']:.2f}°"),
                "通过" if dynamics.get("limits_passed") else "不通过",
            ])
        video = (Path(manifest_path).parent / "videos" / "comparisons"
                 / f"{trajectory.replace('/', '_')}_four_mount_fixed_time.mp4")
        frame = frame_root / f"{trajectory.replace('/', '_')}_middle.png"
        _middle_frame(video, frame)
        story.extend([
            PageBreak(),
            Paragraph(f"轨迹：{trajectory}", styles["h1"]),
            Paragraph(
                f"本轨迹严格位姿覆盖胜出构型（不含动力学门）："
                f"{MOUNT_LABELS[claims['winners'][trajectory]]}。"
                "下图四格使用相同源时间、相机尺度和颜色语义。",
                styles["body"]),
            Image(str(frame), width=178 * mm, height=100.1 * mm),
            Spacer(1, 3 * mm),
            _table(table_rows,
                   [38 * mm, 30 * mm, 25 * mm, 36 * mm, 30 * mm, 22 * mm],
                   font_size=6.7),
        ])
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(build_report(args.manifest, args.output))


if __name__ == "__main__":
    main()
