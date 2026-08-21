"""Build the evidence-gated Chinese PiperX fixed-time comparison PDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "reports/piperx_two_task_fixed_time"
DEFAULT_MANIFEST = DEFAULT_ROOT / "fixed_time_manifest.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "PiperX双任务Fixed-Time完全跟随报告.pdf"


def validate_report_manifest(manifest):
    if manifest.get("schema") != "piperx-two-task-fixed-time-manifest-v1":
        raise ValueError("unexpected fixed-time manifest schema")
    if (manifest.get("timing_mode") != "fixed_source_time"
            or manifest.get("retiming_applied") is not False):
        raise ValueError("fixed-time report must not contain retiming")
    if set(manifest.get("tasks", {})) != {"fold_box", "seal_bag"}:
        raise ValueError("fixed-time report requires Fold_Box and Seal_Bag")
    for key, task in manifest["tasks"].items():
        timing = task.get("timing", {})
        tracking = task.get("tracking", {})
        dynamics = task.get("dynamics", {})
        if (timing.get("timestamp_identity") is not True
                or timing.get("qpos_identity") is not True
                or timing.get("inserted_frames") != 0
                or timing.get("added_duration_s") != 0.0):
            raise ValueError(f"{key}: fixed-time identity evidence is absent")
        if tracking.get("coverage") != 1.0 or tracking.get("collision_frames") != 0:
            raise ValueError(f"{key}: tracking or collision gate failed")
        if dynamics.get("limits_passed") is not False:
            raise ValueError(f"{key}: dynamic violation disclosure is absent")
    return manifest


def _styles():
    try:
        pdfmetrics.registerFont(TTFont("MicrosoftYaHei", r"C:\Windows\Fonts\msyh.ttc"))
        font = "MicrosoftYaHei"
    except Exception:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font = "STSong-Light"
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleZH", parent=base["Title"], fontName=font,
            fontSize=24, leading=31, alignment=TA_CENTER,
            textColor=colors.HexColor("#17243A"), spaceAfter=8 * mm),
        "subtitle": ParagraphStyle(
            "SubtitleZH", parent=base["Normal"], fontName=font,
            fontSize=11, leading=17, alignment=TA_CENTER,
            textColor=colors.HexColor("#506078"), spaceAfter=8 * mm),
        "h1": ParagraphStyle(
            "H1ZH", parent=base["Heading1"], fontName=font,
            fontSize=16, leading=22, textColor=colors.HexColor("#183B63"),
            spaceBefore=4 * mm, spaceAfter=3 * mm),
        "h2": ParagraphStyle(
            "H2ZH", parent=base["Heading2"], fontName=font,
            fontSize=12, leading=17, textColor=colors.HexColor("#2D5E8C"),
            spaceBefore=3 * mm, spaceAfter=2 * mm),
        "body": ParagraphStyle(
            "BodyZH", parent=base["BodyText"], fontName=font,
            fontSize=9.2, leading=15, textColor=colors.HexColor("#202936"),
            spaceAfter=2.5 * mm),
        "small": ParagraphStyle(
            "SmallZH", parent=base["BodyText"], fontName=font,
            fontSize=7.8, leading=12, textColor=colors.HexColor("#526174")),
        "font": font,
    }


def _table(rows, widths, styles, *, header=True):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), styles["font"]),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BAC7D5")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244F73")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ])
    for row in range(1 if header else 0, len(rows)):
        if row % 2 == 0:
            commands.append(
                ("BACKGROUND", (0, row), (-1, row), colors.HexColor("#EFF4F8")))
    table.setStyle(TableStyle(commands))
    return table


def _footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#738094"))
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{document.page}")
    canvas.restoreState()


def _wrapped_hash(value, styles):
    chunks = [value[index:index + 16] for index in range(0, len(value), 16)]
    return Paragraph("<br/>".join(chunks), styles["small"])


def build_report(manifest_path=DEFAULT_MANIFEST, output_path=DEFAULT_OUTPUT):
    manifest = validate_report_manifest(json.loads(
        Path(manifest_path).read_text(encoding="utf-8")))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    fold = manifest["tasks"]["fold_box"]
    seal = manifest["tasks"]["seal_bag"]
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title="PiperX 双任务 Fixed-Time 完全跟随报告")
    story = [
        Spacer(1, 12 * mm),
        Paragraph("PiperX 双任务 Fixed-Time 完全跟随报告", styles["title"]),
        Paragraph(
            "Fold_Box 161044 + Seal_Bag 161504 | 原始 60 Hz 时间戳 | "
            "无重定时、无插帧、无新增时长", styles["subtitle"]),
        Paragraph("结论", styles["h1"]),
        Paragraph(
            "本版本不是把重定时视频加速。每个视频直接使用 source_time_s 和 "
            "source_qpos：fixed_time_s 与 source_time_s 逐元素相同，fixed_time_qpos "
            "与 source_qpos 逐元素相同。两任务仍保持 calibrated TCP 1 mm / 0.5°、"
            "100% 位姿覆盖和零源路径碰撞。", styles["body"]),
        _table([
            ["任务", "固定时长", "位姿覆盖", "碰撞帧", "视频"],
            ["Fold_Box", f"{fold['timing']['fixed_time_duration_s']:.6f} s",
             f"{fold['tracking']['reached_frames']}/{fold['timing']['source_frames']}",
             "0", f"{fold['video']['decode_check']['frame_count']} 帧 / 30 fps"],
            ["Seal_Bag", f"{seal['timing']['fixed_time_duration_s']:.6f} s",
             f"{seal['tracking']['reached_frames']}/{seal['timing']['source_frames']}",
             "0", f"{seal['video']['decode_check']['frame_count']} 帧 / 30 fps"],
        ], [31*mm, 35*mm, 35*mm, 25*mm, 41*mm], styles),
        Spacer(1, 5 * mm),
        Paragraph("重要限制", styles["h1"]),
        Paragraph(
            "Fixed-time 只证明运动学跟随。它不满足配置中的 PiperX 关节速度 "
            "1 rad/s 与加速度 4 rad/s² 限制，因此不是可直接下发真机的执行轨迹。"
            "零 MuJoCo 碰撞也不是真机安全许可。", styles["body"]),
        PageBreak(),
        Paragraph("1. Fixed-time 身份证据", styles["h1"]),
        _table([
            ["合同项", "Fold_Box", "Seal_Bag"],
            ["timing_mode", "fixed_source_time", "fixed_source_time"],
            ["timestamp_identity", "True", "True"],
            ["qpos_identity", "True", "True"],
            ["retiming_applied", "False", "False"],
            ["inserted_frames", "0", "0"],
            ["added_duration_s", "0.0", "0.0"],
        ], [57*mm, 55*mm, 55*mm], styles),
        Paragraph("1.1 动力学审计", styles["h2"]),
        _table([
            ["任务", "最大速度", "最大加速度", "合规帧", "动力学门"],
            ["Fold_Box", f"{fold['dynamics']['maximum_velocity_rad_s']:.6f} rad/s",
             f"{fold['dynamics']['maximum_acceleration_rad_s2']:.6f} rad/s²",
             f"{fold['dynamics']['accepted_frames']}/{fold['timing']['source_frames']}",
             "不通过"],
            ["Seal_Bag", f"{seal['dynamics']['maximum_velocity_rad_s']:.6f} rad/s",
             f"{seal['dynamics']['maximum_acceleration_rad_s2']:.6f} rad/s²",
             f"{seal['dynamics']['accepted_frames']}/{seal['timing']['source_frames']}",
             "不通过"],
        ], [28*mm, 42*mm, 45*mm, 27*mm, 25*mm], styles),
        Paragraph(
            "Fold_Box 最大速度约为限制的 13.9 倍，最大加速度约为限制的 "
            "199.8 倍；Seal_Bag 分别约为 8.8 倍和 135.8 倍。该结果如实保留，"
            "不会通过放宽限制或隐藏超限来标记为可执行。", styles["body"]),
        Paragraph("1.2 跟踪与碰撞", styles["h2"]),
        Paragraph(
            "跟踪基准仍为 registered_resampled_calibrated_tcp。fixed-time NPZ "
            "保留 raw hand、calibrated target、actual TCP、误差、候选数、腕部调度、"
            "source velocity/acceleration 和零碰撞源路径证据。", styles["body"]),
        PageBreak(),
        Paragraph("2. 工件与使用边界", styles["h1"]),
        Paragraph(
            "每项任务包含 fixed-time summary、trajectory NPZ、scene XML、MP4 和 "
            "video provenance。manifest 将两项任务绑定到同一 schema，并明确 "
            "retiming_applied=false。", styles["body"]),
        _table([
            ["任务", "视频 SHA-256", "轨迹 SHA-256"],
            ["Fold_Box", _wrapped_hash(fold["artifacts"]["video_mp4"]["sha256"], styles),
             _wrapped_hash(fold["artifacts"]["trajectory_npz"]["sha256"], styles)],
            ["Seal_Bag", _wrapped_hash(seal["artifacts"]["video_mp4"]["sha256"], styles),
             _wrapped_hash(seal["artifacts"]["trajectory_npz"]["sha256"], styles)],
        ], [28*mm, 69*mm, 70*mm], styles),
        Paragraph("2.1 正确解释", styles["h2"]),
        Paragraph(
            "可以使用本版本检查原采集节奏下的视觉跟随、姿态分支和碰撞几何。"
            "不得将其标记为满足 PiperX 动力学的真机轨迹。若后续要求真机在相同"
            "节奏执行，需要提高硬件能力、重新采集更平滑轨迹或重新定义任务，而"
            "不是删除本报告中的超限证据。", styles["body"]),
        Paragraph(
            "报告由 fixed_time_manifest.json 生成，并在生成前强制检查两任务、"
            "时间与 qpos 身份、零插帧、零新增时长、完整覆盖、零碰撞和动力学"
            "不通过披露。", styles["small"]),
    ]
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
