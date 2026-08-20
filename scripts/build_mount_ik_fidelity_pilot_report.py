"""Build and validate the isolated mount/IK fidelity pilot report."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CACHE_ARRAYS = (
    "success", "failure_reason", "table_collision", "self_collision",
    "edge_collision", "joint_discontinuity", "recovery_mode",
)


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _failure_counts(cache_path: str | Path) -> dict[str, int]:
    with np.load(_absolute(cache_path)) as cache:
        reasons = np.asarray(cache["failure_reason"]).astype(str)
        counts = Counter(reason for reason in reasons if reason != "none")
        for name in ("table_collision", "self_collision", "edge_collision",
                     "joint_discontinuity"):
            count = int(np.count_nonzero(cache[name]))
            if count:
                counts[name] = max(counts.get(name, 0), count)
        if "recovery_mode" in cache:
            recovery = np.asarray(cache["recovery_mode"]).astype(str)
            for mode, count in Counter(
                    value for value in recovery if value not in {"none", "initialize"}
            ).items():
                counts[f"solver:{mode}"] = int(count)
    return dict(sorted(counts.items()))


def _video_decodes(path: Path) -> bool:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-f", "null", "-",
    ]
    return subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False).returncode == 0


def validate_pilot_completeness(
    rows: Iterable[dict], *, videos: Mapping[str, str | Path],
    expected_robots: Iterable[str], verify_video_decode: bool = True,
) -> list[str]:
    """Return actionable completeness errors; an empty list means complete."""
    by_robot = {str(row.get("robot")): row for row in rows}
    errors: list[str] = []
    for robot in expected_robots:
        row = by_robot.get(robot)
        if row is None:
            errors.append(f"missing robot result: {robot}")
            continue
        for key in ("cross_fidelity", "stage_counts", "winner",
                    "native_winner", "cache"):
            if key not in row:
                errors.append(f"{robot}: missing {key}")
        winner = row.get("winner", {})
        if len(winner.get("mount", [])) != 6:
            errors.append(f"{robot}: winner mount must contain X/Y/Z/Tilt/Yaw/Roll")
        cache_path = _absolute(row.get("cache", ""))
        if not cache_path.is_file():
            errors.append(f"{robot}: missing strict cache {cache_path}")
        else:
            try:
                with np.load(cache_path) as cache:
                    missing = [key for key in REQUIRED_CACHE_ARRAYS if key not in cache]
                    if missing:
                        errors.append(f"{robot}: cache missing arrays {missing}")
                    elif len(cache["success"]) == 0:
                        errors.append(f"{robot}: strict cache is empty")
            except (OSError, ValueError) as exc:
                errors.append(f"{robot}: unreadable strict cache: {exc}")
        video_value = videos.get(robot)
        if video_value is None:
            errors.append(f"{robot}: missing video path")
            continue
        video_path = _absolute(video_value)
        if not video_path.is_file() or video_path.stat().st_size == 0:
            errors.append(f"{robot}: missing or empty video {video_path}")
        elif verify_video_decode and not _video_decodes(video_path):
            errors.append(f"{robot}: video does not decode {video_path}")
    return errors


def build_report_payload(
    rows: Iterable[dict], *, videos: Mapping[str, str | Path],
    expected_robots: Iterable[str] | None = None,
    verify_video_decode: bool = False,
) -> dict:
    rows = list(rows)
    expected = tuple(expected_robots or [row["robot"] for row in rows])
    errors = validate_pilot_completeness(
        rows, videos=videos, expected_robots=expected,
        verify_video_decode=verify_video_decode)
    robot_rows = []
    for row in rows:
        winner = row["winner"]
        native = row["native_winner"]
        robot_rows.append({
            "robot": row["robot"],
            "episode_success": bool(winner["episode_success"]),
            "frame_coverage": float(winner["frame_coverage"]),
            "mount": [float(value) for value in winner["mount"]],
            "position_error_m_p95": float(winner.get("position_error_m_p95", np.nan)),
            "orientation_error_rad_p95": float(
                winner.get("orientation_error_rad_p95", np.nan)),
            "native_episode_success": bool(native["episode_success"]),
            "native_frame_coverage": float(native["frame_coverage"]),
            "native_mount": [float(value) for value in native["mount"]],
            "native_position_error_m_p95": float(
                native.get("position_error_m_p95", np.nan)),
            "native_orientation_error_rad_p95": float(
                native.get("orientation_error_rad_p95", np.nan)),
            "cross_fidelity": row["cross_fidelity"],
            "stage_counts": row["stage_counts"],
            "failure_counts": _failure_counts(row["cache"]),
            "cache": str(_absolute(row["cache"])),
            "video": str(_absolute(videos[row["robot"]])),
            "elapsed_s": float(row.get("elapsed_s", np.nan)),
        })
    common_ranked = sorted(
        robot_rows, key=lambda item: (
            item["episode_success"], item["frame_coverage"],
            -item["position_error_m_p95"], -item["orientation_error_rad_p95"]),
        reverse=True)
    native_ranked = sorted(
        robot_rows, key=lambda item: (
            item["native_episode_success"], item["native_frame_coverage"],
            -item["native_position_error_m_p95"],
            -item["native_orientation_error_rad_p95"]),
        reverse=True)
    conclusions = []
    if common_ranked:
        leader = common_ranked[0]
        conclusions.append(
            f"Common-limit leader: {leader['robot']}; "
            f"episode={'PASS' if leader['episode_success'] else 'FAIL'}, "
            f"coverage={100 * leader['frame_coverage']:.2f}%.")
        native_leader = native_ranked[0]
        conclusions.append(
            f"Native-limit leader: {native_leader['robot']}; "
            f"episode={'PASS' if native_leader['native_episode_success'] else 'FAIL'}, "
            f"coverage={100 * native_leader['native_frame_coverage']:.2f}%.")
        for item in robot_rows:
            if item["failure_counts"]:
                reason, count = max(
                    item["failure_counts"].items(), key=lambda value: value[1])
                conclusions.append(
                    f"{item['robot']} dominant failure evidence: {reason} ({count} frames).")
    return {
        "complete": not errors,
        "completeness_errors": errors,
        "expected_robots": list(expected),
        "robots": robot_rows,
        "conclusions": conclusions,
        "method_changes": [
            "All 4096 mounts participate in the diverse 256-candidate shortlist; proxy score is not a hard rejection gate.",
            "Strict windows and full episodes use the official model/TCP with real per-frame multi-branch IK.",
            "Path selection uses real timestamps, an effective finite horizon, state collision, and swept collision.",
            "Joint discontinuity is separated from pose failure and collision-blocked recovery.",
            "Common-limit and official native-limit finalists are evaluated separately; failures are rendered.",
        ],
    }


def _write_figures(payload: dict, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    robots = [row["robot"] for row in payload["robots"]]
    colors = ["#15803d" if row["episode_success"] else "#c2410c"
              for row in payload["robots"]]
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    coverage = [100 * row["frame_coverage"] for row in payload["robots"]]
    native_coverage = [100 * row["native_frame_coverage"] for row in payload["robots"]]
    x = np.arange(len(robots)); width = .36
    axes[0].bar(x - width / 2, coverage, width, color=colors, label="Common")
    axes[0].bar(x + width / 2, native_coverage, width,
                color="#2563eb", label="Native")
    axes[0].set_xticks(x, robots); axes[0].legend()
    axes[0].set_ylim(0, 105); axes[0].set_ylabel("Frame coverage (%)")
    axes[0].set_title("Strict full-episode follow")
    for index, value in enumerate(coverage):
        axes[0].text(index - width / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=8)
    for index, value in enumerate(native_coverage):
        axes[0].text(index + width / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=8)
    spearman = [row["cross_fidelity"].get("spearman_rank_correlation", np.nan)
                for row in payload["robots"]]
    recall = [row["cross_fidelity"].get("top_k_recall", np.nan)
              for row in payload["robots"]]
    x = np.arange(len(robots)); width = .36
    axes[1].bar(x - width / 2, spearman, width, label="Spearman", color="#2563eb")
    axes[1].bar(x + width / 2, recall, width, label="Recall@K", color="#7c3aed")
    axes[1].set_xticks(x, robots); axes[1].set_ylim(-1.05, 1.05)
    axes[1].axhline(0, color="#444", linewidth=.7)
    axes[1].set_title("Proxy-to-strict fidelity"); axes[1].legend()
    path = figure_dir / "outcome_and_fidelity.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    reason_names = sorted({name for row in payload["robots"]
                           for name in row["failure_counts"]})
    if not reason_names:
        reason_names = ["none"]
    matrix = np.asarray([[row["failure_counts"].get(name, 0) for name in reason_names]
                         for row in payload["robots"]], dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, .8 * len(reason_names)), 3.7),
                           constrained_layout=True)
    image = ax.imshow(matrix, cmap="OrRd", aspect="auto")
    ax.set_xticks(range(len(reason_names)), reason_names, rotation=25, ha="right")
    ax.set_yticks(range(len(robots)), robots); ax.set_title("Strict failure reasons (frames)")
    for i in range(len(robots)):
        for j in range(len(reason_names)):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=.8)
    path = figure_dir / "failure_reason_heatmap.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    mounts = np.asarray([row["mount"] for row in payload["robots"]])
    labels = ("X m", "Y m", "Z m", "Tilt deg", "Yaw deg", "Roll deg")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for column, (ax, label) in enumerate(zip(axes.flat, labels)):
        ax.bar(robots, mounts[:, column], color="#0891b2")
        ax.axhline(0, color="#555", linewidth=.7); ax.set_title(label)
    path = figure_dir / "winner_mounts.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    return paths


def _markdown(payload: dict, figures: Iterable[Path], output: Path) -> str:
    status = "通过" if payload["complete"] else "未通过"
    lines = [
        "# Mount / IK Fidelity Pilot Report", "",
        f"完整性检查：**{status}**", "",
        "本报告只描述隔离的 xArm6/OpenArm 机制试验，不写入正式十臂排名。",
        "窗口阶段不能声明完整 episode 成功；最终结论来自真实 URDF 的完整时间序列复评。",
        "",
    ]
    if payload["completeness_errors"]:
        lines.extend(["## 完整性问题", ""] +
                     [f"- {error}" for error in payload["completeness_errors"]] + [""])
    lines.extend([
        "## 结果", "",
        "| Robot | Common | Native | Common cov. | Native cov. | P95 pos. | P95 ori. | Spearman | Recall@K | Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["robots"]:
        fidelity = row["cross_fidelity"]
        lines.append(
            f"| {row['robot']} | {'PASS' if row['episode_success'] else 'FAIL'} | "
            f"{'PASS' if row['native_episode_success'] else 'FAIL'} | "
            f"{100*row['frame_coverage']:.2f}% | {100*row['native_frame_coverage']:.2f}% | "
            f"{row['position_error_m_p95']:.4f} m | "
            f"{row['orientation_error_rad_p95']:.4f} rad | "
            f"{fidelity.get('spearman_rank_correlation', float('nan')):.3f} | "
            f"{fidelity.get('top_k_recall', float('nan')):.3f} | {row['elapsed_s']:.1f} s |")
    lines.extend(["", "## 最优安装与失败原因", ""])
    for row in payload["robots"]:
        x, y, z, tilt, yaw, roll = row["mount"]
        causes = ", ".join(f"{key}={value}" for key, value in row["failure_counts"].items()) or "none"
        video = Path(row["video"])
        try:
            video_label = video.relative_to(output).as_posix()
        except ValueError:
            video_label = video.as_posix()
        lines.extend([
            f"- **{row['robot']}**: X={x:.3f}, Y={y:.3f}, Z={z:.3f} m; "
            f"Tilt={tilt:.1f}°, Yaw={yaw:.1f}°, Roll={roll:.1f}°. "
            f"Failures: {causes}. [MuJoCo video]({video_label})",
        ])
    lines.extend(["", "## 可视化", ""])
    for figure in figures:
        lines.append(f"![{figure.stem}]({figure.relative_to(output).as_posix()})")
        lines.append("")
    lines.extend([
        "## 结论口径", "",
        "- coarse 胶囊/代理结果仅用于排序和分散保留，不作为完整 episode 成功证据。",
        "- 最终结果使用官方模型、真实 TCP、逐帧多分支 IK、真实时间戳、关节速度/跳变约束，以及状态和扫掠碰撞检查。",
        "- 失败视频仍被保留，用于定位不可达、姿态误差、关节不连续和碰撞失败。",
    ])
    lines.extend(["", "## Pilot conclusions", ""])
    lines.extend([f"- {conclusion}" for conclusion in payload["conclusions"]])
    lines.extend(["", "## Changes from the old flow", ""])
    lines.extend([f"- {change}" for change in payload["method_changes"]])
    return "\n".join(lines) + "\n"


def _write_pdf(payload: dict, figures: Iterable[Path], output: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError(
            "Microsoft YaHei is required for the bilingual pilot PDF")
    pdfmetrics.registerFont(TTFont(
        "PilotSans", str(font_path), subfontIndex=0))
    pdf_dir = output / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / "mount_ik_fidelity_pilot_report.pdf"
    document = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=18 * mm, bottomMargin=17 * mm,
        title="Mount / IK Fidelity Pilot Report")
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "PilotTitle", parent=styles["Title"], fontName="PilotSans",
        fontSize=20, leading=25, textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER, spaceAfter=7 * mm)
    heading = ParagraphStyle(
        "PilotHeading", parent=styles["Heading2"], fontName="PilotSans",
        fontSize=13, leading=17, textColor=colors.HexColor("#0f4c5c"),
        spaceBefore=5 * mm, spaceAfter=3 * mm)
    body = ParagraphStyle(
        "PilotBody", parent=styles["BodyText"], fontName="PilotSans",
        fontSize=9.5, leading=14, textColor=colors.HexColor("#1f2937"),
        spaceAfter=2.5 * mm)

    story = [
        Paragraph("Mount / IK Fidelity Pilot", title),
        Paragraph(
            "隔离机制试验：xArm6 / OpenArm，官方模型与 TCP，真实逐帧多分支 IK，"
            "状态与扫掠碰撞检查。该报告不会改写正式十臂排名。", body),
        Paragraph(
            "Artifact completeness: " + ("PASS" if payload["complete"] else "FAIL"),
            heading),
    ]
    if payload["completeness_errors"]:
        for error in payload["completeness_errors"]:
            story.append(Paragraph(f"- {error}", body))
    story.extend([Paragraph("Strict full-episode results", heading)])
    table_data = [[
        "Robot", "Common", "Native", "Common cov", "Native cov",
        "P95 pos", "P95 ori", "Spearman", "Recall@K"]]
    for row in payload["robots"]:
        fidelity = row["cross_fidelity"]
        table_data.append([
            row["robot"], "PASS" if row["episode_success"] else "FAIL",
            "PASS" if row["native_episode_success"] else "FAIL",
            f"{100 * row['frame_coverage']:.2f}%",
            f"{100 * row['native_frame_coverage']:.2f}%",
            f"{row['position_error_m_p95']:.4f}",
            f"{row['orientation_error_rad_p95']:.4f}",
            f"{fidelity.get('spearman_rank_correlation', float('nan')):.3f}",
            f"{fidelity.get('top_k_recall', float('nan')):.3f}",
        ])
    table = Table(table_data, colWidths=[20*mm, 17*mm, 17*mm, 21*mm, 21*mm,
                                        20*mm, 20*mm, 19*mm, 19*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164e63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#94a3b8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f1f5f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 3 * mm), Paragraph("Winner mounts and failures", heading)])
    for row in payload["robots"]:
        x, y, z, tilt, yaw, roll = row["mount"]
        causes = ", ".join(
            f"{name}={count}" for name, count in row["failure_counts"].items()) or "none"
        story.append(Paragraph(
            f"{row['robot']}: X={x:.3f}, Y={y:.3f}, Z={z:.3f} m; "
            f"Tilt={tilt:.1f} deg, Yaw={yaw:.1f} deg, Roll={roll:.1f} deg. "
            f"Failure frames: {causes}.", body))
    story.append(PageBreak())
    for index, figure in enumerate(figures):
        story.append(Paragraph(figure.stem.replace("_", " ").title(), heading))
        image = Image(str(figure))
        maximum_width, maximum_height = 177 * mm, 105 * mm
        scale = min(maximum_width / image.imageWidth,
                    maximum_height / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        story.extend([image, Spacer(1, 4 * mm)])
        if index == 1 and len(list(figures)) > 2:
            story.append(PageBreak())
    story.extend([
        Paragraph("Interpretation", heading),
        Paragraph(
            "The coarse capsule/proxy stage is a ranking and diversity mechanism only. "
            "Window screening cannot claim episode success. Final outcomes require every "
            "retained frame and transition to satisfy pose, continuity, velocity, joint-limit, "
            "state-collision, and swept-collision constraints.", body),
    ])
    story.append(Paragraph("Pilot conclusions", heading))
    for conclusion in payload["conclusions"]:
        story.append(Paragraph(f"- {conclusion}", body))
    story.append(Paragraph("Changes from the old flow", heading))
    for change in payload["method_changes"]:
        story.append(Paragraph(f"- {change}", body))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(16 * mm, 9 * mm, "Mount / IK Fidelity Pilot")
        canvas.drawRightString(194 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def write_report(
    matrix_path: Path, output: Path, *, expected_robots: Iterable[str],
    verify_video_decode: bool = True,
) -> dict:
    rows = json.loads(matrix_path.read_text(encoding="utf-8"))
    videos = {robot: output / "videos" / f"{robot}.mp4" for robot in expected_robots}
    payload = build_report_payload(
        rows, videos=videos, expected_robots=expected_robots,
        verify_video_decode=verify_video_decode)
    figures = _write_figures(payload, output)
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        _markdown(payload, figures, output), encoding="utf-8")
    payload["pdf"] = str(_write_pdf(payload, figures, output))
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robots", nargs="+", required=True)
    parser.add_argument("--skip-video-decode", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = write_report(
        output / "matrix.json", output, expected_robots=args.robots,
        verify_video_decode=not args.skip_video_decode)
    print(json.dumps({"complete": payload["complete"],
                      "errors": payload["completeness_errors"]}, ensure_ascii=False))
    if not payload["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
