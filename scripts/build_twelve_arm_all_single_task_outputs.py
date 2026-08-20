"""Build final tables, figures, comparison videos, HTML, and PDF for the 12x12 study."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/single_arm/twelve_arm_all_single_tasks"
ROBOTS = ("doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5",
          "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "nero",
          "openarm", "piperx", "willow")
TASKS = ("cap-left", "in-case-left", "open-box", "open-box-2", "open-box-3",
         "pick-from-high-left", "pick-right-left", "stick-battery", "toss-high",
         "tube", "tube-left-random", "tube-left-upright")


def load_rows() -> list[dict]:
    rows = []
    for robot in ROBOTS:
        for task in TASKS:
            audit = ROOT / "videos/single_arm/strict_cache/local" / robot / f"{task}.json"
            if not audit.is_file():
                raise FileNotFoundError(audit)
            row = json.loads(audit.read_text(encoding="utf-8"))
            row["robot"] = robot; row["task"] = task
            rows.append(row)
    return rows


def write_tables(rows: list[dict]) -> list[dict]:
    raw = OUT / "all_results.csv"
    fields = ("robot", "task", "status", "episode_success", "frame_coverage",
              "longest_failure_run_frames", "table_collision_frames", "self_collision_frames",
              "base_x_m", "base_y_m", "base_z_m", "tilt_deg", "yaw_deg", "roll_deg",
              "position_rmse_mm", "orientation_rmse_deg", "joint_discontinuity_frames")
    with raw.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            base = row["base_xyz_m"]; failures = row.get("failure_reasons", {}).get("affected_frame_counts", {})
            writer.writerow({
                "robot": row["robot"], "task": row["task"], "status": row.get("status"),
                "episode_success": row.get("episode_success", False), "frame_coverage": row.get("frame_coverage", 0),
                "longest_failure_run_frames": row.get("longest_failure_run_frames", 0),
                "table_collision_frames": row.get("table_collision_frames", 0),
                "self_collision_frames": row.get("self_collision_frames", 0),
                "base_x_m": base[0], "base_y_m": base[1], "base_z_m": base[2],
                "tilt_deg": row.get("tilt_deg", 0), "yaw_deg": row.get("yaw_deg", 0),
                "roll_deg": row.get("roll_deg", 0),
                "position_rmse_mm": 1000 * row["position_error_m"]["mean"],
                "orientation_rmse_deg": row["orientation_error_deg"]["mean"],
                "joint_discontinuity_frames": failures.get("joint_discontinuity", 0),
            })
    audit = json.loads((ROOT / "reports/single_arm/model_audit.json").read_text(encoding="utf-8"))["robots"]
    summary = []
    for robot in ROBOTS:
        group = [row for row in rows if row["robot"] == robot]
        summary.append({
            "robot": robot,
            "active_dof": int(audit[robot]["active_dof"]),
            "native_reach_m": float(audit[robot]["native_sampled_max_reach_m"]),
            "successful_episodes": sum(bool(row.get("episode_success")) for row in group),
            "episodes": len(group),
            "episode_success_rate": float(np.mean([bool(row.get("episode_success")) for row in group])),
            "mean_frame_coverage": float(np.mean([row.get("frame_coverage", 0) for row in group])),
            "position_rmse_mm": float(np.sqrt(np.mean([(1000 * row["position_error_m"]["mean"]) ** 2 for row in group]))),
            "orientation_rmse_deg": float(np.sqrt(np.mean([row["orientation_error_deg"]["mean"] ** 2 for row in group]))),
        })
    summary.sort(key=lambda row: (row["episode_success_rate"], row["mean_frame_coverage"]), reverse=True)
    (OUT / "summary.json").write_text(json.dumps({"robots": summary, "tasks": list(TASKS),
        "matrix_jobs": len(rows)}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)
    return summary


def save_figures(rows: list[dict], summary: list[dict]) -> list[Path]:
    figures = OUT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    order = [row["robot"] for row in summary]
    success = [100 * row["episode_success_rate"] for row in summary]
    coverage = [100 * row["mean_frame_coverage"] for row in summary]
    fig, ax = plt.subplots(figsize=(11, 6)); y = np.arange(len(order))
    ax.barh(y + .18, success, .34, label="Complete episode success")
    ax.barh(y - .18, coverage, .34, label="Mean frame coverage")
    ax.set(yticks=y, yticklabels=order, xlim=(0, 100), xlabel="Percent")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.25); ax.legend(); fig.tight_layout()
    rank = figures / "success_ranking.png"; fig.savefig(rank, dpi=200); plt.close(fig)
    matrix = np.asarray([[next(r for r in rows if r["robot"] == robot and r["task"] == task)["frame_coverage"]
                          for robot in order] for task in TASKS])
    fig, ax = plt.subplots(figsize=(15, 7)); image = ax.imshow(100 * matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set(xticks=np.arange(len(order)), xticklabels=order, yticks=np.arange(len(TASKS)), yticklabels=TASKS)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right"); fig.colorbar(image, ax=ax, label="Frame coverage (%)")
    fig.tight_layout(); heat = figures / "task_robot_heatmap.png"; fig.savefig(heat, dpi=200); plt.close(fig)
    reasons = ("joint_discontinuity", "self_collision", "table_collision", "position", "orientation", "position_and_orientation")
    reason_matrix = []
    for robot in order:
        totals = []
        for reason in reasons:
            totals.append(sum(row.get("failure_reasons", {}).get("affected_frame_counts", {}).get(reason, 0)
                              for row in rows if row["robot"] == robot))
        reason_matrix.append(totals)
    fig, ax = plt.subplots(figsize=(9, 7)); im = ax.imshow(np.log1p(reason_matrix), cmap="magma", aspect="auto")
    ax.set(yticks=np.arange(len(order)), yticklabels=order, xticks=np.arange(len(reasons)), xticklabels=reasons)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right"); fig.colorbar(im, ax=ax, label="log(1 + affected frames)")
    fig.tight_layout(); failure = figures / "failure_reason_heatmap.png"; fig.savefig(failure, dpi=200); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8)); names = ("X (m)", "Y (m)", "Z (m)", "Tilt (deg)", "Yaw (deg)", "Roll (deg)")
    for index, (axis, name) in enumerate(zip(axes.flat, names)):
        for robot in order:
            group = [r for r in rows if r["robot"] == robot]
            values = [r["base_xyz_m"][index] if index < 3 else r[("tilt_deg", "yaw_deg", "roll_deg")[index-3]] for r in group]
            axis.scatter([order.index(robot)] * len(values), values, s=10, alpha=.55)
        axis.set_title(name); axis.grid(alpha=.2)
        if index >= 3: axis.set_xticks(np.arange(len(order)), order, rotation=70, fontsize=7)
        else: axis.set_xticks([])
    fig.tight_layout(); mounts = figures / "optimal_mount_distribution.png"; fig.savefig(mounts, dpi=200); plt.close(fig)
    return [rank, heat, failure, mounts]


def render_comparisons() -> list[Path]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe(); destination = OUT / "comparison_videos"
    destination.mkdir(parents=True, exist_ok=True); outputs = []
    for task in TASKS:
        inputs = [OUT / "videos" / robot / f"{task}.mp4" for robot in ROBOTS]
        if not all(path.is_file() for path in inputs):
            raise FileNotFoundError(f"missing videos for {task}")
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        for path in inputs: command += ["-i", str(path)]
        filters = [f"[{i}:v]scale=480:270[v{i}]" for i in range(12)]
        layout = "|".join(f"{(i%4)*480}_{(i//4)*270}" for i in range(12))
        filters.append("".join(f"[v{i}]" for i in range(12)) + f"xstack=inputs=12:layout={layout}:fill=black[out]")
        output = destination / f"{task}__12arm_comparison.mp4"
        command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-shortest", str(output)]
        subprocess.run(command, check=True); outputs.append(output)
    concat = destination / "concat.txt"
    concat.write_text("\n".join(f"file '{path.as_posix()}'" for path in outputs), encoding="utf-8")
    overall = destination / "all_single_tasks__12arm_overview.mp4"
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(concat), "-c", "copy", str(overall)], check=True)
    return outputs + [overall]


def build_report(rows: list[dict], summary: list[dict], figures: list[Path], videos: list[Path]) -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_name = "Helvetica"
    if font_path.is_file(): pdfmetrics.registerFont(TTFont("MicrosoftYaHei", str(font_path))); font_name = "MicrosoftYaHei"
    styles = getSampleStyleSheet(); title = ParagraphStyle("TitleCN", parent=styles["Title"], fontName=font_name, alignment=TA_CENTER)
    heading = ParagraphStyle("HeadingCN", parent=styles["Heading1"], fontName=font_name)
    body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName=font_name, leading=16)
    pdf = OUT / "twelve_arm_all_single_tasks_report.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=landscape(A4), rightMargin=14*mm, leftMargin=14*mm,
                            topMargin=12*mm, bottomMargin=12*mm)
    story = [Paragraph("12臂全单手任务 MuJoCo 轨迹跟随实验", title), Spacer(1, 6*mm),
             Paragraph("真实 URDF/MJCF · 原生臂长与自由度 · 每任务独立6维安装搜索 · 完整episode成功率", body),
             Spacer(1, 5*mm)]
    table_data = [["排名", "机械臂", "DOF", "原生臂展", "完整成功", "任务数", "帧覆盖率", "位置RMSE", "姿态RMSE"]]
    for index, row in enumerate(summary, 1):
        table_data.append([index, row["robot"], row["active_dof"], f'{row["native_reach_m"]:.3f} m',
                           f'{row["episode_success_rate"]:.1%}', row["episodes"],
                           f'{row["mean_frame_coverage"]:.1%}', f'{row["position_rmse_mm"]:.1f} mm',
                           f'{row["orientation_rmse_deg"]:.1f}°'])
    table = Table(table_data, repeatRows=1); table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), font_name), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D9EAF7")),
        ("GRID", (0,0), (-1,-1), .35, colors.grey), ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 8), ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
    story += [table, PageBreak()]
    captions = ("总体成功率排名", "任务×机械臂帧覆盖率", "失败原因主次程度", "最优基座六维分布")
    for path, caption in zip(figures, captions):
        story += [Paragraph(caption, heading), Image(str(path), width=245*mm, height=135*mm), PageBreak()]
    reach_rho = float(spearmanr([row["native_reach_m"] for row in summary],
                                [row["episode_success_rate"] for row in summary]).statistic)
    dof_rho = float(spearmanr([row["active_dof"] for row in summary],
                              [row["episode_success_rate"] for row in summary]).statistic)
    story += [Paragraph("臂展、轴数与构型观察", heading), Paragraph(
        f"在本次12臂样本内，原生最大臂展与完整episode成功率的 Spearman 相关系数为 {reach_rho:.3f}，"
        f"主动自由度与成功率的相关系数为 {dof_rho:.3f}。这些是描述性相关而非因果结论；"
        "任务姿态分布、腕部轴布局、关节范围和可用安装区域会共同影响结果。Panda锁J3与原版Panda的成对结果用于观察冗余自由度受限后的变化。", body),
        PageBreak(), Paragraph("实验协议与产物", heading), Paragraph(
        f"矩阵包含12种机械臂、12个纯单手任务，共{len(rows)}个完整episode评估。双手任务未进入本矩阵。"
        "成功定义为完整episode所有目标帧均满足位置、姿态、关节连续性及真实模型碰撞约束；帧覆盖率仅作为次级指标。"
        "轨迹按真实时间均匀插值，Local任务高度使用机械臂无关的统一桌面净空策略。", body), Spacer(1, 4*mm),
        Paragraph(f"视频：{len(rows)}个逐臂逐任务视频、{len(TASKS)}个任务级12臂对比视频和1个总体串联视频。原始结果见 all_results.csv。", body)]
    doc.build(story)
    html = OUT / "report.html"
    cards = "".join(f'<section><h2>{caption}</h2><img src="figures/{path.name}"></section>' for path, caption in zip(figures, captions))
    html.write_text(f"<!doctype html><meta charset='utf-8'><title>12-arm Local study</title><style>body{{font:16px Arial;margin:32px;color:#17202a}}img{{max-width:100%}}section{{margin:40px 0}}table{{border-collapse:collapse}}td,th{{padding:7px;border:1px solid #aaa}}</style><h1>12臂全单手任务 MuJoCo 轨迹跟随实验</h1>{cards}<p>PDF: {pdf.name}</p><p>总体视频: {videos[-1].relative_to(OUT).as_posix()}</p>", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(); summary = write_tables(rows); figures = save_figures(rows, summary)
    videos = render_comparisons(); build_report(rows, summary, figures, videos)
    (OUT / "completion.json").write_text(json.dumps({"status": "complete", "jobs": len(rows),
        "figures": [str(p.relative_to(OUT)) for p in figures], "videos": len(videos),
        "pdf": "twelve_arm_all_single_tasks_report.pdf"}, indent=2), encoding="utf-8")
    print(OUT / "completion.json")


if __name__ == "__main__":
    main()
