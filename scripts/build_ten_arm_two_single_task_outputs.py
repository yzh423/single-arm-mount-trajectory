"""Build one unified report for the formal ten-arm, two-task experiment."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/single_arm/ten_arm_two_single_tasks"
ROBOTS = ("doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5",
          "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "openarm", "piperx")
TASKS = ("cap-left", "open-box-2")


def load_rows() -> list[dict]:
    rows = []
    for robot in ROBOTS:
        for task in TASKS:
            path = ROOT / "videos/single_arm/strict_cache/local" / robot / f"{task}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            row.update(robot=robot, task=task)
            rows.append(row)
    return rows


def build_data(rows: list[dict]) -> list[dict]:
    fields = ("robot", "task", "status", "episode_success", "frame_coverage",
              "longest_failure_run_frames", "table_collision_frames", "self_collision_frames",
              "base_x_m", "base_y_m", "base_z_m", "tilt_deg", "yaw_deg", "roll_deg",
              "position_rmse_mm", "orientation_rmse_deg", "joint_discontinuity_frames",
              "primary_failure_reason")
    with (OUT / "all_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            base = row["base_xyz_m"]
            failures = row.get("failure_reasons", {}).get("affected_frame_counts", {})
            primary = max(failures, key=failures.get) if failures and max(failures.values()) else "none"
            writer.writerow({
                "robot": row["robot"], "task": row["task"], "status": row.get("status"),
                "episode_success": row.get("episode_success", False),
                "frame_coverage": row.get("frame_coverage", 0),
                "longest_failure_run_frames": row.get("longest_failure_run_frames", 0),
                "table_collision_frames": row.get("table_collision_frames", 0),
                "self_collision_frames": row.get("self_collision_frames", 0),
                "base_x_m": base[0], "base_y_m": base[1], "base_z_m": base[2],
                "tilt_deg": row.get("tilt_deg", 0), "yaw_deg": row.get("yaw_deg", 0),
                "roll_deg": row.get("roll_deg", 0),
                "position_rmse_mm": 1000 * row["position_error_m"]["mean"],
                "orientation_rmse_deg": row["orientation_error_deg"]["mean"],
                "joint_discontinuity_frames": failures.get("joint_discontinuity", 0),
                "primary_failure_reason": primary,
            })
    audit = json.loads((ROOT / "reports/single_arm/model_audit.json").read_text(encoding="utf-8"))["robots"]
    summary = []
    for robot in ROBOTS:
        group = [r for r in rows if r["robot"] == robot]
        summary.append({
            "robot": robot, "active_dof": int(audit[robot]["active_dof"]),
            "native_reach_m": float(audit[robot]["native_sampled_max_reach_m"]),
            "successful_episodes": sum(bool(r.get("episode_success")) for r in group),
            "episodes": len(group),
            "episode_success_rate": float(np.mean([bool(r.get("episode_success")) for r in group])),
            "mean_frame_coverage": float(np.mean([r.get("frame_coverage", 0) for r in group])),
            "position_rmse_mm": float(np.sqrt(np.mean([(1000*r["position_error_m"]["mean"])**2 for r in group]))),
            "orientation_rmse_deg": float(np.sqrt(np.mean([r["orientation_error_deg"]["mean"]**2 for r in group]))),
        })
    summary.sort(key=lambda r: (r["episode_success_rate"], r["mean_frame_coverage"]), reverse=True)
    (OUT / "summary.json").write_text(json.dumps({"robots": summary, "tasks": list(TASKS), "matrix_jobs": len(rows)}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)
    return summary


def figures(rows: list[dict], summary: list[dict]) -> list[Path]:
    folder = OUT / "figures"; folder.mkdir(parents=True, exist_ok=True)
    order = [r["robot"] for r in summary]; y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y+.18, [100*r["episode_success_rate"] for r in summary], .34, label="Episode success")
    ax.barh(y-.18, [100*r["mean_frame_coverage"] for r in summary], .34, label="Frame coverage")
    ax.set(yticks=y, yticklabels=order, xlim=(0, 100), xlabel="Percent"); ax.invert_yaxis(); ax.legend(); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); rank = folder / "success_ranking.png"; fig.savefig(rank, dpi=200); plt.close(fig)
    matrix = np.array([[next(r for r in rows if r["robot"] == robot and r["task"] == task).get("frame_coverage", 0) for robot in order] for task in TASKS])
    fig, ax = plt.subplots(figsize=(13, 4)); im = ax.imshow(100*matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set(xticks=range(len(order)), xticklabels=order, yticks=range(len(TASKS)), yticklabels=TASKS); plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(im, ax=ax, label="Frame coverage (%)"); fig.tight_layout(); heat = folder / "task_robot_heatmap.png"; fig.savefig(heat, dpi=200); plt.close(fig)
    reasons = ("joint_discontinuity", "self_collision", "table_collision", "position", "orientation", "position_and_orientation")
    values = [[sum(r.get("failure_reasons", {}).get("affected_frame_counts", {}).get(reason, 0) for r in rows if r["robot"] == robot) for reason in reasons] for robot in order]
    fig, ax = plt.subplots(figsize=(9, 6)); im = ax.imshow(np.log1p(values), cmap="magma", aspect="auto")
    ax.set(yticks=range(len(order)), yticklabels=order, xticks=range(len(reasons)), xticklabels=reasons); plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(im, ax=ax, label="log(1 + affected frames)"); fig.tight_layout(); fail = folder / "failure_reason_heatmap.png"; fig.savefig(fail, dpi=200); plt.close(fig)
    return [rank, heat, fail]


def comparison_videos() -> list[Path]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe(); folder = OUT / "comparison_videos"; folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    for task in TASKS:
        paths = [OUT / "videos" / robot / f"{task}.mp4" for robot in ROBOTS]
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        for path in paths: command += ["-i", str(path)]
        filters = [f"[{i}:v]scale=480:270[v{i}]" for i in range(len(paths))]
        layout = "|".join(f"{(i%5)*480}_{(i//5)*270}" for i in range(len(paths)))
        filters.append("".join(f"[v{i}]" for i in range(len(paths))) + f"xstack=inputs={len(paths)}:layout={layout}:fill=black[out]")
        output = folder / f"{task}__10arm_comparison.mp4"
        command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", "-shortest", str(output)]
        subprocess.run(command, check=True); outputs.append(output)
    return outputs


def report(rows: list[dict], summary: list[dict], images: list[Path], videos: list[Path]) -> Path:
    font = "Helvetica"; font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file(): pdfmetrics.registerFont(TTFont("MicrosoftYaHei", str(font_path))); font = "MicrosoftYaHei"
    styles = getSampleStyleSheet(); title = ParagraphStyle("cn-title", parent=styles["Title"], fontName=font, alignment=TA_CENTER)
    heading = ParagraphStyle("cn-heading", parent=styles["Heading1"], fontName=font); body = ParagraphStyle("cn-body", parent=styles["BodyText"], fontName=font, leading=16)
    pdf = OUT / "ten_arm_two_single_tasks_report.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=landscape(A4), rightMargin=14*mm, leftMargin=14*mm, topMargin=12*mm, bottomMargin=12*mm)
    table_data = [["排名", "机械臂", "DOF", "原生臂展", "完整成功", "任务数", "帧覆盖率", "位置RMSE", "姿态RMSE"]]
    for i, row in enumerate(summary, 1): table_data.append([i, row["robot"], row["active_dof"], f'{row["native_reach_m"]:.3f} m', f'{row["episode_success_rate"]:.1%}', row["episodes"], f'{row["mean_frame_coverage"]:.1%}', f'{row["position_rmse_mm"]:.1f} mm', f'{row["orientation_rmse_deg"]:.1f}°'])
    table = Table(table_data, repeatRows=1); table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), font), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D9EAF7")), ("GRID", (0,0), (-1,-1), .35, colors.grey), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTSIZE", (0,0), (-1,-1), 8)]))
    story = [Paragraph("10臂×2个单手任务 MuJoCo 轨迹跟随实验", title), Spacer(1, 5*mm), Paragraph("排除 Willow、Big YAM、Nero；任务为 cap-left 与 open-box-2。采用真实URDF复筛、每任务独立六维安装搜索和完整episode成功标准。", body), Spacer(1, 4*mm), table, PageBreak()]
    captions = ("完整episode成功率与帧覆盖率排名", "任务×机械臂帧覆盖率", "失败原因程度热图")
    for path, caption in zip(images, captions): story += [Paragraph(caption, heading), Image(str(path), width=245*mm, height=130*mm), PageBreak()]
    story += [Paragraph("结论解释与后续实验", heading), Paragraph("排名首先按完整episode成功率，其次按帧覆盖率。失败原因和原始逐任务指标可在 all_results.csv 中逐项复核。两任务样本只支持初步构型比较；后续应扩展到全部纯单手任务，并按任务类型报告臂展、轴数与构型的分层效应。", body), Spacer(1, 4*mm), Paragraph(f"产物包括 {len(rows)} 个逐臂逐任务视频、{len(videos)} 个任务级10臂对比视频、原始表、汇总表和可视化。", body)]
    doc.build(story); return pdf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(); summary = build_data(rows); images = figures(rows, summary); videos = comparison_videos(); pdf = report(rows, summary, images, videos)
    completion = {"status": "complete", "robots": list(ROBOTS), "tasks": list(TASKS), "jobs": len(rows), "per_task_videos": len(rows), "comparison_videos": len(videos), "pdf": pdf.name}
    (OUT / "completion.json").write_text(json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
