"""Build the final fixed-vs-legacy ten-arm report after all videos exist."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "reports/single_arm/ten_arm_two_single_tasks_handbook_fixed_4096"
LEGACY = ROOT / "reports/single_arm/twelve_arm_two_single_tasks"
FIG = NEW / "figures"
ROBOTS = ("doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5", "franka_panda",
          "franka_panda_locked_j3", "i2rt_yam", "openarm", "piperx")
TASKS = ("cap-left", "open-box-2")
REASONS = ("joint_discontinuity", "self_collision", "table_collision", "position",
           "orientation", "position_and_orientation")

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                      "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False,
                      "axes.spines.right": False, "figure.dpi": 160})


def load_legacy() -> list[dict]:
    """Load the immutable per-video JSONs from the original 12-arm experiment."""
    rows = []
    with (LEGACY / "results_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summary = {(r["robot"], r["task"]): r for r in csv.DictReader(stream)}
    for robot in ROBOTS:
        for task in TASKS:
            item = summary[(robot, task)]
            audit = json.loads((LEGACY / "videos" / robot / f"{task}.json").read_text(encoding="utf-8"))
            base = audit["base_xyz_m"]
            rows.append({"robot": robot, "task": task, "frames": int(item["frames"]),
                         "episode_success": item["episode_success"].lower() == "true",
                         "frame_coverage": float(item["frame_coverage"]),
                         "position_mean_mm": 1000 * float(item["position_error_mean_m"]),
                         "position_p95_mm": 1000 * audit["position_error_m"]["p95"],
                         "orientation_mean_deg": float(item["orientation_error_mean_deg"]),
                         "orientation_p95_deg": audit["orientation_error_deg"]["p95"],
                         "base_x_m": base[0], "base_y_m": base[1], "base_z_m": base[2],
                         "tilt_deg": audit.get("tilt_deg", 0), "yaw_deg": audit.get("yaw_deg", 0),
                         "roll_deg": audit.get("roll_deg", 0),
                         "joint_discontinuity": 0, "self_collision": int(item["self_collision_frames"]),
                         "table_collision": int(item["table_collision_frames"]), "position": 0,
                         "orientation": 0, "position_and_orientation": 0})
    return rows


def load_fixed() -> list[dict]:
    matrix = json.loads((NEW / "matrix.json").read_text(encoding="utf-8"))
    if len(matrix) != 20:
        raise RuntimeError(f"expected 20 matrix rows, got {len(matrix)}")
    rows = []
    for item in matrix:
        audit = Path(item["cache"]).with_suffix(".json")
        row = json.loads(audit.read_text(encoding="utf-8"))
        affected = row.get("failure_reasons", {}).get("affected_frame_counts", {})
        base = row["base_xyz_m"]
        rows.append({"robot": item["robot"], "task": item["task"], "frames": row["frames"],
                     "episode_success": bool(row["episode_success"]), "frame_coverage": row["frame_coverage"],
                     "position_mean_mm": 1000 * row["position_error_m"]["mean"],
                     "position_p95_mm": 1000 * row["position_error_m"]["p95"],
                     "orientation_mean_deg": row["orientation_error_deg"]["mean"],
                     "orientation_p95_deg": row["orientation_error_deg"]["p95"],
                     "base_x_m": base[0], "base_y_m": base[1], "base_z_m": base[2],
                     "tilt_deg": row.get("tilt_deg", 0), "yaw_deg": row.get("yaw_deg", 0),
                     "roll_deg": row.get("roll_deg", 0),
                     **{reason: affected.get(reason, 0) for reason in REASONS},
                     "planner_failure_reason": row.get("planner_failure_reason", ""),
                     "actual_planner_type": row.get("actual_planner_type", ""),
                     "timeline_used": row.get("timeline_used", False)})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summaries(rows: list[dict]) -> list[dict]:
    result = []
    for robot in ROBOTS:
        group = [r for r in rows if r["robot"] == robot]
        result.append({"robot": robot, "success_rate": np.mean([r["episode_success"] for r in group]),
                       "coverage": np.mean([r["frame_coverage"] for r in group]),
                       "position_mm": np.mean([r["position_mean_mm"] for r in group]),
                       "orientation_deg": np.mean([r["orientation_mean_deg"] for r in group])})
    return sorted(result, key=lambda r: (r["success_rate"], r["coverage"]), reverse=True)


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def make_figures(old: list[dict], new: list[dict]) -> list[Path]:
    summary = summaries(new); order = [r["robot"] for r in summary]; y = np.arange(len(order))
    old_s = {r["robot"]: r for r in summaries(old)}; new_s = {r["robot"]: r for r in summary}
    fig, ax = plt.subplots(figsize=(8.2, 4.5)); w = .22
    ax.barh(y+w, [100*old_s[r]["success_rate"] for r in order], w, label="Legacy episode success", color="#A7A7A7")
    ax.barh(y, [100*new_s[r]["success_rate"] for r in order], w, label="Fixed episode success", color="#2474A6")
    ax.barh(y-w, [100*new_s[r]["coverage"] for r in order], w, label="Fixed frame coverage", color="#73C6B6")
    ax.set(yticks=y, yticklabels=order, xlim=(0, 105), xlabel="Rate (%)"); ax.invert_yaxis(); ax.legend(); ax.grid(axis="x", alpha=.2)
    f1 = save(fig, "01_fixed_vs_legacy_ranking")

    matrix = np.array([[next(r for r in new if r["robot"] == robot and r["task"] == task)["frame_coverage"] for robot in order] for task in TASKS])
    passed = np.array([[next(r for r in new if r["robot"] == robot and r["task"] == task)["episode_success"] for robot in order] for task in TASKS])
    fig, ax = plt.subplots(figsize=(9.2, 2.7)); im = ax.imshow(100*matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(10): ax.text(j, i, f"{100*matrix[i,j]:.1f}%\n" + ("PASS" if passed[i,j] else "FAIL"), ha="center", va="center", fontsize=6, color="white" if matrix[i,j]>.6 else "black")
    ax.set(xticks=range(10), xticklabels=order, yticks=range(2), yticklabels=TASKS); plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(im, ax=ax, label="Frame coverage (%)", fraction=.025); f2 = save(fig, "02_fixed_task_robot_coverage")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2)); x=np.arange(10)
    for task, marker in zip(TASKS, ("o", "s")):
        g=[next(r for r in new if r["robot"]==robot and r["task"]==task) for robot in order]
        axes[0].scatter(x, [r["position_mean_mm"] for r in g], label=task, marker=marker)
        axes[1].scatter(x, [r["orientation_mean_deg"] for r in g], label=task, marker=marker)
    for ax,label in zip(axes,("Mean position error (mm)","Mean orientation error (deg)")):
        ax.set(xticks=x, xticklabels=order, ylabel=label); ax.tick_params(axis="x", rotation=45); ax.grid(axis="y",alpha=.2)
    axes[0].legend(); f3=save(fig,"03_fixed_error_distribution")

    fm=np.array([[100*sum(r[k] for r in new if r["robot"]==robot)/sum(r["frames"] for r in new if r["robot"]==robot) for k in REASONS] for robot in order])
    fig,ax=plt.subplots(figsize=(8.2,4.8)); im=ax.imshow(fm,cmap="magma",aspect="auto")
    for i in range(10):
        for j in range(6): ax.text(j,i,f"{fm[i,j]:.1f}",ha="center",va="center",fontsize=6,color="white" if fm[i,j]>.5*max(1,fm.max()) else "black")
    ax.set(yticks=range(10),yticklabels=order,xticks=range(6),xticklabels=REASONS); plt.setp(ax.get_xticklabels(),rotation=35,ha="right"); fig.colorbar(im,ax=ax,label="Affected frames (%)")
    f4=save(fig,"04_fixed_failure_reason_heatmap")

    keys=("base_x_m","base_y_m","base_z_m","tilt_deg","yaw_deg","roll_deg"); labels=("X (m)","Y (m)","Z (m)","Tilt (deg)","Yaw (deg)","Roll (deg)")
    fig,axes=plt.subplots(2,3,figsize=(10,6.2),sharex=True)
    for ax,key,label in zip(axes.flat,keys,labels):
        for task,off in zip(TASKS,(-.12,.12)):
            g=[next(r for r in new if r["robot"]==robot and r["task"]==task) for robot in order]; ax.scatter(x+off,[r[key] for r in g],label=task,s=24)
        ax.set_ylabel(label); ax.set_xticks(x,order,rotation=48,ha="right",fontsize=6); ax.grid(axis="y",alpha=.2)
    axes[0,0].legend(); f5=save(fig,"05_fixed_optimal_mounts")

    metrics=("success_rate","coverage","position_mm","orientation_deg"); titles=("Episode success (pp)","Coverage (pp)","Position error (mm)","Orientation error (deg)")
    fig,axes=plt.subplots(2,2,figsize=(10,6.2))
    for ax,key,title in zip(axes.flat,metrics,titles):
        scale=100 if key in ("success_rate","coverage") else 1; delta=[scale*(new_s[r][key]-old_s[r][key]) for r in order]
        ax.bar(np.arange(10),delta,color=["#2E86C1" if d>=0 else "#C0392B" for d in delta]); ax.axhline(0,color="black",lw=.7); ax.set_ylabel(title); ax.set_xticks(range(10),order,rotation=45,ha="right",fontsize=6)
    f6=save(fig,"06_fixed_minus_legacy_deltas")
    return [f1,f2,f3,f4,f5,f6]


def build_pdf(old: list[dict], new: list[dict], figs: list[Path]) -> Path:
    font="Helvetica"; font_path=Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists(): pdfmetrics.registerFont(TTFont("MicrosoftYaHei",str(font_path))); font="MicrosoftYaHei"
    styles=getSampleStyleSheet(); body=ParagraphStyle("body-cn",parent=styles["BodyText"],fontName=font,leading=15); h=ParagraphStyle("h-cn",parent=styles["Heading1"],fontName=font)
    out=NEW/"ten_arm_fixed_vs_legacy_report.pdf"; doc=SimpleDocTemplate(str(out),pagesize=landscape(A4),leftMargin=14*mm,rightMargin=14*mm,topMargin=12*mm,bottomMargin=12*mm)
    os={r["robot"]:r for r in summaries(old)}; ns=summaries(new)
    data=[["排名","机械臂","修复后成功率","修复前成功率","变化","修复后帧覆盖率","位置误差","姿态误差"]]
    for i,r in enumerate(ns,1): data.append([i,r["robot"],f'{r["success_rate"]:.0%}',f'{os[r["robot"]]["success_rate"]:.0%}',f'{100*(r["success_rate"]-os[r["robot"]]["success_rate"]):+.0f} pp',f'{r["coverage"]:.1%}',f'{r["position_mm"]:.2f} mm',f'{r["orientation_deg"]:.2f} deg'])
    table=Table(data,repeatRows=1); table.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),font),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#D6EAF8")),("GRID",(0,0),(-1,-1),.35,colors.grey),("ALIGN",(0,0),(-1,-1),"CENTER"),("FONTSIZE",(0,0),(-1,-1),7.5)]))
    success_new=sum(r["episode_success"] for r in new); success_old=sum(r["episode_success"] for r in old)
    story=[Paragraph("10种机械臂双任务轨迹跟随：修复版与旧方案对比报告",h),Spacer(1,3*mm),Paragraph("实验口径：完整 episode 全程满足位置、姿态、关节连续性和碰撞约束才计为成功；帧覆盖率仅作为次级指标。新结果使用 4096 GPU 初筛、真实 URDF 严格复筛、真实时间轴与滚动多分支 IK。旧结果取自 twelve_arm_two_single_tasks 中相同 10 臂×2 任务的独立视频 JSON 与 results_summary.csv 冻结快照，仅用于说明程序修复带来的变化，不作为最终机械臂能力结论。",body),Spacer(1,3*mm),table,PageBreak()]
    captions=["结果可视化：完整 episode 成功率排名及修复前后对比","结果可视化：任务×机械臂帧覆盖率热图","结果可视化：位置与姿态误差分布","结果可视化：失败原因严重程度热图","结果可视化：最优安装参数分布","结果可视化：修复后减修复前的逐臂指标变化"]
    for fig,cap in zip(figs,captions): story += [Paragraph(cap,h),Image(str(fig),width=245*mm,height=130*mm),PageBreak()]
    changes="完成内容：统一权威 Dense 复筛为滚动多分支 IK；恢复真实时间戳；加入状态与插值边碰撞检查；统一真实 URDF/TCP 渲染；记录真实规划器、时间轴和根因诊断；失败 episode 也生成视频。"
    conclusions=f"结论：修复版在 20 个机械臂×任务 episode 中成功 {success_new} 个，修复前为 {success_old} 个。应以修复后的完整 episode 成功率、覆盖率及失败原因共同评价机械臂；旧方案因求解器路径、时间轴和失败归因不一致，其排名不具备可靠参考价值。两任务样本量仍有限，关于臂展、轴数和构型的普适 insight 需在全部纯单手任务上复验。"
    story += [Paragraph("完成了什么",h),Paragraph(changes,body),Spacer(1,5*mm),Paragraph("结论",h),Paragraph(conclusions,body)]
    doc.build(story); return out


def main() -> None:
    videos=list((NEW/"videos").rglob("*.mp4"))
    if len(videos)!=20 or any(p.stat().st_size==0 for p in videos): raise RuntimeError(f"videos incomplete: {len(videos)}/20")
    old=load_legacy(); new=load_fixed(); FIG.mkdir(parents=True,exist_ok=True)
    write_csv(NEW/"fixed_results.csv",new)
    delta=[]
    for n in new:
        o=next(r for r in old if r["robot"]==n["robot"] and r["task"]==n["task"])
        delta.append({"robot":n["robot"],"task":n["task"],"episode_success_before":o["episode_success"],"episode_success_after":n["episode_success"],"coverage_delta":n["frame_coverage"]-o["frame_coverage"],"position_mm_delta":n["position_mean_mm"]-o["position_mean_mm"],"orientation_deg_delta":n["orientation_mean_deg"]-o["orientation_mean_deg"]})
    write_csv(NEW/"fixed_vs_legacy.csv",delta); figs=make_figures(old,new); pdf=build_pdf(old,new,figs)
    (NEW/"report_completion.json").write_text(json.dumps({"status":"complete","videos":20,"figures":len(figs),"pdf":str(pdf),"fixed_successes":sum(r["episode_success"] for r in new),"legacy_successes":sum(r["episode_success"] for r in old)},indent=2,ensure_ascii=False),encoding="utf-8")
    print(pdf)


if __name__ == "__main__": main()
