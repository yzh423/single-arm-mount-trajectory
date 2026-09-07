# 报告与可视化索引

本目录发布可复核的正式产物。当前推荐入口是 PiperX controller-event raw Fixed-time v4。该协议保留成对控制器接收事件的真实间隔，不重定时、不平滑目标、不使用时变翻腕，并将位姿覆盖、安全和动力学分开验收。

## 当前推荐：controller-event raw Fixed-time v4

- [根因诊断与优化 PDF](../output/pdf/PiperX双手Fixed-Time-v4根因诊断与优化报告.pdf)
- [`piperx_controller_event_v4/`](piperx_controller_event_v4/)：Seal_Bag 与 Fold_Box 的完整 baseline、upright_table 几何跟踪证据。
- [`piperx_controller_event_v4_prefix300/`](piperx_controller_event_v4_prefix300/)：两个任务各 300 个控制器事件、四种安装构型的等长比较。
- [四构型 bundle manifest](piperx_controller_event_v4_prefix300/bundle_manifest.json)
- [Seal_Bag 四构型 MuJoCo 视频](piperx_controller_event_v4_prefix300/videos/comparisons/8-11_Seal_Bag_161504_four_mount_fixed_time.mp4)
- [Fold_Box 四构型 MuJoCo 视频](piperx_controller_event_v4_prefix300/videos/comparisons/8-11_Fold_Box_161044_four_mount_fixed_time.mp4)

完整 baseline 的双手严格几何覆盖率为 Seal_Bag 100.00%、Fold_Box 94.83%。当前发布轨迹的 `collision_frames`、`edge_collision_frames`、`topology_invalid_frames` 均为零，但 `dynamics_enforced=false`，关节速度和加速度也超过 Piper SDK V2 的 3 rad/s、5 rad/s² 配置上限，因此不能直接下发实机。

## 保留的单臂报告与视频

- [安装与 IK 保真度报告](single_arm/mount_ik_fidelity_pilot/pdf/mount_ik_fidelity_pilot_report.pdf)
- [OpenArm 可视化](single_arm/mount_ik_fidelity_pilot/videos/openarm.mp4)
- [xArm6 可视化](single_arm/mount_ik_fidelity_pilot/videos/xarm6.mp4)
- [十机械臂三回合报告](single_arm/ten_arm_pick_right_left_three_episodes/report.pdf)
- [Fixed-time 与历史方案对比报告](single_arm/ten_arm_two_single_tasks_handbook_fixed_4096/ten_arm_fixed_vs_legacy_report.pdf)

## 历史双臂结果

- [`piperx_multitask_fixed_time_mount_study/`](piperx_multitask_fixed_time_mount_study/)：早期 27 条轨迹 × 4 构型研究。它使用主机轮询时间、目标条件化，并在 Fold_Box 使用时变翻腕；其覆盖率不能与 v4 原始目标覆盖率直接混用。
- [`piperx_two_task_complete_follow/`](piperx_two_task_complete_follow/)：Fold_Box 与 Seal_Bag 的早期 complete-follow 对照，包含 retiming 或残余碰撞，仅用于说明方案演化。
- [`piperx_complete_follow/`](piperx_complete_follow/)：早期 Fold_Box 单任务完整跟随结果。
- [`piperx_recommended_v31/`](piperx_recommended_v31/)：recommended v3.1 历史基线。

用户指定不上传的四份输入/阶段性 PDF 不属于发布产物，因此本索引不链接它们。参考资料只用于设计方案，不作为仓库内可分发文件。
