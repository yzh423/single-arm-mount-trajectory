# 报告与可视化索引

本目录发布可复核的正式产物。当前推荐入口是 PiperX 双手多任务 Fixed-time 四构型研究；旧的 complete-follow 和 recommended v3.1 仅作历史对照，不能替代当前的无重定时、安全硬门实验。

## 当前推荐：27 条双手轨迹 × 4 种安装构型

- [多任务中文 PDF 报告](piperx_multitask_fixed_time_mount_study/PiperX多任务Fixed-Time四构型对比报告.pdf)
- [108 分片 bundle manifest](piperx_multitask_fixed_time_mount_study/bundle_manifest.json)
- [PDF/图表/视频/实现哈希 release manifest](piperx_multitask_fixed_time_mount_study/release_manifest.json)
- [汇总 CSV](piperx_multitask_fixed_time_mount_study/aggregate.csv)
- [四构型对比图](piperx_multitask_fixed_time_mount_study/figures/)
- [27 个同步 MuJoCo 对比视频](piperx_multitask_fixed_time_mount_study/videos/comparisons/)

正式协议固定为源时间戳、禁止 retiming、1 mm / 0.5°，并要求 `collision_frames`、`edge_collision_frames`、`topology_invalid_frames` 全部为零。位姿覆盖率与动力学通过率分别报告；“覆盖率胜者”不等同于可部署结果。

## 保留的单臂报告与视频

- [安装与 IK 保真度报告](single_arm/mount_ik_fidelity_pilot/pdf/mount_ik_fidelity_pilot_report.pdf)
- [OpenArm 可视化](single_arm/mount_ik_fidelity_pilot/videos/openarm.mp4)
- [xArm6 可视化](single_arm/mount_ik_fidelity_pilot/videos/xarm6.mp4)
- [十机械臂三回合报告](single_arm/ten_arm_pick_right_left_three_episodes/report.pdf)
- [Fixed-time 与历史方案对比报告](single_arm/ten_arm_two_single_tasks_handbook_fixed_4096/ten_arm_fixed_vs_legacy_report.pdf)

## 历史双臂结果

- [`piperx_two_task_complete_follow/`](piperx_two_task_complete_follow/)：Fold_Box 与 Seal_Bag 的早期 complete-follow 对照，包含 retiming 或残余碰撞，仅用于说明方案演化。
- [`piperx_complete_follow/`](piperx_complete_follow/)：早期 Fold_Box 单任务完整跟随结果。
- [`piperx_recommended_v31/`](piperx_recommended_v31/)：recommended v3.1 历史基线。

用户指定不上传的四份输入/阶段性 PDF 不属于发布产物，因此本索引不链接它们。参考资料只用于设计方案，不作为仓库内可分发文件。
