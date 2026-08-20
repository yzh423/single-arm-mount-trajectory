# 报告与可视化索引

本目录只发布可复核的代表性产物。大型搜索缓存、烟雾测试和原始批量渲染保留在本地工作区，不进入 GitHub。

## 当前推荐结果：Fold_Box + Seal_Bag 双任务 complete-follow

- [双任务中文 PDF 报告](piperx_two_task_complete_follow/PiperX双任务严格完全跟随实验报告.pdf)
- [机器可读 manifest](piperx_two_task_complete_follow/two_task_manifest.json)
- [汇总 CSV](piperx_two_task_complete_follow/two_task_summary.csv)
- [优化实验日志](piperx_two_task_complete_follow/two_task_experiment_log.json)
- [Fold_Box 真实 MuJoCo 视频](piperx_two_task_complete_follow/fold_box/8-11_Fold_Box_161044_complete_follow.mp4)
- [Seal_Bag 真实 MuJoCo 视频](piperx_two_task_complete_follow/seal_bag/8-11_Seal_Bag_161504_complete_follow.mp4)

Fold_Box 为 1061/1061，Seal_Bag 为 1757/1757，两个任务均在 1 mm / 0.5° 下达到 registered raw 60 Hz 位姿完整覆盖。原时序均只有首帧通过全部动力学门，受限重定时执行时长分别为 74.285 s 和 100.633 s。碰撞审计仍分别有 71 和 17 帧，因此无碰撞严格覆盖为 93.31% 和 99.03%，不能解释为真机安全许可。

## 保留的单任务结果：complete-follow v4.0

- [中文 PDF 报告](piperx_complete_follow/PiperX双臂完全跟随优化与验证报告_v4.0.pdf)
- [真实 MuJoCo 渲染视频](piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.mp4)
- [汇总指标](piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.summary.json)
- [逐帧轨迹证据](piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.trajectory.npz)
- [视频来源旁车](piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.provenance.json)
- [可复现场景](piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.scene.xml)

注册后原始位姿严格结果为 1061/1061，阈值为 1 mm / 0.5°，没有启用目标平滑。原 60 Hz 节拍只有 1/1061 个源位姿同时满足执行时间；计入零速度起止边界的局部无损重定时路径为 74.285 s，并通过 1 rad/s、4 rad/s² 上限。视频在 execution knot 之间线性插值。碰撞审计有 71 个源帧，因此无碰撞严格覆盖为 990/1061（93.31%）。位姿可达、节拍可执行和碰撞安全三项指标不可互相替代。

## 保留的历史基线：recommended v3.1

- [旧版 PDF 报告](piperx_recommended_v31/PiperX双臂IK跟随优化与验证报告_v3.1.pdf)
- [旧版真实渲染视频](piperx_recommended_v31/8-11_Fold_Box_161044_recommended_v31.mp4)
- [旧版汇总指标](piperx_recommended_v31/8-11_Fold_Box_161044_recommended_v31.summary.json)

该基线在同一 Fold_Box 轨迹上的严格同步覆盖为 112/1061（10.56%），用于展示任务级工具映射和独立单臂状态机优化前后的差异。

## 原始参考报告

四份输入 PDF 保留在仓库根目录，方便按原文件名追溯：四臂四安装位姿方案、26 条轨迹综合分析、Seal_Bag 三构型对比和单轨迹优化分析。
