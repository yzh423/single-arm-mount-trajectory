# xArm6 单轨迹优化分析报告设计

日期：2026-08-14

## 目标

将 `seal_bag_dual_xarm6` 与 `fold_box_dual_xarm6` 的最终轨迹跟随结果整理为一份静态中文 PDF。报告保持 6–8 页的短篇幅，但通过多面板图、关键帧和完整指标表详细展示数据，不包含下一轮优化建议。

## 权威输入

### 封袋任务

- 最终视频：`seal_bag_v6_global_retimed_no_flip_front_720p.mp4`
- 最终轨迹：`seal_bag_global_retimed_no_visual_flip.trajectory.npz`
- 汇总指标：`seal_bag_global_retimed_no_visual_flip.summary.json`
- 渲染溯源：`seal_bag_v6_global_retimed_no_flip_front_720p.provenance.json`

### 折盒任务

- 最终视频：`fold_box_full_se3_acceleration_smoothed_front_720p.mp4`
- 最终轨迹：`fold_box_full_se3_acceleration_smoothed_front_720p.trajectory.npz`
- 汇总指标：`fold_box_full_se3_acceleration_smoothed_front_720p.summary.json`
- 渲染溯源：`fold_box_full_se3_acceleration_smoothed_front_720p.provenance.json`

## 页面结构

1. **总览**：报告范围、最终版本、核心指标对比，包括左右臂严格覆盖率、同步覆盖率、逐帧成功率、失败帧、碰撞帧、源轨迹时长、执行时长、重定时增量、位置误差和姿态误差。
2. **最终渲染摘要**：每个任务使用相同时间分位点抽取关键帧，标注时间；同时列出原视频相对路径、帧率和时长。静态 PDF 不嵌入视频，只提供关键帧与源文件索引。
3. **封袋轨迹与误差**：左右臂目标/实际 XYZ 三维轨迹，轨迹颜色编码 Yaw 或时间；位置误差、姿态误差和同步成功状态折线图。
4. **封袋失败诊断**：放大右臂 12.11–12.62 s 失败窗口，分别展示 X/Y/Z 目标—实际差值、位置误差、姿态误差、成功状态、关节极限裕量与奇异性裕量；标出峰值及失败起止点。
5. **折盒轨迹与误差**：与封袋使用相同视觉尺度和指标定义，展示左右臂目标/实际 XYZ 三维轨迹、Yaw、位置/姿态误差和成功状态。
6. **折盒重定时与裕量**：执行时间增量、被重定时帧、执行步长、关节加速度约束、关节极限裕量、奇异性裕量及候选解数量。
7. **完整指标表与客观结论**：保留所有关键汇总字段、单位、数据来源和简短事实性结论；不提出后续优化方案。

若图表在 A4 横向页面中出现标签拥挤，可将第 7 页拆为两页，但总页数不得超过 8 页。

## 指标定义

- **严格覆盖率**：轨迹文件中对应 `*_success` 布尔序列的均值，并与 summary JSON 交叉核对。
- **同步覆盖率/成功率**：`synchronous_success` 的均值；单次确定性轨迹不虚构跨试验成功率。
- **位置误差**：目标 TCP 与实际 TCP 的欧氏距离，单位 mm；同时报告均值、P95、最大值。
- **姿态误差**：四元数角距离，单位 deg；同时报告均值、P95、最大值。
- **失败点轨迹差值**：失败窗口内 `target_position - actual_position` 的 X/Y/Z 分量，单位 mm。
- **Yaw**：从目标和实际四元数按统一旋转约定计算，展开相位后显示，单位 deg。
- **裕量**：关节极限裕量使用 rad，奇异性裕量使用原始无量纲值。
- **重定时**：比较源时间与执行时间，并展示 `execution_dt_s`、`time_retimed` 及汇总中的加速度约束字段。

## 可视化规范

- PDF 使用 A4 横向、中文字体、白底和适合打印的高对比配色。
- 左臂与右臂、目标与实际、成功与失败采用全报告一致的颜色和线型，并辅以图例/线型，避免只靠颜色表达。
- 三维轨迹使用固定视角，并补充 XY/XZ 投影视图或坐标范围，以防静态视角遮挡关键信息。
- 误差图共享时间轴；失败区间使用浅色背景带，阈值存在时绘制阈值线。
- 数值保留与测量精度匹配的小数位，不显示无意义的长小数。
- 叙述控制在图注和结论要点内，避免重复解释图中已明确的信息。

## 数据流与产物

1. 读取两个 summary JSON、trajectory NPZ 和 provenance JSON。
2. 计算 P95、成功率、XYZ 差值、Yaw、失败窗口及版本间相对变化。
3. 从最终 MP4 按时间分位点抽取关键帧。
4. 生成矢量图优先的多面板页面，并导出 PDF。
5. 保存一份机器可读的派生指标 JSON，便于追溯和复核。

产物放置于：

`single-arm-mount/reports/factory_bimanual/xarm6_single_trajectory_optimization_report/`

至少包含：

- `xarm6_single_trajectory_optimization_report.pdf`
- `derived_metrics.json`
- `figures/`
- 可重复执行的报告生成脚本

## 验证

- 对照 summary JSON 检查覆盖率、均值、最大值、时长、失败帧与碰撞帧。
- 从 NPZ 独立重算均值、P95、最大值和成功率，发现不一致时在报告中明确标注口径差异。
- 验证失败窗口起止帧和时间与轨迹数组一致。
- 检查视频关键帧来自最终视频且时间标签正确。
- 编译 PDF 后逐页渲染为图片，检查缺字、溢出、图例遮挡、坐标裁切和分辨率。
- 确认最终页数为 6–8 页且不包含优化建议。

## 范围边界

- 不重新运行 IK、碰撞检测或轨迹优化。
- 不修改原始轨迹、视频、summary 或 provenance 文件。
- 不把单次轨迹的逐帧成功率描述成多次实验的统计成功率。
- 不提供下一轮优化建议。
