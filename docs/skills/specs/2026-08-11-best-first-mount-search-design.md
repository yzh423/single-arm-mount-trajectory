# Best-first 分层安装位姿搜索设计

日期：2026-08-11  
状态：待用户书面审阅  
适用入口：`scripts/search_strict_urdf_mount.py`、`scripts/run_twelve_arm_two_single_tasks.py`

## 1. 背景与目标

现行严格搜索对每个 robot-task 固定评估 218 个安装位姿，并在 medium、strict、micro、final 阶段对多个候选重复运行高成本 IK。候选从低保真度晋级到高保真度时，已求解帧、关节路径和父候选结果没有得到系统复用。

本设计将固定漏斗改为“廉价排序 + best-first 前沿扩展 + 增量 IK 复用”：

1. Global/Local 只需建立可靠排序，不承担完整轨迹证明。
2. Local 总预算固定为 64 个候选，集中在多个有前景且相互分离的区域。
3. Dense refinement 只提升可能超过当前最优解的前沿节点。
4. 找到首个完整成功解后继续扩展有限候选，以获得接近全局最优的经验证据。
5. 只有最终 1-2 个候选使用正式高 restart 预算完整复核。

本方案是 B* 风格的启发式 best-first 搜索。由于 IK 排序函数不是严格可采纳上界，输出不得宣称数学意义上的全局最优，只能报告“在给定预算和覆盖范围内未发现更优候选”。

## 2. 非目标

- 不改变任务成功阈值、碰撞规则、关节限位或最终正式复核协议。
- 不用增加 workers 的方式掩盖单个 shard 的 worker-hours。
- 不要求第一版实现可微 IK、GPU MuJoCo 或新的机器人模型格式。
- 不删除现有层级搜索；保留 legacy 模式用于结果回归和配对消融。

## 3. 搜索预算

| 阶段 | 候选预算 | 帧预算 | 首帧 restart | 目的 |
|---|---:|---:|---:|---|
| Geometry gate | Global 128 | 0 | 0 | 剔除桌面安装、边界和必要条件不满足项 |
| Global rank | gate 后候选 | 16-24 个代表帧 | 最多 5 | 建立全局低成本排序 |
| Region retain | 8 个区域 | 0 | 0 | 保留空间多样性，避免单峰塌缩 |
| Local rank | 8 × 8 = 64 | 与 Global 相同 | 继承失败后最多 2 | 局部排序与前沿初始化 |
| Medium | 前沿累计最多 8 个 | 64 帧 | 继承失败后最多 3 | 提升排序可信度 |
| Dense | 首解前累计最多 3 个 | 完整轨迹 | 继承失败后最多 5 | 寻找首个完整成功解 |
| Post-success | 最多再扩展 8 个前沿节点 | 按需晋级 | 同对应层级 | 验证 incumbent 附近和其他区域 |
| Final | 1-2 个 | 完整轨迹 | 正式 16 | 最终严格复核 |

预算应集中定义在一个不可变配置对象中，CLI 可覆盖，但默认值必须进入结果 JSON 和 fingerprint。

## 4. 候选与前沿模型

每个候选节点包含：

```text
candidate_id
mount = [base_x, base_y, base_z, pitch, yaw, roll]
parent_id
region_id
fidelity = gate | rank | medium | dense | final
evaluated_frame_indices
q_by_frame
success_by_frame
position_error_by_frame
orientation_error_by_frame
collision_by_frame
restart_count_by_frame
score
optimistic_score
elapsed_s
```

前沿使用以下稳定排序键，从高到低比较：

1. `optimistic_score`
2. 当前已验证的无碰撞帧覆盖率
3. 最长连续失败区间的相反数
4. 位置误差与姿态误差
5. 空间新颖度
6. `candidate_id`，用于确定性 tie-break

候选状态只能沿 `gate → rank → medium → dense → final` 单向晋级。晋级不得清空已有帧结果。

## 5. 排序分数与乐观界

### 5.1 已验证分数

沿用现有 `follow_rank` 的目标优先级：

1. 完整 episode 成功
2. 帧覆盖率
3. 最长失败区间
4. 碰撞帧数
5. 位置误差
6. 姿态误差

低保真度阶段只在确定性代表帧上计算同构指标，不能把抽样覆盖率当作完整成功率。

### 5.2 乐观分数

未验证帧暂按成功计入覆盖率上界；已验证失败、碰撞或关节不连续帧仍按失败处理。误差项使用已验证帧统计量的有利界，但不得优于零误差。

`optimistic_score` 只用于决定谁值得继续计算，不作为最终报告指标。它不是严格 admissible bound，因此停止报告必须包含预算限定。

### 5.3 区域多样性

Global 排序后，从 Top-K 候选中按归一化安装空间距离选择 8 个区域中心。距离覆盖位置和旋转六个维度，并使用现有 bounds 归一化。每个区域生成 8 个 Local 候选，总计 64 个；第一项保留中心本身，其余 7 项使用确定性 scrambled Sobol 局部扰动。

## 6. 代表帧策略

廉价排序使用 16-24 个确定性代表帧，而不是直接均匀抽取 64 帧：

- 必含首帧和末帧。
- 按轨迹弧长分位数选取主体帧，避免静止区间占满预算。
- 加入位置曲率、姿态变化率的局部峰值。
- 去重后不足最低数量时再用均匀采样补齐。

同一 robot-task 的代表帧集合对所有候选一致，并写入审计 JSON，以确保排序公平和结果可复现。

## 7. 增量 IK 复用

### 7.1 父子候选复用

Local 子候选优先使用父候选对应帧的 `q` 作为首个 seed。若该帧父解不存在，则使用最近已成功帧或当前候选前一成功帧。只有继承 seed 失败后才支付有限随机 restart。

### 7.2 保真度晋级复用

- `rank → medium`：保留代表帧结果，只计算新增帧。
- `medium → dense`：保留 64 帧结果，为未计算区间从最近成功解双向展开。
- `dense → final`：正式复核单独运行，以保证最终结果不依赖启发式缓存；但可以记录启发式 seed 与正式随机 restart 的差异。

### 7.3 不允许的复用

- 不得直接把父候选误差复制给子候选。
- 不得跨机器人或跨任务复用 `q`。
- mount 变化后必须重新执行 FK、误差和碰撞检查。
- 最终正式复核不得跳过任何完整轨迹帧。

## 8. Best-first 扩展流程

1. 生成 128 个 Global candidates。
2. 运行 geometry gate。
3. 对 gate 通过项运行廉价 rank，选出 8 个相互分离的区域。
4. 每区域生成并评估 8 个 Local candidates。
5. 将 Global elites 和 Local candidates 放入统一优先队列。
6. 弹出 `optimistic_score` 最高节点并提升一个 fidelity 等级。
7. 若提升后仍有潜力，将节点重新放回前沿；否则关闭节点。
8. 在最多 3 个 Dense 晋级预算内寻找首个完整成功解。
9. 找到首解后继续扩展最多 8 个最高优先级节点。
10. 从完整成功节点中选出最优 1-2 个执行 Final 正式复核。

若首解前 3 个 Dense 晋级全部失败，搜索不得直接结束；它应继续 best-first 扩展，但进入明确的 fallback 预算，默认再允许 5 个 Dense 晋级，并在结果中标记 `fallback_used=true`。

## 9. 首解后的停止条件

找到首个成功解后，满足以下任一硬限制即停止扩展：

- 已额外扩展 8 个前沿节点；
- post-success wall-time 达到该 shard 首解前耗时的 25%；
- 前沿为空。

若以下软条件同时满足，可以提前停止：

- 至少验证 3 个不同区域；
- 连续 4 次扩展未改善 incumbent；
- 所有剩余节点的乐观排序均不能超过 incumbent，或归一化差距小于配置容差。

最终 JSON 必须记录实际触发的停止原因、扩展数量、已覆盖区域数、incumbent 改善历史和剩余前沿最优乐观分数。

## 10. 模型生命周期

第一版必须把“搜索算法减少 IK 次数”和“模型编译复用”分开计量。

推荐每个 robot subprocess 预编译一个固定拓扑模型模板。候选之间只更新 root body 安装位姿及固定安装几何，然后创建或重置 `MjData`。如果 MuJoCo 模型中的安装几何不能安全原地更新，应先保留当前每候选编译方式，但仍实现 best-first 和 IK 复用；不得为了编译优化改变碰撞语义。

## 11. 缓存与 fingerprint

缓存分为三层：

1. `search_state`：候选、前沿、代表帧和增量 IK 状态。
2. `strict_solution`：完整轨迹正式求解结果。
3. `render_output`：视频及渲染审计。

三层使用独立 fingerprint。渲染脚本变化不得使 `search_state` 或 `strict_solution` 失效。GPU 粗搜参数不得直接进入严格搜索 fingerprint；严格搜索只依赖实际读取的 incumbent 内容及其相关输入。

中断恢复应从最近一次原子 checkpoint 继续，不重复已完成候选。checkpoint 只在节点晋级完成后写入，临时文件写完并校验后再替换正式文件。

## 12. 可观测性

每个 shard 聚合记录：

- 各 fidelity 的候选数、晋级数和淘汰数；
- geometry gate 拒绝数量；
- 新算帧数与复用帧数；
- 父 seed 成功数、随机 restart 分布和失败数；
- IK 调用数与 iteration 总数；
- model compile、gate、IK、selection、checkpoint、final replay 的 monotonic elapsed time；
- 首解时间、post-success 时间、停止原因；
- incumbent 改善历史及区域覆盖数。

只写阶段聚合和少量候选抽样，不同步记录每次 iteration。

## 13. 失败处理

- 代表帧评估异常：候选标记为 closed，并记录异常类型；其他候选继续。
- 前沿无候选：返回明确失败，不使用旧 cache 冒充本轮结果。
- checkpoint 损坏：忽略该 checkpoint，从最后一个可校验版本恢复。
- Final 正式复核失败：尝试下一名完整成功候选；没有候选通过时本轮状态为 fail。
- 数值非有限：候选立即关闭并计入 `numerical_failure_candidates`。

## 14. 测试策略

### 14.1 单元测试

- 64 个 Local 候选严格由 8 个区域各生成 8 个。
- 前沿排序和 tie-break 在固定 seed 下确定。
- fidelity 晋级只计算新增帧。
- 子候选优先使用父 q seed，失败后才触发 restart。
- post-success 硬限制和软停止条件逐项可触发。
- 三层 fingerprint 只受各自相关输入影响。

### 14.2 回归测试

使用 xarm6/cap-left 小预算 fixture 对比 legacy 与 best-first：

- 最终正式成功判定一致；
- best-first 的完整轨迹候选数更少；
- 已求解帧不会在晋级时重复计算；
- 固定 seed 两次运行输出相同。

### 14.3 配对消融

对至少一个容易任务和两个困难 robot-task 运行：

- legacy；
- 仅廉价 restart；
- 廉价 restart + best-first；
- 完整方案（含增量复用）。

比较 Top-K recall、最终 winner、任务状态、worker-seconds、IK iterations 和完整轨迹回放次数。

## 15. 验收标准

第一阶段以不牺牲正式结果为前提：

- 已有 10 个 pass 组合的 Final 正式判定不得退化。
- 24 组合中最终 winner 的安装位姿可以变化，但正式指标不得低于 legacy incumbent；若降低，必须自动保留 legacy incumbent 参与 Final。
- 每个 shard 的 Dense 完整轨迹候选默认不超过 8，常规路径目标不超过 3。
- Local 候选固定为 64。
- 晋级过程中代表帧重复 IK 次数为 0。
- 相比 legacy，完整轨迹 frame opportunities 至少降低 70%。
- 在选定的 3 个性能样本上，累计 worker time 至少降低 50%。
- 所有性能结论必须来自新增聚合计时字段，不使用文件时间戳代替阶段计时。

## 16. 交付顺序

1. 增加聚合计时和 legacy 基准反馈信号。
2. 抽象候选状态、前沿和 fidelity 晋级接口。
3. 实现代表帧、64-Local 和 best-first 扩展。
4. 实现父子 seed 与跨 fidelity 帧复用。
5. 拆分 fingerprint 与 checkpoint。
6. 在不改变碰撞语义的前提下评估模型编译复用。
7. 完成配对消融后再调整默认预算。
