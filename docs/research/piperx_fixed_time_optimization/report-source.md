# PiperX 双手多任务 Fixed-time 项目优化审计

## 结论摘要

本轮审计没有把“低覆盖”简单归因于安装位姿。对 27 条双手轨迹、4 种安装方式、108 个正式分片重新核算后，最重要的结论是：现有结果在碰撞安全上是自洽的，但“源轨迹严格跟随”“条件化目标跟随”和“动力学可执行”必须分开。

按 409,760 个双手有效帧加权，现有正式结果对条件化目标的严格双手覆盖率为 11.8811%，对固定工具映射后、未平滑且未做时变翻腕的源目标仅为 1.8313%。所有 108 个分片的条件化目标相对源目标都超过 1 mm / 0.5° 的严格容差，因此旧覆盖率不能解释为对源记录本身的严格完全跟随。

安全硬约束仍全部通过：正式分片的帧碰撞、扫掠边碰撞和拓扑违规均为 0。动力学方面，只有 35/108 个分片满足研究中使用的 1 rad/s、4 rad/s² 限制；即使改用 Piper SDK V2 可配置的 3 rad/s、5 rad/s² 上限，仍只有相同的 35/108 个分片通过，而且这些分片的严格覆盖均为 0。源 TCP 的测得峰值达到 15.054 m/s 和 45.176 rad/s，实际采样间隔中位数约 12.37–12.40 ms。这表明“原始空间轨迹 + 原始时间戳 + 1 mm / 0.5° + PiperX 动力学 + 禁止重定时”在大量窗口中相互冲突。

## 本轮已实施的项目改进

1. 新增独立于 IK 求解器的证据审计模块，正确处理四元数符号等价，按原始时间间隔计算任务空间速度，并对未条件化源目标重新计算双手同时通过率。
2. 新增 108 分片审计命令，输出逐分片 CSV 与总体 JSON；它分别报告源目标覆盖、条件化目标覆盖、目标偏移、采样间隔、TCP 速度、关节动力学和三类安全计数。
3. 把“时变翻腕”和“平滑条件化”从固定工具坐标映射中分离，避免把优化后的目标当作原始目标。
4. 更新 README 的指标口径和复现入口，使条件化结果不能再被误读为源轨迹严格完全跟随。

## 根因分解

### 1. 指标口径

现有求解器先对目标执行 9 帧 Savitzky–Golay SE(3) 条件化，允许最多 5 mm 平移和 1° 姿态变化，然后再按 1 mm / 0.5° 统计通过。该条件化并不改变时间，但改变空间目标；因此它可以作为求解器目标，却不能代替原始源目标的验证基准。Fold_Box 的右臂还存在最大 12.5° 的时变翻腕适配，必须单独披露。

### 2. 固定时间动力学

TOPP-RA 将路径参数化明确建模为满足速度、加速度与力矩约束的独立问题；当禁止重定时后，这个自由度被移除，算法应报告固定时刻下的不可行窗口，而不是把快速运动隐藏成 IK 失败或在渲染中执行超限跳变。[Pham & Pham, 2018](https://doi.org/10.1109/TRO.2018.2819195)

### 3. 逐帧局部 IK 与分支连续性

RelaxedIK 指出逐帧 point-wise IK 即使每帧误差很小，也可能产生不连续、自碰撞和奇异姿态；其目标函数同时考虑位置、姿态、关节速度、加速度、jerk、自碰撞与可操作度。这与本项目中 warm-start 后仍出现大速度/大加速度的现象一致。[Rakita et al., 2018](https://doi.org/10.15607/RSS.2018.XIV.043)

TRAC-IK 通过并行使用改进 Jacobian 方法与 SQP，提高了有界关节空间中的求解鲁棒性；论文也支持为各笛卡尔维度设置容差，而不是使用未经归一化的单一残差尺度。[Beeson & Ames, 2015](https://doi.org/10.1109/HUMANOIDS.2015.7363472)

SDLS 针对每个奇异方向选择性调整阻尼，比固定全局阻尼更适合奇异附近的 IK；Nakamura 的奇异鲁棒逆运动学则给出了更早的理论基础。[Buss & Kim, 2005](https://doi.org/10.1080/2151237X.2005.10129202)；[Nakamura & Hanafusa, 1986](https://doi.org/10.1115/1.3143764)

### 4. 约束层级与连续碰撞

层级二次规划可以把关节界、速度、加速度、碰撞与拓扑作为高优先级不等式，把姿态和姿势优化放在更低层级，避免软惩罚用精度交换安全。[Escande et al., 2014](https://doi.org/10.1177/0278364914521306)

TrajOpt 使用序列凸优化和连续时间碰撞检查，这支持把整段轨迹和扫掠碰撞放入同一优化问题，而不是只在每帧 IK 完成后做检查。[Schulman et al., 2014](https://doi.org/10.1177/0278364914528132)

### 5. 官方硬件限制

AgileX 官方 Piper SDK V2 接口允许设置 J1–J6 的最大关节速度 0–3.0 rad/s、最大关节加速度 0–5.0 rad/s²，并支持查询电机当前限制。正式实验采用 1 rad/s、4 rad/s² 是保守设置；审计表明即使使用官方可配置上限，也没有让任何已有非零覆盖分片同时通过动力学。[AgileX Piper SDK V2](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD)

## 推荐的求解架构

### P0：结果合同与准入检查

- 把“源目标 1 mm / 0.5° 双手同时覆盖”设为主指标；条件化目标覆盖只作为诊断指标。
- 在 mount 搜索前计算真实采样率、TCP 段速度和已知跳变；禁止把高动态行静默排除出分母。
- 发布门槛必须同时要求原始目标覆盖、关节动力学和零碰撞/零边碰撞/零拓扑违规。
- 渲染仍可展示安全 HOLD，但不得把 HOLD 显示为跟随成功。

### P1：Fixed-time 联合轨迹求解

- 用时间索引的双臂分支图或窗口化优化替代独立逐帧 DLS；状态包含上一帧速度，使速度与加速度在候选选择时就是硬约束。
- 使用层级 QP：第一层为关节界、速度、加速度、碰撞、拓扑；第二层为 1 mm / 0.5° 源目标 tube；第三层最小化速度、加速度、jerk 和奇异风险。
- 用容差归一化残差和 SDLS/奇异鲁棒阻尼替换固定 0.3 阻尼；将位置和姿态容差的量纲显式纳入尺度。
- 保留 MuJoCo 连续边碰撞复核；窗口优化的每条转移仍需经过正式扫掠检查。

### P2：mount 与数据层优化

- mount 排名改用“源目标严格覆盖 + 固定时间动力学可行 + 安全硬门”，不再以条件化覆盖为主目标。
- 对采集链路做时间戳和位姿异常诊断；允许的空间滤波必须位于误差预算内，并始终回到未滤波目标验证。
- 如果业务目标仍是 100% 完全跟随，必须至少改变一个不可兼得条件：重新采集/修复源空间轨迹、允许时间参数化、提高硬件能力，或放宽 1 mm / 0.5°。仅增加 IK 重启次数无法解决动力学矛盾。

## 可复现证据

- `reports/piperx_multitask_fixed_time_mount_study/literature_audit/raw_source_shard_audit.csv`
- `reports/piperx_multitask_fixed_time_mount_study/literature_audit/raw_source_audit_summary.json`
- `python -m scripts.audit_piperx_fixed_time_evidence`
- `python -m pytest tests/factory_bimanual/test_fixed_time_evidence_audit.py -q`

## 参考文献

1. Beeson, P. & Ames, B. TRAC-IK: An Open-Source Library for Improved Solving of Generic Inverse Kinematics. IEEE-RAS Humanoids (2015). DOI: 10.1109/HUMANOIDS.2015.7363472.
2. Rakita, D., Mutlu, B. & Gleicher, M. RelaxedIK: Real-time Synthesis of Accurate and Feasible Robot Arm Motion. RSS (2018). DOI: 10.15607/RSS.2018.XIV.043.
3. Escande, A., Mansard, N. & Wieber, P.-B. Hierarchical quadratic programming: Fast online humanoid-robot motion generation. IJRR 33, 1006–1028 (2014). DOI: 10.1177/0278364914521306.
4. Schulman, J. et al. Motion planning with sequential convex optimization and convex collision checking. IJRR 33, 1251–1270 (2014). DOI: 10.1177/0278364914528132.
5. Pham, H. & Pham, Q.-C. A New Approach to Time-Optimal Path Parameterization Based on Reachability Analysis. IEEE Transactions on Robotics 34, 645–659 (2018). DOI: 10.1109/TRO.2018.2819195.
6. Buss, S. R. & Kim, J.-S. Selectively Damped Least Squares for Inverse Kinematics. Journal of Graphics Tools 10, 37–49 (2005). DOI: 10.1080/2151237X.2005.10129202.
7. Nakamura, Y. & Hanafusa, H. Inverse Kinematic Solutions With Singularity Robustness for Robot Manipulator Control. Journal of Dynamic Systems, Measurement, and Control 108, 163–171 (1986). DOI: 10.1115/1.3143764.
8. AgileX Robotics. Piper SDK V2 Interface Documentation. Official repository, accessed 2026-09-07.
