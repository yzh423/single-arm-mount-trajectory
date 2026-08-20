# PiperX 双臂 recommended_v3_1 优化设计

## 目标

在保留现有 `single-arm-mount` 资产的前提下，为 PiperX 双臂新增一条独立、可审计的推荐执行链，逐项落实《双臂IK跟随方案四臂四种安装位姿报告.pdf》的 PiperX 方案。硬验收门限固定为位置误差不大于 1 mm、姿态误差不大于 0.5°，任何入口不得降级或放宽。

## 方案边界

- 复用官方 PiperX URDF、MuJoCo 场景构建、TCP 标定、候选 IK 和碰撞检查组件。
- 不删除旧求解器、旧报告或旧视频；新链路使用 `recommended_v3_1` 名称和独立输出目录，避免历史结果被静默覆盖。
- 不把 PDF 的四臂汇总数字冒充本项目重新实测结果。最终报告明确区分“PDF 参考结论”和“本次 PiperX MuJoCo 验证”。
- GitHub 只上传可复现源码、必要模型/示例数据和精选产物；本地原始大数据、缓存和批量历史输出通过清单保留但不进入 Git。

## 安装方案

### 任务族

任务族键为 `(日期, 任务)`，例如 `8-11/Fold_Box` 与 `8-12/Fold_Box` 必须分开。日期从 `data/factory/<date>/<task>/...csv` 的相对路径获取，禁止只按任务目录名合并。

### 四种构型

- `baseline_frozen`：历史冻结台座，仅作为对照，不作为默认部署。
- `upright_table`：底部桌面正立，PiperX 默认首选。
- `horizontal_forward`：人形立杆横装，作为第二选择。
- `inverted`：顶部倒挂，只在任务族推荐表明确指定时使用。

### PiperX 共享底座

配置文件固化 PDF 第 7 页列出的 12 个 PiperX 任务族、构型、左右 `p_base` 和来源 take。原始 `p_base` 与轨迹同坐标系；进入 MuJoCo 时统一应用任务刚体注册：

`p_base_world = R_registration @ p_base_source + t_registration`

底座姿态由构型生成，左右 base Z 取变换后的各自 Z；若数值误差不超过 1e-9 m 则归一为共享高度。执行前仍经过场景编译、状态碰撞和扫掠边碰撞检查。

### 后续搜索漏斗

新配置同时表达 PDF 的四级漏斗：0.1 m XY 粗网格和构型高度规则、可达性代理 top-150、首帧 40 次重启锚定、top-12 每 60 帧 warm-start 探测、top-3 全程 v3.1 评估。已有搜索器可以逐步迁移到该协议，但本次真实视频直接使用 PDF 推荐共享底座，避免重复搜索改变推荐结论。

## IK 与跟随协议

### 严格在线层

1. 目标轨迹以 60 Hz 语义处理，位置和四元数只做有界平滑，不能移动求解结果。
2. 第 0 帧使用 40 个确定性重启锚定，选择严格误差盒内且关节/腕部风险最低的解。
3. 后续帧从上一条已下发关节指令 warm-start DLS。
4. ACCEPT 必须同时满足：位置误差 ≤0.001 m、姿态误差 ≤`deg2rad(0.5)`、每个关节相对上一指令的最短角差 ≤0.30 rad、状态和扫掠边无碰撞。
5. 任一条件失败则 HOLD；本帧记录失败，但整条轨迹继续，下一帧仍从最后下发位形求解。

### 翻腕与分支处理

- FOLLOW 阶段禁止 J4/J5 分支签名无过渡变化；周期关节用最短角差。
- 腕部风险使用现有 PiperX J4/J5 硬限位障碍函数，并把腕部距离纳入候选排序。
- 只有 rescue 事件可以切换腕部分支。切换必须走速度/加速度受限的五次 smoothstep，不允许瞬时翻腕。
- 每个事件记录起止分支、最大单关节位移、腕部位移范数、过渡帧数和接缝误差。

### rescue_v3.1

- 触发条件：当前分支无 strict-OK 解，或 lookahead 显示当前分支可跟随长度短于替代分支。
- 防抖：距上次跳变少于驻留帧数时只 HOLD。
- 选支：候选在 `f+p` 恢复点之后按预计连续 strict-OK 长度排序；同分时依次偏好更小执行时间、更低腕部风险、更大关节裕量。
- 执行时间模型固定为 PiperX：最大速度 1 rad/s、最大加速度 4 rad/s²、稳定时间 0.1 s、决策时间 0.015 s。
- 模式 A：未来恢复点可达，目标继续前进；`[f, f+p)` 计 dropped，过渡终点直接等于 `f+p` 的跟踪流形解，接缝目标为 0 rad。
- 模式 B：未来恢复点不可达但当前目标存在安全替代分支，暂停目标并插入 p 个执行帧；源帧不丢失，单独累计 cycle delay。
- 若无安全替代分支，继续 HOLD，绝不放宽 1 mm/0.5° 门限。

## 软件结构

- `configs/piperx_recommended_v31.json`：严格阈值、执行模型、漏斗参数、12 族共享底座。
- `factory_bimanual/task_family.py`：日期/任务族解析和代表 take 选择。
- `factory_bimanual/piperx_recommended.py`：配置加载、底座注册变换、PiperX 推荐协议数据类型。
- `factory_bimanual/rescue_v31.py`：纯 NumPy 的分支守卫、腕部分支签名、执行时间、A/B 调度和接缝审计。
- `factory_bimanual/recommended_follow.py`：MuJoCo warm-start DLS、40 重启锚定、候选生成、碰撞门和双臂运行结果。
- `scripts/run_piperx_recommended_v31.py`：加载任务族、构建推荐 mount 场景、运行 IK、保存 JSON/NPZ、渲染 MP4。
- `scripts/build_piperx_recommended_v31_report.py`：读取验证 JSON/NPZ 和精选截图生成中文 PDF。
- `tests/factory_bimanual/`：任务族、配置、门限、翻腕、A/B rescue、零接缝和短段 MuJoCo 集成测试。

## 数据流与失败处理

CSV 轨迹经现有数据清洗和 TCP 标定后进入推荐 mount 场景。在线 DLS 每帧只使用最后下发状态；rescue 才调用多分支候选搜索。所有候选先过姿态盒和碰撞门，再进入分支守卫/救援选择。输出 NPZ 保留源帧索引、执行帧索引、关节状态、TCP 误差、FOLLOW/HOLD/RESCUE 状态、分支签名和碰撞审计；JSON 汇总覆盖率、误差分位数、救援事件、丢帧与节拍开销。

异常处理遵循安全优先：配置缺项、任务族不匹配、首帧无法锚定、模型/TCP 不一致或碰撞审计失败均产生明确错误；运行中单帧 IK 失败只 HOLD，不终止后续源帧。

## 真实渲染与报告

验收任务使用 `8-11/Fold_Box` 来源 take 161044 和 PDF 推荐的底部共享底座。产物位于 `reports/piperx_recommended_v31/`：

- `fold_box_8-11_v31.mp4`：MuJoCo 官方 PiperX 双臂真实几何渲染；
- `fold_box_8-11_v31.summary.json` 与 `.trajectory.npz`：量化与逐帧证据；
- `figures/`：报告使用的轨迹、误差、状态和事件图；
- `piperx_recommended_v31_optimization_report.pdf`：方案、实现、验证、限制和复现命令。

视频必须通过 ffprobe/解码检查，并人工检查开头、中段 rescue、末尾三处截图。PDF 必须逐页渲染为 PNG，检查中文字体、表格、图片、页码和无裁切。

## 测试与验收

- 单元测试先失败再实现，覆盖精确边界：1.000 mm 接受、1.001 mm 拒绝，0.500° 接受、0.501° 拒绝，0.300 rad 接受、0.301 rad HOLD。
- 模式 A 过渡终点与恢复候选逐元素一致，接缝最大误差 ≤1e-12 rad。
- 模式 B 保留全部源帧并增加执行时长；丢帧和 cycle delay 分开统计。
- FOLLOW 路径不得出现未标记腕部分支改变；所有标记翻腕事件满足速度/加速度模型。
- MuJoCo 集成段没有状态或扫掠碰撞，输出所有 ACCEPT 帧均满足 ≤1 mm、≤0.5°。
- 完成前运行相关测试、完整测试集、视频解码、PDF 逐页渲染和 Git 变更/大文件审计。

## 目录整理与发布

更新根 README 为双臂 PiperX 推荐链的使用入口，并保留旧单臂/批量实验的索引。新增 `docs/ARTIFACT_MANIFEST.md`，区分精选上传产物、本地保留资产和可再生缓存。`.gitignore` 排除原始大数据、第三方压缩包、临时目录、批量历史输出、PID/日志和 Python 缓存，同时显式纳入官方 PiperX 模型、复现所需示例 CSV、最终 PDF、验证视频、关键截图与 JSON/NPZ。

目标 GitHub 仓库为 `yzh423/single-arm-mount-trajectory`。如果仓库不存在，使用当前已认证的 `yzh423` 账号创建公开仓库；提交前确保不存在单文件超过 GitHub 100 MB 限制或凭据泄漏。
