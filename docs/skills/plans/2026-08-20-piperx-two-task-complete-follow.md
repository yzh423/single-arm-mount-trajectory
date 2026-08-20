# PiperX Fold_Box + Seal_Bag 严格完全跟随实施计划

> 执行方式：本会话内按任务顺序实施；每个逻辑单元先红测、再最小实现、再回归并提交。

**目标：** 在 1 mm / 0.5° 下完成 Fold_Box 与 Seal_Bag 的逐帧严格 IK、原时序诊断、受限重定时、碰撞审计、真实 MuJoCo 视频和一个双任务 PDF，并将精选工件提交到主仓库。

**架构：** 保留 `CompleteFollowRunner` 的多分支 IK、翻腕 rescue 和动力学重定时；把 Fold_Box 专用入口泛化为按任务配置构建直立/水平/倒置场景。新增确定性候选排序与双任务证据聚合，实验阶段只写无视频轻量工件，最终候选才渲染。报告只读取最终 summary/NPZ/provenance 并复算关键数字。

**技术栈：** Python 3.11、NumPy/SciPy、MuJoCo 3.3、OpenCV、PyMuPDF/ReportLab、pytest、FFmpeg/OpenCV 解码审计。

---

## Task 1：锁定非直立构型和候选选择契约

**文件：**

- 修改：`tests/factory_bimanual/test_run_piperx_recommended_v31.py`
- 新建：`tests/factory_bimanual/test_two_task_complete_follow.py`
- 修改：`factory_bimanual/piperx_recommended.py`
- 修改：`scripts/run_piperx_recommended_v31.py`

1. 写失败测试：`horizontal_forward` 不再被执行入口拒绝，场景收到显式 base quaternion；直立构型仍使用 yaw。
2. 写失败测试：不同 family/take 产生独立 stem，summary 中记录构型 quaternion、yaw 和坐标域。
3. 写失败测试：候选排序以严格覆盖、重定时执行、误差裕量、无碰撞率、固定时序、执行时长为字典序；碰撞不能压过覆盖率。
4. 运行：`E:\Anaconda\python.exe -m pytest tests/factory_bimanual/test_run_piperx_recommended_v31.py tests/factory_bimanual/test_two_task_complete_follow.py -q`，确认新测试先失败。
5. 最小实现 mount pose 解析、任务 stem 和纯函数候选评分，再运行同一命令至通过。
6. 提交：`git commit -m "feat: generalize PiperX complete-follow mounts"`。

## Task 2：建立 Seal_Bag 可重放优化器

**文件：**

- 新建：`scripts/optimize_piperx_two_task_follow.py`
- 修改：`tests/factory_bimanual/test_two_task_complete_follow.py`
- 修改：`configs/piperx_recommended_v31.json`

1. 写失败测试：优化器枚举 PDF 构型、历史直立构型、三构型 shortlist 与确定性局部邻域；相同输入顺序和结果完全可复现。
2. 写失败测试：候选日志保存参数、严格覆盖、固定时序、重定时、误差、碰撞、时长和淘汰原因，且断点续跑不会重复已完成候选。
3. 运行双测试文件并确认红测。
4. 实现 `--family`、`--source-take`、`--output-dir`、`--resume`、`--max-rounds`、`--maximum-candidates` 和 `--no-video`。
5. 更新 Seal_Bag 最终配置时保留搜索前配置与候选日志，禁止覆盖源数据。
6. 运行单元测试至通过并提交：`git commit -m "feat: add deterministic PiperX mount optimization"`。

## Task 3：无视频基线、诊断和持续优化

**文件：**

- 生成：`reports/piperx_two_task_complete_follow/search/*.summary.json`
- 生成：`reports/piperx_two_task_complete_follow/two_task_experiment_log.json`

1. Fold_Box 快跑：`E:\Anaconda\python.exe -m scripts.run_piperx_recommended_v31 --family 8-11/Fold_Box --source-take 161044 --output-dir reports/piperx_two_task_complete_follow/fold_box --no-video`。
2. 验证 Fold_Box 严格覆盖和重定时执行均为 1061/1061；若不一致，用已有最终 NPZ 逐字段定位首次差异后修复。
3. Seal_Bag PDF 构型快跑，保存基线而不只保留最优结果。
4. 运行优化器，逐轮读取候选 summary；若严格覆盖未达 100%，根据左右侧失败分布、关节限位、翻腕 rescue 和碰撞分布缩小下一轮 mount 邻域。
5. 达到 100% 后继续优化等覆盖候选的碰撞率和执行时长，直到一轮无改进或达到计划上限。
6. 从最终 JSON/NPZ 独立复算帧数、覆盖、误差极值、速度/加速度极值、碰撞结点和入边；不一致即阻止后续渲染。
7. 提交实现/配置，不提交批量搜索缓存：`git commit -m "perf: optimize PiperX Seal Bag complete follow"`。

## Task 4：双任务正式视频与机器可读证据包

**文件：**

- 生成：`reports/piperx_two_task_complete_follow/fold_box/*.mp4`
- 生成：`reports/piperx_two_task_complete_follow/seal_bag/*.mp4`
- 生成：`reports/piperx_two_task_complete_follow/*/*.trajectory.npz`
- 生成：`reports/piperx_two_task_complete_follow/*/*.scene.xml`
- 生成：`reports/piperx_two_task_complete_follow/*/*.provenance.json`
- 新建：`scripts/validate_piperx_two_task_bundle.py`

1. 写失败测试：validator 拒绝 family/take、帧数、哈希、时间轴、容差或视频 provenance 不匹配。
2. 实现 validator，并用篡改 fixture 验证每类失败路径。
3. 对最终 Seal_Bag 构型运行正式视频；只有共享代码改变 Fold_Box 执行轨迹时才重渲染 Fold_Box，否则复制/引用现有哈希一致视频并在 manifest 记录来源。
4. OpenCV 完整解码两个 MP4，验证帧数、FPS、时长、首中尾 QA 帧和 provenance 的执行结点/入边碰撞映射。
5. 运行 validator 生成 `two_task_manifest.json` 和 `two_task_summary.csv`。
6. 提交：`git commit -m "feat: publish PiperX two-task evidence videos"`。

## Task 5：生成并目检双任务 PDF

**文件：**

- 新建：`scripts/build_piperx_two_task_report.py`
- 修改：`tests/factory_bimanual/test_piperx_complete_follow_report.py`
- 生成：`reports/piperx_two_task_complete_follow/PiperX双任务严格完全跟随实验报告.pdf`

1. 写失败测试：报告数据加载器从 manifest/summary/NPZ 复算关键指标，拒绝硬编码或缺失字段。
2. 在第一次 PDF authoring 命令前执行一次必需的 artifact marker。
3. 生成中文 PDF，包含方法、两任务对比表、Seal_Bag 迭代轨迹、误差/动力学/碰撞图、视频与工件清单、安全边界。
4. 提取 PDF 文本检查关键数字和页数，将每页渲染为 PNG 并逐页目检；修正任何裁切、重叠、乱码或空白。
5. 重新运行 validator，确认 PDF 引用哈希与最终工件一致。
6. 提交：`git commit -m "docs: add PiperX two-task experiment report"`。

## Task 6：整理 README、全量验证并发布

**文件：**

- 修改：`README.md`
- 修改：`.gitignore`
- 保留：现有报告、可视化和精选历史视频

1. README 首屏加入双任务复现命令和结果表，链接新 PDF、两个视频、manifest、CSV、NPZ、scene 和 provenance；清楚标注碰撞不等于完全跟随失败，也不是真机许可。
2. `.gitignore` 只放行权威 Seal_Bag 源 CSV 和最终精选工件，保持搜索缓存忽略。
3. 运行相关测试；再运行除已知 SciPy/LAPACK 隔离项外的完整测试，并在 `arm-design` 环境单独运行该项。
4. 运行 README 链接/代码块/CLI 检查、PDF 逐页检查、两个视频完整解码、manifest SHA-256 复验。
5. `git diff --check`，核对新增大文件、敏感信息与最终状态；明确暂存路径。
6. 提交：`git commit -m "feat: publish PiperX Fold Box and Seal Bag complete-follow bundle"`。
7. 推送当前 HEAD 到 `origin/main`，读取远端 SHA 并与本地 HEAD 精确比对。
