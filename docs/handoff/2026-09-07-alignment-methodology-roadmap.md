# 交接：当前对齐方法论评审与分阶段实施路线

**交接日期**：2026-09-07

**适用范围**：free-helix 主线、broken-helix Gate 4 支线及后续全局对齐研究

**依据**：`docs/architecture.md`、`docs/ml_alignment_principles.md`、
`docs/history/ml_optimization_log.md`、当前实现与既有实验 manifest

**文档性质**：设计评审与实施路线；本文件不记录新实验结果，不改变任何预注册判据

---

## 0. 执行摘要

当前生产流程应继续作为基线，但必须准确描述为：

```text
非目标 SMT 点 GN/LM seed
    -> 非目标 measurement GN refit
    -> 固定径迹的逐 sensor block-LM 几何求解
    -> calibration 候选选择
    -> 一次性 blind validation
```

生产 block-LM 并不是保留 track/geometry 交叉项的完整联立正规方程。它先在非目标层上
拟合径迹，再固定径迹求目标层几何。这个设计成功避免了目标层 hit 把径迹本身拉偏，且在
trial-008 上通过了端到端 blind validation；代价是未传播径迹不确定度及其引起的跨 sensor
相关性。因此它适合作为“精确参考层锚定下的目标层校准”基线，尚不能直接外推为全探测器
全局对齐方法。

建议的实施顺序是：

1. 先审计并补齐 block-LM 的数值收敛契约；
2. 同时修复实验 profile 与运行代码身份的可复现性契约；
3. 再做可观测性和参考层误差注入；
4. 只有固定径迹基线的数值与可观测性边界明确后，才启动 Schur 全局联立研究；
5. broken-helix 保持支线状态，在 Gate 4 和 24:4 backward LOC0 稳定性问题关闭前，
   不接入主线几何求解。

最值得先做的是工作包 L1“block-LM 收敛审计”。它不消费 calibration 或 validation，
不改变物理模型，而且能直接回答当前 13--16 次迭代早停究竟是驻点、边界解，还是算法性
拒步。

---

## 1. 当前可引用状态

### 1.1 已确认

- 主线有效路线是跨层参考径迹加目标层几何校准；历史 per-surface 独立拟合已经判死。
- measurement refit 已从未到驻点的 Adam 换为阻尼 Gauss--Newton，并在真实全量输入上
  达到驻点。不要再次把“替换 refit 求解器”列为新建议。
- SMT seed 也已换为逐轨迹 GN/LM；全事件 trial-010 在 390 次迭代收敛，增加预算的问题
  已关闭。不得沿用旧记录中“seed 仍是 Adam”的过期描述。
- 两段 surface warm start 于 2026-09-01 移除。直接从原始错位几何运行 block-LM，与
  warm-start 流水线的逐 sensor 几何差为 `2.4e-7 mm / 8.8e-9 rad`，且 LM 迭代从 15
  降至 13。不得恢复 warm start。
- trial-008 是当前主线可引用的 holdout 结果：80/10/10 固定分区，calibration 用于选择，
  validation 一次性使用，四个目标层分量全部通过。
- trial-010 的 `--no-event-split` 结果只证明全事件流水线可以完成。它的验收阈值仍按
  validation 样本规模标定，不能与 trial-008 的 holdout 指标比较。
- broken-helix Gate 4 已四次失败；插值改善远低于预注册的 20%。
  `24:4|backward_extrapolation|LOC0` 对协方差 damping 仍不稳定，其 NLL 结果不得引用。
- 已实测：将目标层 hit 并入径迹拟合会使 truth 距离恶化约 9.9 倍，同时可能让 chi2
  看起来更好。因此模拟研究的最终否决权必须来自 truth 距离／几何闭合，而不是 chi2。
- trial-010 的 manifest 只记录了 commit，但运行时工作树含未提交改动。该记录不可由
  commit 单独重放；历史记录已如实保留，禁止事后回填。

### 1.2 尚未证明

- block-LM 的几何解满足逐 sensor 驻点或边界 KKT 条件；当前只有 loss 下降和早停状态。
- 当前 Fisher 对角正则不会改变弱可观测模的物理解。
- 目标层增加、非目标参考层减少后，固定径迹方法仍可识别所有开放 DOF。
- 精确 nominal 非目标几何被替换为带误差的参考几何后，主线仍保持闭合。
- 完整 track/geometry Schur 联立能够避免已观察到的目标 hit 吸收偏差。
- broken-helix 或 GBL 式散射模型能改善几何，而不仅是改变预测协方差。

对以上项目不得给出置信度；必须通过相应工作包的预注册实验验证。

---

## 2. 架构诊断

### 2.1 固定径迹 block-LM 丢失局部／全局交叉信息

对几何参数 `a` 和逐径迹参数 `q`，完整线性化正规矩阵为：

```text
H = [ H_aa  H_aq ]
    [ H_qa  H_qq ]
```

消去径迹参数后的有效几何信息是：

```text
S = H_aa - H_aq H_qq^-1 H_qa
```

当前 `run_block_lm_alignment()` 把 refit 后的径迹 `detach`，随后逐 sensor 形成几何
Fisher。它近似使用条件信息 `H_aa`，没有形成上述 Schur 补。这使实现简单、内存可控、
逐 sensor 易诊断，也切断了目标 hit 直接移动径迹的路径；但参考径迹的误差可能被目标
几何吸收，几何协方差也会偏乐观。

### 2.2 当前 block-LM 的停止条件不等价于收敛证明

`scripts/ml_alignment_autograd.py::run_block_lm_alignment` 当前有四个需要先测量的问题：

- 所有 sensor 共用一个 damping；一个 sensor 接受步会降低其他 sensor 下一轮的 damping。
- 整轮无接受步时先增大 damping，随后立即退出，没有用新 damping 重试。
- `stopped_early` 由无接受步或步长小触发，没有逐 sensor Newton decrement、投影梯度或
  边界 KKT 门控。
- 候选损失使用候选自己的有效 mask；在允许少量求交失败时，下降可能来自困难 hit 消失。

这些是实现层面的可证伪风险，不表示现有几何已经错误。必须先运行 L1，才能决定是否需要
改求解器。

### 2.3 交点求解和正规方程的数值边界

helix／平面交点由有限次 Newton 更新、有效性分支和 `|s| <= 1200 mm` 截断构成。
autograd 对当前程序路径求导；在近切向、根切换或 Newton 未充分收敛处，该导数未必是稳定
交点映射的导数。几何 Fisher 又由 `J^T J` 构造，会放大 Jacobian 的条件数。普通样本上的
float32／float64 一致性不能自动覆盖弱模和极端行。

### 2.4 可观测性不能由 damping 代替

需要区分三类问题：

1. helix 原点沿轨迹滑动是参数化 gauge，当前五参数 gauge-fixed 表示已经处理；
2. 全局刚体或形变模式需要几何参考、束斑、survey、磁场等外部约束，不能靠 L2／Fisher
   floor 变成可测量；
3. strip 未测量方向、有限入射角覆盖、法向平移与倾斜组合等是数据依赖的弱模。

当前 Fisher 正则向整个矩阵增加标量对角项，能够稳定步长，但不能作为模式已被数据识别的
证据。全探测器放开后到底有多少零模或弱模不确定，必须由 M1 的无正则谱和注入闭合判断。

### 2.5 扩展更多层／事件的瓶颈

- `_sensor_residual_data()` 为每个 sensor 从整批数据构造选择，目标 sensor 增多后会重复
  扫描批数据。
- 更多事件主要增加逐径迹 seed/refit 和求交成本；几何块虽小，反复 autograd 与数据搬运
  仍会放大。
- 更严重的限制是统计结构：目标层越多，独立拟合参考径迹的非目标 hit 越少。增加事件只能
  降低统计误差，不能补回缺失的几何锚点。

---

## 3. 与全局对齐方法的设计对标

| 维度 | 当前生产方案 | Millepede II / GBL 的设计启示 |
|---|---|---|
| 局部／全局耦合 | 先 refit、后固定径迹逐 sensor 解 | 消去局部参数但保留几何间的 Schur 耦合 |
| 非线性 | helix 交点和几何更新可由 autograd 重线性化 | 全局线性解仍需外层重线性化，Schur 不会自动扩大收敛域 |
| 散射 | single helix 为主线；broken 支线未过 Gate 4 | GBL 利用散射节点的带状结构并传播完整径迹协方差 |
| 约束 | 固定参考层、活动 DOF、Fisher 正则 | 明确区分 gauge、外部物理约束和纯数值 damping |
| 规模 | 目标 sensor 可独立更新，但反复扫描数据 | 局部轨迹可并行消元，全局稀疏系统维数与径迹数解耦 |
| 验收 | 分区和预注册较严格，但 chi2 可自洽偏置 | 需要几何闭合、独立残差、重叠模块和径迹参数等互补检查 |

这里的建议不是“换一个库”。应先在现有 ACTS-independent 核心里验证 Schur 数学、约束和
闭合；Millepede II 可作为结果 oracle 或后续后端，而不是绕过当前模型问题的捷径。

参考资料：

- Millepede II 官方说明：<https://millepede.pages.desy.de/millepede-ii/>
- General Broken Lines：<https://arxiv.org/abs/1201.4320>
- CMS tracker alignment：<https://arxiv.org/abs/1010.2039>

---

## 4. 跨分支实施规则

每个工作包应使用独立分支或隔离 worktree。不要让多个分支同时修改
`ml_alignment_autograd.py` 的同一求解循环；建议按 `L1 -> M1 -> H1` 串行集成，L2 和
M3 可在不改求解器语义时并行开发。

每个算法实验都必须：

1. 在运行前写预注册块，固定假设、输入事件、主判据、次判据和证伪条件；
2. 从开始即创建 schema-versioned `manifest.json`；
3. 记录有效配置、完整命令、输入哈希、事件键／行集哈希、指标、输出、状态和代码身份；
4. 正常、失败或中止都 finalize manifest；
5. 先核对两臂的事件、轨迹、hit、component 和 sensor 集合，再读取效果指标；
6. calibration 只能用于候选选择；validation 一旦使用，不得根据其结果继续调参；
7. 算法改动先在合成回归上制造失败，再写最小修复；
8. ML、绘图和单元测试使用 conda `pytorch`；ACTS shell 与 conda Python 不得混用。

本文件本身不是算法实验，无需新增 manifest。实现分支不得修改历史 manifest 来补充缺失
字段；只能追加新的说明或产生新的可复现实验。

---

## 5. 分级工作包

### 5.1 低成本、高确定性

#### L1. block-LM 收敛与固定有效集审计（第一优先级）

> **进展（2026-09-08 追加，本段以下的预注册文字一字未改）**：代码与合成回归已完成，
> 分两个 commit —— `e9c63b2`（A 臂，只加观测，逐位不变）与 `428ef57`（B 臂，契约修复）。
> 后续评审发现四条 fail-open／漏报路径，已在预注册 v2 工作树中补回归并修复；新的 B 臂
> commit 尚待冻结。真实数据两臂尚未运行。见 `docs/handoff/2026-09-08-block-lm-convergence-contract.md`
> 与 `docs/history/ml_optimization_log.md` Iteration 27。

**目标**：判断当前早停是否为真实驻点，并防止通过丢失有效 hit 获得表面下降。

**改动位置**：

- `scripts/ml_alignment_autograd.py`
  - `run_block_lm_alignment()`
  - `regularize_fisher_matrix()`
  - `_sensor_residual_data()`
- `tests/test_ml_alignment_autograd.py`
- 新增只读审计脚本时放入 `scripts/analysis/`，不要先改生产默认值。

**建议分两阶段实施**：

1. 审计阶段只增加观测量：逐 sensor 未正则梯度、无量纲 Newton decrement、边界状态、
   原／候选固定 component loss、damping、rank 和退出原因。
2. 只有审计命中失败后，才实现逐 sensor damping、拒绝后重试和固定有效集接受规则。

**必须先写的回归场景**：

- 首次 LM 步被拒绝，但增大 damping 后存在下降步；求解器不得立即报告收敛。
- 候选仅因一个高残差交点失效而使均值下降；该步必须拒绝。
- 参数撞硬边界且投影梯度满足／不满足 KKT 的两种情况必须区分。
- 满秩线性合成问题应收敛到直接最小二乘解。

**预注册主判据**：所有被标记为 `converged` 的非边界 sensor 均满足预先冻结的无量纲
驻点阈值；合成的“拒绝后可下降”案例会继续迭代并到达直接解。

**次判据**：真实 train 输入上，候选不得丢失当前有效 component；改进臂／基线臂的
逐 sensor 几何 truth RMS 比值，其 paired 95% CI 上界不超过 `1.01`。

**证伪条件**：任一非驻点块仍被报告收敛，或 loss 改善依赖有效 component 减少。若生产
几何与基线等价，只能宣称“收敛契约得到验证”，不能宣称精度提升。

**建议验证命令**：

```bash
conda run -n pytorch pytest -q tests/test_ml_alignment_autograd.py
conda run -n pytorch make test
```

真实数据阶段必须新建实验目录并完成 manifest；不要消费 calibration／validation。

#### L2. 验收 profile 和运行代码身份绑定

> **进展（2026-09-08 追加，本段以下的预注册文字一字未改）**：代码与测试已实现，
> 尚未提交 commit。`scripts/experiment_manifest.py` 的 manifest schema 升至 2，
> 新增 `git` 字段（`dirty`／tracked diff 哈希／untracked 文件清单哈希／`probe_failed`），
> 以及 ACTS setup 脚本、conda 环境名、conda setup 脚本的身份记录（新增 CLI
> `--acts-setup`/`--conda-env`/`--conda-setup`，`run_surface_helix_experiment.sh`
> 已接上）。diff 只落哈希，不把完整 diff 文本写进 manifest 或落地成 sidecar 文件——
> 最初实现把完整 diff 文本写成同目录下的 `git_tracked_diff.patch`，直接弄坏了
> `test_compare_helix_recovery.py` 里断言输出目录精确文件集合的两个既有测试；已改为
> 仅存哈希，并把 git 探测挪到 preflight manifest 写入之后（不再抢在慢哈希失败测试
> 之前打断 "crash 前必须先有可见 running manifest" 的既有契约）。
> `scripts/reconstruction_acceptance.py`／`scripts/calibrate_acceptance.py` 的
> profile 新增 `selection_version` 与逐分区事件键哈希；消费端新增
> `AcceptanceProfileIdentityError`，在 `event_partition` 不为空时校验 seed 与事件键，
> 不符即 fail closed（不产出 verdict），且删除了原来"静默用 profile 里的 seed 覆盖
> 调用方 seed"的行为；每条验收结果新增 `profile_type`／`verdict_eligible`
> 字段（`--no-event-split` 结果为 `diagnostic_only`／`False`，有分区时为分区名／
> `True`）。`scripts/write_experiment_summary.py` 增加对 `verdict_eligible` 的纵深
> 防御检查，非 verdict-eligible 的验收结果不得产出 `passed`。全部改动在
> `conda run -n pytorch pytest -q tests`（510 passed）与 `make test-shell` 下验证
> 通过，未消费 calibration／validation。真实全流程（实际跑一次
> `run_surface_helix_experiment.sh` 产生 manifest／acceptance 并检视）尚未执行。

> **进展（2026-09-09 追加第二轮，本段以下的预注册文字一字未改）**：针对评审确认
> 尚未关闭的两类 L2 缺口继续实现，仍未提交 commit。(1) **dirty 工作树不可重放**：
> `experiment_manifest.py` 新增源码快照——`_write_source_snapshot()` 用
> `git ls-files`／`git ls-files --others --exclude-standard` 枚举当前 tracked 及
> 非 ignored untracked 的普通文件（`.git`、`data/*` 等 ignored 内容因此天然不会入档），
> 打包成 `<run_dir>/provenance/source_snapshot.tar.gz`，逐文件哈希列表作为内部成员
> 存进 tar 本身（不写进 `manifest.json`，避免小型 manifest 膨胀），manifest 只留
> `format_version`／`archive_path`／`archive_sha256`／`file_count`。新增
> `restore-source` CLI 子命令与 `restore_source_snapshot()`，恢复到空目录时校验整包
> 哈希、内部逐文件哈希、文件数三重一致，任一不符或目标目录非空即抛
> `SourceSnapshotError`。这是**主动选择的不对称设计**：`_git_dirty_state()`
> （已有的脏树探测）失败时仍是 `probe_failed=True` 的诊断性降级，但源码快照是重放
> 的唯一依据，探测或打包失败会直接抛出 `SourceSnapshotError` 终止启动（preflight
> manifest 此时已可见，由 shell 的 EXIT trap 落 `failed`）。**该功能是 opt-in**
> （`capture_source_snapshot=False` 默认关闭，CLI 对应 `--capture-source-snapshot`）：
> `write_experiment_manifest()` 还被 `compare_helix_recovery.py`／
> `material_artifact.py`／`validate_helix_model.py`／`compare_geometries.py`／
> `alignment_odd.py`等多处按 stage 调用，且多数既有测试用真实仓库根（`load_project_config()`
> 默认 `repository_root` 就是本仓库）——如果默认强制打包，每个单测都会去扫描/打包整个
> 真实仓库，且会在 `test_compare_helix_recovery.py` 断言输出目录精确文件集合的用例上
> 重演此前 diff-sidecar 文件那次回归。因此只有 `run_surface_helix_experiment.sh`
> 里唯一那次顶层 `experiment_manifest.py start` 调用打开了这个开关，其余调用点行为
> 不变。(2) **非 validation 结果可误报通过**：`reconstruction_acceptance.py` 的
> `verdict_eligible` 此前只看 `event_partition is not None`——calibration 分区的结果
> 也会被标记为 `True`，可被误用于 blind-validation verdict。现在改为要求
> `event_partition == "validation"` 且新增字段 `profile_identity_verified`（仅在
> `_verify_profile_identity()` 真正跑过且未抛异常时为 `True`）同时成立。另外新增
> `--threshold-profile-sha256`：只要 `--event-partition` 与 `--threshold-profile`
> 同时给出就强制要求这个整文件哈希，缺失或不匹配都在既有的 try/except 里 fail
> closed（成为失败结果而非崩溃）。`run_surface_helix_experiment.sh` 在
> `calibrate_acceptance.py` 冻结 profile 后只算一次 `PROFILE_SHA256`，后续所有
> `reconstruction_acceptance.py` 调用统一传入。`write_experiment_summary.py` 在
> `verdict_eligible` 之外新增对 `event_partition`／`profile_type`／
> `profile_identity_verified` 三个字段的独立复查，防的是一份手工拼出的、
> `verdict_eligible` 被直接写 `true` 但其余字段不自洽的验收 JSON。验证：
> `conda run -n pytorch make test`（539 passed / 3 skipped，跳过项是既有的 ACTS
> 绑定／pybind 扩展缺失，与本轮无关）与 `make test-shell` 全过；`make test-root-io`
> 有一个既有失败（`test_material_artifact.py::test_build_artifact_round_trips_a_synthetic_three_table_run`，
> awkward 版本不兼容导致的 `ak.to_numpy` 报错），已用 `git stash` 验证改动前就失败，
> 与本轮无关，未处理。真实跑一次 `run_surface_helix_experiment.sh` 产生并检视带
> `source_snapshot` 的 manifest，仍未执行。

> **进展（2026-09-10 追加第三轮，本段以下的预注册文字一字未改）**：评审指出前两轮
> 只是"实现完成"，预注册判据本身（真实跑通、独立目录重放、1e-6mm/1e-7rad 公差）
> 从未被验证过；另外指出冻结的 profile 哈希只活在 shell 变量里、`capture_source_snapshot`
> 只在顶层一处开启。本轮据此收口：(1) `scripts/reconstruction_acceptance.py::compute_reconstruction_acceptance`
> 新增 `threshold_profile_sha256` 关键字参数，写进返回结果字典；`main()` 把已经算出的
> `actual_profile_sha256` 传进去。这样 `experiment_summary.json`／manifest 的
> `metrics` 会自动带上这个字段（两者都是把 acceptance JSON 原样并入，不需要改
> `write_experiment_summary.py`／`experiment_manifest.py`），一份 acceptance 结果
> JSON 单独拿出来就能查到它当时锁定的是哪份 profile，不再只靠 shell 变量。
> (2) `--capture-source-snapshot` 开关（默认关闭，行为不变）扩展到另外三个可测试的
> 直接调用 `write_experiment_manifest()` 的入口：`compare_helix_recovery.py`、
> `material_artifact.py`、`validate_helix_model.py`；`alignment_odd.py` 的
> `material` 子命令也加了同名 CLI flag 并线传下去，但该文件顶部 `import acts`，
> 在 conda pytorch 环境下无法导入（`ModuleNotFoundError: No module named 'acts'`），
> 仓库里也没有 `tests/test_alignment_odd.py`，所以这一处只做了语法检查
> （`ast.parse`），没有、也无法用 pytest 验证，真实验证要等真实 ACTS 环境跑一次
> `material` 子命令。`compare_geometries.py` 按约束保持不动。三个可测试入口各新增
> 一条回归测试，确认默认值仍是 `False`（不影响现有断言，包括
> `test_compare_helix_recovery.py` 里那几处输出目录精确文件集合断言）、且显式传
> flag 后 manifest 的 `git.source_snapshot.status == "written"`
> （`compare_helix_recovery.py` 那条还额外确认 `provenance/` 出现在输出目录里）。
> (3) L3 新增独立 closure 脚本 `scripts/analysis/run_truth_veto_closure.py`，不改
> `measure_all_hit_refit.py` 一行代码，只从中 `import` 现成的
> `load_states`／`require_paired`／`select_tracks`／`build_batches`／`run_arm`，
> 把它原有的"nominal vs misaligned"两臂对比换成"baseline（对齐前）vs
> candidate（对齐后）"两臂对比，只用非目标层 hit 做 holdout refit（避开已实测的
> "目标 hit 并入拟合把 truth 距离拉偏 9.9 倍"那个坑），限定在 `validation`
> 分区事件上，对"两个目标层合并"（primary，`ci_upper < 1.0`）和"每个目标层单独"
> （required，`ci_upper <= 1.01`）分别调用 `paired_truth_distance_ratio_ci` 后交给
> `evaluate_truth_veto(after_candidate_selection=True)`；用现有
> `write_experiment_manifest`／`finalize_experiment_manifest` 写自己的 manifest，
> 不写回、不接入 `run_surface_helix_experiment.sh`。`tests/test_truth_veto_gate_isolation.py`
> 的禁止名单不包含这个新文件，属于设计使然（它就是被明确要求存在的真实调用方），
> 跑一遍确认仍然全绿。新增 `tests/test_run_truth_veto_closure.py`：`run_arm`
> 用假实现替换掉（Gauss-Newton 数值细节已经有
> `measure_all_hit_refit.py`／`validate_helix_model.py` 自己的设计和测试兜底，这
> 里只测本脚本自己的职责——分区过滤、逐层/合并指标构造、pass/fail 传导），验证了
> train 分区事件被正确排除、候选变好时 verdict passed、候选变差时 primary
> 指标失败、`--capture-source-snapshot` 正确线传。以上代码改动的验证：
> `conda run -n pytorch make test`（558 passed / 3 skipped，跳过项同前，与本轮
> 无关）、`make test-shell` 全过。**真实端到端验证（本轮跑通）**：探测时 CVMFS
> 未挂载（`setup_acts.sh` 报自动挂载需要 sudo 密码，属于交互式操作，无法自动完成），
> 请用户手动挂载后重新探测，确认 `import acts` 可用。随后真实执行了一次
> `run_surface_helix_experiment.sh --run-tag l2a-verify-20260910-c --events 20
> --max-tracks 10 --surface-epochs 2 --joint-iters 5
> --joint-measurement-refit-iters 5 --lm-max-iters 5 --skip-blind-validation
> --no-event-split`（故意用很小的迭代数/事件数/track 数只为验证机制，不追求物理
> 结果；`--joint-smt-seed-iters` 保留默认 2000——曾经把它也调小过，直接触发了
> `require_stationary_helix_point_fit` 的 fail-closed 报错，见下）。产出
> `manifest.json` 里 `git.source_snapshot.status == "written"`
> （`provenance/source_snapshot.tar.gz`，139 个文件）；用
> `experiment_manifest.py restore-source` 把源码复原到一个全新空目录，三重哈希
> 校验（整包哈希／内部逐文件哈希／文件数）全部通过。用复原目录里的
> `ml_alignment_autograd.py`，配上 manifest 里记录的完全相同的输入路径（并重新
> 核对了这些输入文件的 sha256，与 manifest 记录的一致，确认输入数据本身也没变）
> 和完全相同的参数，重跑了一次 joint-free block-LM 求解，把新产出的 geometry
> JSON 与原始运行的 2268 个 surface 逐值比较：最大平移分量差和最大旋转矩阵元素差
> 均为 `0.0`——严格优于预注册的 `1e-6 mm / 1e-7 rad` 公差，判据满足。过程中先后
> 两次尝试因为参数选得不合适而失败（一次是把 `--joint-smt-seed-iters` 调到 5
> 触发了预期内的 fail-closed 报错；另一次是外部 `timeout` 命令没能把 SIGTERM
> 正确传导到多层 subshell 里的 ACTS 子进程，导致 manifest 卡在 `running`
> 状态没被 finalize）——这两次连同其产物已作为本轮自建的临时验证数据清理掉，只
> 保留跑通的 `l2a-verify-20260910-c`。**仍未跑通**：真实 ACTS 环境下跑一次
> `alignment_odd.py material --capture-source-snapshot`（该文件顶层 `import acts`，
> 无法在 conda pytorch 环境下用 pytest 验证，需要单独在 ACTS 环境里手动跑一次）。

**目标**：阻止 full-event 与 holdout 口径混用，并补上 dirty 工作树不可重放的问题。

**改动位置**：

- `scripts/experiment_manifest.py`
- `scripts/reconstruction_acceptance.py`
- `scripts/run_surface_helix_experiment.sh`
- `tests/test_experiment_manifest.py`
- 相应 acceptance／shell contract 测试

**具体实现要求**：

- manifest schema 升版，记录 `git_commit`、dirty 状态、tracked diff 哈希、相关 untracked
  文件清单／哈希以及 ACTS 和 Python 环境身份；不要把 diff 内容直接塞入小型 manifest。
- acceptance profile 记录 split seed、partition、事件键哈希、有效计数、选择版本、bootstrap
  规模和输入 nominal states 哈希。
- profile 消费端逐项匹配；`--no-event-split` 必须使用独立 profile 类型或明确写
  `diagnostic_only`，不能继承 validation verdict。

**预注册主判据**：修改任一 profile 身份字段后，消费端在形成任何 verdict 前 fail closed；
归档代码快照和 manifest 能在独立目录恢复完全一致的输入与事件键哈希。

**次判据**：同机同后端重放的几何差不超过 `1e-6 mm / 1e-7 rad`，失败和中止路径均
产生完整 manifest。

**证伪条件**：full-event 产物仍能被标为 blind-validation passed，或重放仍依赖未归档
工作树内容。

#### L3. 新模拟实验的 truth 否决门

> **进展（2026-09-08 追加，本段以下的预注册文字一字未改）**：新增独立模块
> `scripts/analysis/truth_veto_gate.py`（不 import acts/torch，只依赖 numpy），
> 尚未提交 commit。`paired_truth_distance_ratio_ci()` 按事件分块 bootstrap 计算
> RMS(candidate)/RMS(baseline) 的 paired 比值及其 95% CI 上界（`ci_quantile=0.975`，
> 对齐 `validate_helix_model.paired_bootstrap_ci`／`compare_truth_rms.py` 既有惯例；
> 只比较两臂都有效的 hit，即 `candidate_valid & baseline_valid`）；`evaluate_truth_veto()`
> 实现主判据（严格 `< 1.0`）／次判据（`<= 1.01`）逐项独立判定、不做跨层跨分量平均，
> 且 `after_candidate_selection` 是强制参数而非默认值，必须显式传 `True`。新增
> `tests/test_truth_veto_gate_isolation.py`，grep 求解器／校准脚本确认无人引用这个
> 模块，作为"truth 不得反馈进校准循环"的操作性回归。**范围收窄，明确排除在本轮之外**：
> 尚未接入 `measure_all_hit_refit.py` 的真实 ACTS per-hit 输出（本轮只实现了统计
> 判据核心，用合成数组测试）；也未实现逐层逐 DOF 的几何误差分解——那需要要么改动
> 在途的 `scripts/compare_geometries.py`（本轮按交接文档要求未触碰），要么先独立核实
> `ml_alignment_autograd.py` 里未文档化的 DOF 索引约定，两者都留作后续工作。真实数据
> 端到端否决门尚未跑通。

> **进展（2026-09-09 追加第二轮，本段以下的预注册文字一字未改）**：修了评审确认的
> "truth 非有限值可漏过否决"缺口，仍未提交 commit。此前
> `paired_truth_distance_ratio_ci()` 对非有限 bootstrap 比值的处理是
> `resampled_ratio[np.isfinite(resampled_ratio)]`——静默丢弃非有限重采样，只用剩下
> 的有限子集算分位数；这意味着只要有部分 resample 因为撞上零 baseline RMS 而炸掉，
> 结果仍然可能是一个"看起来正常"的、可以通过否决门的 `ci_upper`。现在改为三重
> fail-closed：(1) 计算前检查两臂共享有效 hit 上的 candidate／baseline distance
> 是否全部有限，任一非有限直接抛 `ValueError`；(2) 检查全局 baseline RMS
> 是否非零且有限，否则抛错（比值本身无定义）；(3) 要求全部 `n_bootstrap` 次重采样
> 都有限，只要有一次非有限就抛错，不再静默过滤后继续。`evaluate_truth_veto()`
> 同步加固：`ci_upper`／`nominal_ratio`（若存在）必须有限，且若指标结果带
> `n_finite_resamples`／`bootstrap.replicates` 字段，两者必须相等——防的是一份
> 陈旧或手工拼出的、重采样不完整的指标结果直接被拿来判否决。新增 10 个回归测试
> （非有限 candidate／baseline distance、零 baseline RMS、部分重采样非有限、
> `evaluate_truth_veto` 对应的非有限 `ci_upper`／`nominal_ratio`／不完整重采样数），
> 连同既有用例共 21 个全部通过。**范围继续排除**：真实 ACTS per-hit NPZ 接入
> （无论是扩展 `measure_all_hit_refit.py` 的诊断输出，还是在
> `run_surface_helix_experiment.sh` 生产流程里新增 helix-refit-to-truth-distance
> 逻辑）与逐层逐 DOF 几何误差分解，2026-09-09 与用户核对后明确选择本轮不做——
> 原因不变，仍是要么触碰在途的 `compare_geometries.py`，要么需要先独立核实
> `ml_alignment_autograd.py` 未文档化的 DOF 索引约定；且生产流程
> `run_surface_helix_experiment.sh` 目前完全不计算 per-hit truth distance（只有
> chi2 验收），把 truth veto 接入生产流程需要新设计一段目前没有先例可循的逻辑，
> 留作独立的后续工作单独设计和验收，不在本轮顺带做掉。真实数据端到端否决门依旧
> 尚未跑通。

> **进展（2026-09-10 追加第三轮，本段以下的预注册文字一字未改）**：评审指出
> `tests/test_truth_veto_gate_isolation.py` 只证明了"没人调用 truth_veto_gate"，
> 而不是"有人正确地调用它"——否决门此前没有任何真实或 closure 调用方，不可能真的
> 挡住任何候选。本轮新增独立 closure 脚本
> `scripts/analysis/run_truth_veto_closure.py`，作为第一个真实调用方，解决"无调用
> 方"和"未接真实数据管线"两个缺口（仍未解决的是"真实 ACTS 数据"——见下）。设计：
> 完全复用 `scripts/analysis/measure_all_hit_refit.py` 已有的
> `load_states`／`require_paired`／`select_tracks`／`build_batches`／`run_arm`（不
> 改该文件一行代码，只 `import` 它的函数），把它原本的"nominal vs misaligned"两臂
> 对比换成"baseline（`run_surface_helix_experiment.sh` 的对齐前输入几何）vs
> candidate（其最终产出几何）"两臂对比——这正是 L3 要回答的问题本身。只用非目标层
> hit 做 holdout refit（复用同一份已实测的"目标 hit 并入拟合把 truth 距离拉偏
> 9.9 倍"教训，refit 阶段避开它），限定在 `--event-partition validation`（复用
> `scripts/event_split.py::event_partition_mask`，默认即 `validation`，避免拿校准
> 阶段挑过候选的事件再来做否决检查），对"两个目标层合并"（primary，
> `ci_upper < 1.0`）和"每个目标层单独"（required，`ci_upper <= 1.01`）分别调用
> `paired_truth_distance_ratio_ci` 后交给
> `evaluate_truth_veto(after_candidate_selection=True)`，用现有
> `write_experiment_manifest`／`finalize_experiment_manifest` 写自己的 manifest
> 和一份 `truth_veto_result.json`，不写回、不接入
> `run_surface_helix_experiment.sh`——`tests/test_truth_veto_gate_isolation.py` 的
> 禁止名单不包含这个新文件，属于设计使然，跑一遍确认仍然全绿。新增
> `tests/test_run_truth_veto_closure.py`：`run_arm` 换成假实现（真实
> Gauss-Newton 数值细节已经有 `measure_all_hit_refit.py`／
> `validate_helix_model.py` 自己的设计和测试兜底，这里只测本脚本自己的职责——
> 事件分区过滤是否发生在拟合之前、逐层/合并指标是否构造正确、pass/fail 是否正确
> 传导），验证了 train 分区事件被正确排除、候选变好时 verdict passed、候选变差时
> primary 指标失败、`--capture-source-snapshot` 正确线传。**范围继续排除**：真实
> ACTS per-hit truth-distance 数据源（本轮的合成 DataFrame 测试绕开了
> `load_states` 的 ROOT 读取）；逐层逐 DOF 的几何误差分解（原因不变：要么触碰
> 在途的 `compare_geometries.py`，要么需要先独立核实 `ml_alignment_autograd.py`
> 未文档化的 DOF 索引约定）；重建效率／求交成功率／尾部 truth distance 分布；
> 把这个 closure 脚本自动接入 `run_surface_helix_experiment.sh`（目前仍是跑完
> 实验后手动调用一次）。这些留作独立的后续工作单独设计和验收。

> **真实数据端到端验证（本轮跑通，紧接 L2-A 的 CVMFS 恢复之后）**：针对
> L2-A 里那次真实 `l2a-verify-20260910-c` 跑出的产物，对
> `scripts/analysis/measure_all_hit_refit.py::load_geometries` 的复用方式做了一处
> 修正——最初实现直接对 `baseline_geometry`/`candidate_geometry` 调用
> `build_geometry(json.loads(path.read_text()))`，把它们当成自包含的全量探测器
> 几何；真实数据上第一次尝试就在 `build_batches` 里对非目标层 hit 抛出
> `KeyError`，因为 `misalignment_applied_v17l8_v24l4.json`／
> `ml_alignment_autograd.py --output-geom` 产出的都只是**目标层区域的 overlay**
> （本次真实数据里是 2268 个 surface，相对全探测器 18824 个 surface 而言），
> 而一条真实径迹的非目标层拟合 hit 会散布在整个探测器上。修正为改用
> `load_geometries(nominal_full_geometry_path, overlay_path, target_layers)`
> （新增必填 CLI 参数 `--nominal-full-geometry`），让 overlay 正确合并到完整
> nominal 几何上再建 `id_to_idx`，与 `measure_all_hit_refit.py` 自己的既有约定
> 一致。同步更新了 `tests/test_run_truth_veto_closure.py` 的几何 fixture（拆成
> "nominal 全量" + "overlay 仅目标层"两个文件，不再共用一个自包含文件）。之后
> 用 `alignment_odd.py reconstruction --num-events 20`（与 L2-A 验证跑同一套
> misaligned 几何、同一批模拟事件）额外重建了一份与候选产物事件数匹配的 baseline
> 重建（这是必要的：`require_paired` 要求两臂逐 hit 完全对齐，而
> `data/reconstruction/misaligned/.../trackstates_kf_v17l8_v24l4.root` 是早先一次
> 更大规模重建的产物，事件集合和这次 20 事件的候选重建对不上，直接拿来配对会
> fail closed）；用它和 `l2a-verify-20260910-c` 的 `surface_polish2` 最终几何/态
> 真跑了一次 `run_truth_veto_closure.py`，产出真实 verdict：`passed: true`，
> `target_combined`/`17_8`/`24_4` 三个指标的 `ci_upper` 都在 1.0 附近（约
> 0.99999999998–1.00000000004，因浮点误差有正有负）。**明确的统计局限，不构成
> 精度改善声明**：这次验证只用了 2 条径迹（`--max-tracks 10`、目标层选择后剩 2
> 条满足 span/hit 数条件），bootstrap 在如此小的样本上几乎不提供任何统计功效，
> `ci_upper` 贴着 1.0 纯属样本量太小、方差被 bootstrap 拉到几乎覆盖整个可能区间，
> 不能读成"候选比基线好"的证据；这次验证的目的只是证明 closure 脚本能在真实
> ACTS 产物上端到端跑通、产出结构正确的 verdict，不是一次有统计意义的精度评估。
> 另外用同一次真实 ACTS 环境验证了 L2-C 里未能用 pytest 覆盖的
> `alignment_odd.py material --capture-source-snapshot`（`--num-events 5` 的最小
> 材料导出），manifest 的 `git.source_snapshot.status == "written"`，验证产物已
> 清理，不留在 `data/` 下。`l2a-verify-20260910-c` 连同它的
> `truth_veto_closure/` 子目录作为本轮真实验证证据保留在
> `data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4/` 下（其余两次因参数
> 选择不当/被外部 timeout 误杀而失败的自建验证目录已清理，不保留)。

**目标**：防止“chi2 更好、truth 更差”的候选进入算法结论。

truth 指标仅用于候选产生之后的一次性否决检验，不得作为校准过程中的优化目标、早停
条件或候选筛选依据。calibration 阶段的候选选择必须只使用 chi2／似然等不依赖 truth
的判据；truth 距离只在候选选定后执行一次门控检查。原因：truth 只在模拟中可得，若
把它接入校准过程本身，方法就会学会利用真实数据不具备的信息压低 truth 距离，得到的
校准规则无法迁移到真实数据。

**改动位置**：

- 先在独立 analysis／closure 脚本中实现，不改变 trial-008 已冻结的生产验收协议。
- 可复用 `scripts/analysis/measure_all_hit_refit.py::truth_distance()`。
- 几何指标应扩展为逐层逐 DOF 的绝对误差、RMS 和 paired 差值，不能只看 correlation／
  recovery。

**预注册主判据**：任何“精度改善”声明都要求预定主 truth 指标的 paired 95% CI 上界
小于 `1.0`；其他必需 truth 指标的上界不超过 `1.01`。

**次判据**：ACTS 重建效率、求交成功率及预定尾部 truth 距离不劣化。

**证伪条件**：chi2 改善而任一必需 truth 指标显著恶化，候选直接失败，不允许跨层或跨
分量平均抵消。任一环节把 truth 距离用于候选选择、超参数搜索或早停，该实验即记为
方法论违规，其否决权判定失效，须重新按合规流程预注册。

该门只适用于 truth-known 模拟。真实数据必须用 survey、重叠模块、独立径迹样本或其他
外部约束替代，不能伪造 truth 等价物。

### 5.2 中成本、需要验证

#### M1. 可观测性和参考层误差注入

**目标**：区分数据可观测模式、参考约束和正则选择出的模式。

**前置条件**：L1 完成；否则谱变化可能只是求解器未收敛。

**改动位置**：

- 新建 `scripts/analysis/measure_alignment_observability.py`；
- 复用生产 residual/Jacobian，但输出未正则化、无量纲 Fisher 的特征值和特征向量；
- 注入目标层正／负扰动，以及非目标参考层的小幅协同扰动；
- 生产默认 active DOF 和 Fisher floor 在测量阶段保持不变。

**预注册主判据**：预先声明为可观测的模式，在正负注入和独立噪声重复中，恢复斜率的
95% CI 完全位于 `[0.9, 1.1]`。

**次判据**：参考层误差传递到目标几何的偏差低于运行前冻结的误差预算；Fisher floor
改变十倍时，truth RMS 比值的 95% CI 上界不超过 `1.01`。

**证伪条件**：低 chi2 下模式恢复失败，或物理解主要随正则变化。该模式必须标记为
`unobservable`／`prior-dominated`，不得宣称已对齐。

#### M2. Gate 4 活动子空间协方差稳定性

**目标**：隔离 `24:4|backward_extrapolation|LOC0` 的数值敏感性，不借修改物理模型
推动 Gate 4 通过。

**前置条件**：冻结 Gate 4 v3 的两臂参数、行集、节点和 segment assignment；不得重新
拟合或改阈值。

**改动位置**：

- `scripts/validate_helix_model.py::predictive_covariance()` 旁新增可测试的 QR／SVD oracle；
- `tests/test_validate_helix_model.py` 增加参数重缩放和病态活动子空间对照；
- 正式对照先放 analysis 脚本，不替换生产 Cholesky 路径。

**预注册主判据**：数学等价的 Cholesky、QR／SVD 与参数重缩放下，该组 `delta_mean_nll`
位移小于原 paired CI 半宽的 `0.1` 倍，并给出相同判定。

**次判据**：均值预测、paired 行集、rank 和所有未修改 Gate 4 字段完全一致。

**证伪条件**：活动秩或 NLL 仍随数值表示显著变化。P1' 保持开放，该组结论继续不可引用。

若 M2 通过但 Gate 4 仍失败，才可独立预注册“保持资格门槛、扩大外推拟合窗口”的实验。
是否有效不确定，不能预设成功。

#### M3. 数据索引、分块和正规矩阵累加优化

**目标**：降低更多 sensor／事件下的重复扫描和峰值内存，不改变统计问题。

**改动位置**：

- 在 `build_track_batches()` 后预计算 sensor 到 `(track_index, hit_index)` 的映射；
- `_sensor_residual_data()` 消费预计算索引；
- 按轨迹或 sensor 分块累加 `J^T J` 和 `J^T r`，正式诊断优先 float64；
- 不改变行集、权重、component mask 或求交规则。

**预注册主判据**：固定硬件和输入上，峰值 RSS 降低至少 30%，或墙钟降低至少 20%；
运行前只能选择其中一个作为主指标。

**次判据**：两实现的几何差不超过 `1e-6 mm / 1e-7 rad`，失败计数、事件键和验收结论
一致。

**证伪条件**：性能收益来自删轨迹／component，或分块顺序使弱模解超过数值容差。

### 5.3 高成本、高风险

#### H1. 受约束的 Schur 全局联立求解

**目标**：在同一次线性化中保留 track/geometry 交叉项，并将逐径迹参数消元后求全局
几何增量。

**前置条件**：L1、L3、M1 全部完成；先只做 single helix 和 `(tx, ty, rz)`，不得同时
引入 broken-helix。

**建议边界**：

- ACTS-independent 核心负责 Jacobian、局部块消元、全局约束和求解；
- 每条径迹形成 `H_qq`、`H_qa` 和梯度，累加 Schur 系统；
- gauge、survey／参考约束和数值 damping 使用不同的数据结构及 summary 字段；
- 小问题以完整稠密正规方程为 oracle，生产路径用 Cholesky／QR 或稀疏迭代法；
- 外层重线性化必须继续检查真实目标下降和求交有效性。

**预注册主判据**：小规模闭合问题中，Schur 步与完整系统步的无量纲相对差低于 `1e-8`；
随后独立模拟中，几何 truth RMS 相对固定径迹基线改善至少 10%，以 paired 95% CI 上界
判定。

**次判据**：径迹 truth 距离不劣化超过 1%，所有必需层／DOF 均通过，且结果不依赖
精确 nominal 非目标参考。

**证伪条件**：再次出现 chi2 降低而 truth 距离显著增加，或改善只在精确 nominal 参考下
成立。

H1 与已失败的“把目标 hit 加进固定错位几何后重拟合径迹”不同：几何在同一线性化中同步
变化，交叉项被保留。但能否避免约 9.9 倍偏差不确定，必须由 L3 否决门判断。

#### H2. GBL 式结构化散射模型

**目标**：仅在数据证明 single helix 模型误差成为几何瓶颈后，研究带散射节点、输运
Jacobian 和完整协方差的局部模型。

**前置条件**：M2 关闭数值不稳定；broken-helix Gate 4 在冻结标准下通过；H1 的 single
helix 闭合基线已建立。

仓库已有更完整的设计和实施计划：

- `docs/superpowers/specs/2026-09-01-broken-joint-geometry-identifiability-v2-design.md`
- `docs/superpowers/plans/2026-09-01-broken-joint-geometry-identifiability-v2.md`

后续分支应复用上述计划，不另建重复框架。当前阶段不得执行其中的主线 joint 接入部分。

**预注册主判据**：在新 blind 样本上满足现有 Gate 4：插值 RMSE 改善 CI 达到 20%，外推
回退不超过 5%，并同时满足 pull、预测 chi2、行数和稳定性门控。

**次判据**：只有 Gate 4 通过后才运行几何联立；几何 truth 与径迹 truth 均不得劣化。

**证伪条件**：改善只来自协方差膨胀，或 kink／材料先验吸收几何而 truth 无改善。

---

## 6. 明确排除的路线

### 6.1 不恢复 per-surface 独立拟合

历史实现因全局刚体／参考自由度退化而失败。L1 的“逐 sensor 数值审计”仍使用跨层非目标
径迹作为固定参考，目的只是验证现有 block 求解器，因此不等价于恢复历史路线。若某实现
取消跨层参考或未来 Schur 耦合，就重新落回被否决的方法。

### 6.2 不恢复 IRLS、Huber 或按残差学习 hit 权重

历史实验已经表明，大残差和模块边缘 hit 正在携带 `ry` 等错位信号；按残差幅度下权会把
信号当异常值删除。M2/H2 的协方差传播来自显式物理模型，不根据拟合残差大小自适应调权，
所以与 IRLS 不同。

### 6.3 不重新数字化

没有新证据表明重新数字化能解决参考误差、局部／全局耦合或弱模问题。除非未来得到新的
因果证据并单独预注册，否则不进入路线图。

### 6.4 不恢复 surface warm start

direct block-LM 已与 warm-start 结果达到数值等价且迭代更少。L1 应检查 direct 求解的
驻点性质，而不是用 warm start 掩盖停止条件问题。

### 6.5 不直接用目标 hit 重拟合径迹

该路线保留为负对照。H1 必须同时更新几何、保留交叉项并接受 truth 否决；否则只是重复
已知会把结果拉偏约 9.9 倍的实验。

### 6.6 不放宽 Gate 4 或重复读取 validation

Gate 4 未通过就是模型证据不足；不能通过修改阈值、删除困难行或用不稳定 NLL 组翻案。
blind validation 一旦用于调参，其确认功能已经消耗，后续正式确认必须使用新的冻结数据。

---

## 7. 第一项实验的具体交接

### 实验名

`block-lm-convergence-audit-v1`

### 实验问题

在冻结的 train 径迹和原始错位几何上，当前 block-LM 早停是否满足逐 sensor 驻点／边界
条件？改为逐 sensor damping、拒绝后重试和固定有效集后，是否改变 truth 几何？

### 两臂

- **A：baseline**。当前生产实现，不改变求解过程，只补审计输出。
- **B：contract-fixed**。逐 sensor damping；拒绝后提高 damping 并重试；接受比较使用当前
  固定有效 component；以驻点／边界 KKT 而非“本轮无接受步”宣告收敛。

### 必须冻结

- 同一 train event-key hash；
- 同一 refitted track 参数文件及哈希；
- 同一目标 sensor、active DOF、measurement、covariance 和初始几何；
- 同一求交容差、Fisher floor、步长上限与硬边界；
- 不运行 calibration，不运行 validation，不运行 surface polish。

### 结果解释

- B 通过且 truth 改善：进入生产候选，但仍需新的 calibration 选择协议；
- B 通过且两臂几何等价：证明当前结果的数值稳定性，停止继续改 block-LM；
- B 驻点更好但 truth 变差：保留 baseline，转入 M1 查弱模／正则；
- 两臂都非驻点：继续归因求交 Jacobian、尺度或边界，禁止直接调大迭代数；
- 改善依赖丢 hit：实验失败，修复接受规则后重新预注册版本。

### 实施检查表

- [ ] 在 `docs/history/ml_optimization_log.md` 先写预注册条目。
- [ ] 为四个合成场景写失败回归测试。
- [ ] 增加只读审计字段，确认 A 臂数值行为未变。
- [ ] 完成 A 臂 manifest，核对输入和事件键哈希。
- [ ] 实现 B 臂最小改动并运行定向测试。
- [ ] 顺序运行真实数据两臂，避免内存竞争。
- [ ] 先核对行集／component／sensor 数，再读取 loss 和 truth。
- [ ] 无论通过、失败或中止，都 finalize 两臂 manifest。
- [ ] 将结果和未关闭问题追加到优化日志，并更新下一份交接。

---

## 8. 分支领取顺序与完成定义

| 顺序 | 工作包 | 可与其他包并行 | 完成定义 |
|---:|---|---|---|
| 1 | L1 收敛审计 | L2 | 合成回归、真实 train A/B、双 manifest、预注册判据核对完成 |
| 1 | L2 身份与 profile | L1 | schema 升版、fail-closed 测试、成功／失败重放测试完成 |
| 2 | L3 truth 否决门 | M3 | 独立模拟协议冻结，9.9 倍负对照可被门控拒绝 |
| 3 | M1 可观测性 | M2、M3 | 正负注入、参考误差注入、谱和正则敏感性结论完成 |
| 3 | M2 Gate 4 数值稳定 | M1、M3 | 等价算法对照完成；稳定则关闭 P1'，否则保留不可引用状态 |
| 3 | M3 性能 | M1、M2 | 行集不变前提下满足预注册性能主判据 |
| 4 | H1 Schur 联立 | 无；依赖 L1/L3/M1 | 稠密 oracle、约束闭合、truth A/B 和 manifest 完成 |
| 5 | H2 GBL／broken joint | 无；依赖 M2/H1/Gate 4 | 执行既有 v2 计划，先模型门控、后几何闭合 |

单个分支只有在以下条件全部满足时才算完成：代码与测试通过、算法实验 manifest 完成、
预注册判据逐条核对、失败结果同样归档、文档没有把 diagnostic-only 数字写成正式结论。

---

## 9. 当前工作树提醒

创建本文档时工作树已有与本任务无关的修改／未跟踪文件：

```text
M  scripts/compare_geometries.py
M  tests/test_compare_geometries.py
?? NUL
?? docs/history/experiment_artifact_cleanup_2026-09-02.md
?? scripts/analysis/measure_all_hit_refit.py
?? surface_alignment_logic.pptx
```

这些内容视为其他工作所有。本交接只新增当前文件；后续提交时不要顺手纳入、覆盖或清理
上述文件。`measure_all_hit_refit.py` 是 L3 可复用的现有未跟踪分析代码，使用前应先确认其
所属分支和预期提交方式。
