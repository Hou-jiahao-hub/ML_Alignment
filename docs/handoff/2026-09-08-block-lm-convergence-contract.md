# 交接：block-LM 收敛与固定有效集契约（roadmap L1）

**日期**：2026-09-08
**分支**：`feat/full-event-calibration`
**上游**：`docs/handoff/2026-09-07-alignment-methodology-roadmap.md` §L1
**性质**：数值契约修复。**不是物理改动。**
**状态**：代码与合成回归完成；真实数据两臂**尚未运行**。

---

## 执行摘要

**结果**：三个 commit（`e9c63b2` A 臂只加观测、`428ef57` B 臂契约修复、`fc89641` 预注册
与离线验证脚本）。四个已知缺陷（全局 damping、拒绝后不重试、比较分量不固定、无驻点
判据）全部修复，`484 passed, 3 skipped`，A 臂对合成问题的求解输出与 `37ed4f5`
逐位相同。合成层的主/次判据、证伪构造（秩亏、求交失败、边界 KKT、block 解耦）均已
写成回归并通过。新增 `scripts/analysis/compare_truth_rms.py` 并在合成几何上验证了
通过/失败/作废三条分支。

**遇到的问题**：详见 §13——既有测试的全等断言随新增诊断字段被打破、既有容差假设与
新驻点契约冲突、damping 汇总统计口径选错导致断言失败、`block_lm_blocks` 的索引假设、
合成 fixture 本身不满秩导致主判据测试空转、float32 精度让求交失败的注入测试误伤当前点、
`compare_truth_rms.py` 的样本核对最初用交集而非全集（差点重犯 damping-v2 记录过的
"静默丢样本"错误）。全部已解决，见 §13 的具体修法。

**待完成**：真实数据两臂尚未运行，详见 §11——这是本次交付的边界，不是遗漏：roadmap L1
要求先在合成层把契约钉死，真实数据的两臂对比需要下一轮会话在冻结好 train 输入之后专门
执行，且预注册明确要求"先写预注册、再跑数据"，本次会话已完成预注册（Iteration 27）
但尚未执行跑数据这一步。

---

## 0. 宣称边界（先读这条）

本改动只修正 block-LM 的**收敛与接受契约**。它使 `converged` 成为可证伪的陈述，
并关闭「χ² 改善来自丢失有效 component」这条路径。

若真实数据上产出的几何与基线等价，**只能宣称契约得到验证，不得宣称物理精度提升**。

本改动**不解释也不修正已知的 9.9 倍 truth 距离偏差**。该偏差的主判据是 truth 距离
而非 χ²，成因是全 hit refit 让径迹吸收几何位移，在固定径迹的 block-LM 之外，不在
本工作包范围内。

**truth 在真实物理场景中不存在。** 它只在带标注的实验里可得，只能用于校准完成之后
核对模型是否正确。求解器内部任何一处引用 truth 都是错的；`compare_truth_rms.py` 是
唯一可以谈 truth 的地方，它的输出也不得反过来指导求解器调参。

---

## 1. 两个 commit

按 roadmap L1「建议分两阶段实施」执行。最终代码里**没有 legacy 开关** —— A 臂的读数
来自 stage-1 commit 本身，不需要在产品代码里留一条旧路径。

| commit | 内容 | 验收 |
|---|---|---|
| `e9c63b2` | **A 臂**：只加观测量 | 合成两例 `predicted_deltas` 与 `history` 与 `37ed4f5` **逐位相同** |
| `428ef57` | **B 臂**：契约修复 | 主/次判据回归全绿 |

A 臂的逐位验证做法：把两个合成 case 的 `predicted_deltas` 和 `history` 序列化成 JSON，
在 `git stash` 前后各跑一次 `diff`。结果 `BIT-IDENTICAL`。

---

## 2. 修了什么（对应 roadmap §2.2 的四条）

| roadmap §2.2 | 旧行为 | 新行为 |
|---|---|---|
| 所有 sensor 共用一个 damping | `damping` 是标量，任一 sensor 接受就整体减半 | `damping_by_sensor` 逐 sensor，跨外层迭代延续 |
| 增大 damping 后立即退出 | `damping *= 10` 后紧接 `break` —— **抬高的 λ 从未被任何一次求解使用过** | 同一 sensor、同一外层迭代内最多 8 级 ×10 重试；λ 写回，下一轮从抬高后起步 |
| `stopped_early` 没有驻点门控 | 布尔，混淆收敛与卡死 | 四类终态：`converged` / `budget_exhausted` / `rank_deficient` / `intersection_failed` |
| 候选用自己的 mask | `current[component_mask].mean()` vs `candidate[candidate_mask].mean()`，集合与分母都不同 | 两侧同在 `fixed_mask` 上求 sum；候选丢任一原有效分量即拒绝 |

重试**不重算 Jacobian**：线性化与 λ 无关，整条阶梯复用一次 `_block_lm_linearized_block`，
成本约为一次额外 block 访问的零头。

---

## 3. 冻结的常量（`scripts/ml_alignment_autograd.py`）

```
BLOCK_LM_PREREGISTRATION_VERSION = 1
BLOCK_LM_DECREMENT_STATIONARY    = 1.0e-6     # 每 component
BLOCK_LM_MAX_DAMPING_RETRIES     = 8
BLOCK_LM_MAX_DAMPING             = 1.0e12
BLOCK_LM_MIN_DAMPING             = 1.0e-12
BLOCK_LM_BOUND_ATOL              = 1.0e-12
```

**驻点判据**：`lambda2 / n_components <= 1e-6`，其中
`lambda2 = g_free^T H_free^-1 g_free`。`standardized()` 返回 `r/σ`，故 `r^T r = χ²`、
`g = J^T r = ∇F`、`H = J^T J`，gauge 是 `F = 0.5·χ²` —— 与
`validate_helix_model.py` 的 `NEWTON_DECREMENT_STATIONARY` 同一 gauge、同一数值，但
**口径不同**（那里每径迹行，这里每 component）。两者独立冻结，不得互相调整。

**刻意分歧**：`newton_decrement()` 解算前加 `GAUSS_NEWTON_MIN_DAMPING`。因为
`(H + εI)^-1 ≼ H^-1`，加正则**只会让 decrement 偏小** —— 把「没收敛」报成「收敛」，
正是本次要消灭的失效模式。所以这里的驻点判据一律不加任何正则，改用谱分解：弱于
Fisher floor 的方向从求和里剔除，剩下的值标记为 `lambda2_is_lower_bound`。

`BLOCK_LM_MAX_DAMPING_RETRIES = 8` 的理由：λ 从 `1e-3` 起 8 级到 `1e5`，此时步长约为
GN 步的 `1e-8` 倍、方向趋近最速下降，对光滑目标必然下降（除非梯度已消失，那由驻点
判据先拦下）。8 级后仍不降 ⇒ 光滑性假设被证伪（`intersect_plane` 的有效性分支与
`|s| ≤ 1200 mm` 截断），值得单独报告而非继续烧预算。

---

## 4. 四类终态的判定优先级

```
1. intersection_failed  阶梯耗尽，且过程中出现过 component_loss 或 invalid_fraction 拒绝
2. rank_deficient       H_free 有方向弱于 Fisher floor，或 μ_min <= 0，或走了 lstsq 回退
3. converged            lambda2/n <= 阈值，或自由集为空（箱约束 KKT 点）
4. budget_exhausted     outer_iters | damping_retries
```

**既秩亏又达阈值 ⇒ 报 `rank_deficient`。** 秩亏时 `lambda2` 只在良态子空间上求和，是
真值的**下界**；用一个系统性偏小的统计量通过「≤ 阈值」的检验，正是主判据要防的事。

**`rank_deficient` 不提前终止该 block。** 今天的生产几何正是由 Fisher floor 撑起的步
产生的；立刻退休会大幅改变真实结果，就无法区分「契约修好了」和「几何变了」，宣称边界
直接失守。该 block 继续按新契约走步，只是**永不被报成 converged**。

**为什么不能只看 `condition_number`**：① 它算在全 active Fisher 上，被箱约束冻结的
弱 DOF 会把好 block 误判秩亏；② 它是比值 —— `μ_max = μ_min = 1e-14` 的条件数是 1，
看起来完美，但 `H^-1` 会把噪声放大成巨大的 `lambda2`；③ `μ_min ≤ 0` 时它是 `inf`，而
`_json_ready` 把非有限 float 落成 `null`，写盘后「精确奇异」与「未计算」不可区分；
④ 它不含「梯度是否落在弱方向上」的信息 —— 那才是关键，由 `null_gradient_max` 补上。

**两类约束必须分开**：`joint_clamp_t/r` 是真实可行域，进 KKT；`lm_max_step_t/r` 是
信赖域，**不进** KKT，但被截断的步单独计数（截断后的步不再是 LM 步）。

---

## 5. block 之间精确解耦 ⇒ converged 可永久退休

`_sensor_residual_data` 的选择集是 `hit_mask & is_target & (sensor_idx == s)`，只依赖
`batches`；几何只取 `centers[s]` / `rotations[s]` 与该行 alignment；`tracks` 全程只读。
⇒ sensor `i` 的残差、Fisher、gradient 只是 `alignment[i]` 的函数，**其它 block 移动
不可能使它离开驻点**。这不是近似，是精确结论，回归见
`test_block_lm_blocks_are_decoupled_across_target_sensors`。

---

## 6. `history` 不再保证单调 —— 这不是 bug

接受守卫保证 `candidate_mask ⊇ fixed_mask`，因此每个 block 的有效分量数沿接受步
**单调不减**。代价是：集合增长后目标函数多了项，`history` 跨迭代比较的是**项数不同的
两个目标**。它的抬升本身不构成异常。已写进 `run_block_lm_alignment` 的 docstring。

---

## 7. 测试容差的修订（**这是修正，不是放松**）

`tests/test_joint_alignment.py::test_block_lm_recovers_target_translation_without_changing_tracks`
原断言 `pytest.approx(-0.4, abs=1e-5)`，在新契约下**必然失败**，改为闭式驻点界。

推导（该 fixture 完全解析）：法向沿 x、局部轴 (y, z)，target 中心 `(10, 0.4, 0)`，
12 条零曲率直线轨迹。loc0 残差恰为 `0.4 + tx`，`∂r/∂tx = 1`；loc1 残差恒 0。
σ = 0.1，m = 12 行，n = 24 分量 ⇒ `H = m/σ² = 1200`、`lambda2 = 1200 e²`。

```
1200 e² / 24 <= 1e-6   <=>   |e| <= sqrt(24 · 1e-6 · 0.01 / 12) = 1.4142e-4 mm
```

新代码的实际轨迹（`λ0=1e-3`，`lm_max_step_t=0.1`，`reg = 1200λ` ⇒ 原始步 `-e/(1+λ)`）：

| iter | 起始 e | λ | 原始步 | 截断后 | 结束 e |
|---|---|---|---|---|---|
| 0 | 0.4 | 1e-3 | −0.3996 | −0.1 | 0.3 |
| 1 | 0.3 | 5e-4 | −0.2999 | −0.1 | 0.2 |
| 2 | 0.2 | 2.5e-4 | −0.2000 | −0.1 | 0.1 |
| 3 | 0.1 | 1.25e-4 | −0.0999875 | 不截断 | **1.2499e-5** |
| 4 | — | — | — | — | 驻点检查 `1200·(1.2499e-5)²/24 = 7.811e-9 ≤ 1e-6` ⇒ **converged** |

实测 `lambda2_per_component = 7.811e-9`，与闭式**精确吻合**。终点误差 1.2499e-5，
超出旧的 `abs=1e-5` 25%。

旧代码会继续走第 5 步把 e 压到 ~7.8e-10，因为它的停止条件是「步长 < 1e-10」。
**驻点判据说的是「模型还能给的下降 ≤ 1e-6/component」，它从未承诺参数精度到 1e-5 mm。**
新断言直接写闭式界，让测试本身成为契约的可执行说明。

---

## 8. 合成层的判据核对（已完成）

| 判据 | 测试 | 结果 |
|---|---|---|
| 主 A：converged 块必须驻点 | `test_block_lm_converged_blocks_are_independently_stationary` | PASS，两个载体 |
| 主 B：首拒绝→加 damping→继续求解 | `test_block_lm_retries_with_more_damping_after_a_rejection` | PASS |
| 次 A：接受步不丢分量 | `test_block_lm_accepted_steps_never_lose_components` | PASS，两个载体 |
| 秩亏 | `test_block_lm_reports_a_structurally_blind_dof_as_rank_deficient` | PASS |
| 求交失败 | `test_block_lm_refuses_a_candidate_that_buys_its_improvement_by_losing_a_hit` | PASS |
| 边界 KKT | `test_block_lm_calls_a_bound_it_cannot_leave_a_converged_kkt_point` | PASS |
| 解耦 | `test_block_lm_blocks_are_decoupled_across_target_sensors` | PASS |

**主判据 A 的独立重算**：测试不复用求解器缓存，自己在**物理单位**下重算
`lambda2 = Σ (q_kᵀg)²/μ_k`（求解器工作在归一化单位）。`lambda2` 对对角重标度不变
（`g → Sg, H → SHS`），所以这是一条真正独立的路径，却必须给出同一个数 —— 测试同时
断言 `rel=1e-6` 内相等。

**主判据 B 用确定性注入，不用真实非线性。** 真实非线性构造不可靠：
`_block_lm_translation_problem` 对 `tx` 严格线性，二次目标下任意 `λ ≥ 0` 的 LM 步都是
GN 步的收缩（`step = −e/(1+λ)`），**永远不会过冲、永远不会被拒**。改用
`build_synthetic_joint_case` 能做出来，但拒绝与否依赖 noise、seed、曲率一串数字，
fixture 微调会让「拒绝」悄悄消失，测试变成永真式。改为 monkeypatch
`_block_lm_fixed_chi2` 让第一个候选必然被拒，并断言**重试确实发生在同一 sensor、
同一外层迭代**（`attempt == 1`、`damping` 恰好 ×10、随后 `accepted`），而不只是
「最后收敛了」。这条在旧代码上必然失败：旧代码 λ×10 后立刻 break，`fisher_diagnostics`
里根本不存在 `attempt` 字段。

**关于第二个测试载体**：`build_synthetic_joint_case` 原样**不是**解析可解的满秩问题
—— 它注入 `noise=0.02` 且把**故意错的冻结径迹**交给求解器，block 是在对一个错设模型
拟合几何，300 次迭代后 `lambda2/n` 仍只降到 `2.577e-6`（且 66% 的步仍被信赖域截断）。
主判据 A 因此改用 `_analytic_multi_dof_case`：同一几何、`noise=0.0`、径迹用 truth、
只往 target 平面注入已知偏移 `(局部 tx=−0.3, ty=+0.2)`。它 12 次迭代内收敛、
`lambda2/n = 3.123e-8`、恢复偏移到 2.5e-5。次判据 A 则**仍用**原来那个错设 fixture ——
守卫必须在远离最优时也成立，不能只在最优附近成立。

---

## 9. `block_lm_summary`（运行级汇总，无 per-sensor 明细）

写进 `write_alignment_summary` 的 `retained_keys`。约 50 个标量，与现有 summary 同量级。
逐 block 明细在返回值的 `block_lm_blocks` 里，**仅内存**，不落盘。

要点：
- `"claim_scope": "convergence-contract-only"` —— 机器可读的宣称边界；
- `converged_detail` 拆 `interior` / `boundary`，**主判据只对 `interior` 那栏断言**
  （roadmap L1 原文：「所有被标记为 `converged` 的**非边界** sensor」）；
- `decrement_per_component` 的分位数**排除 boundary 块**（它们 `lambda2 ≡ 0`，会把
  分布免费拉低），显式标 `interior_only: true`；
- `component_guard.min_components_delta_accepted` 与 `min_components_end_minus_start`
  **≥ 0** 就关闭了「改善来自丢 hit」这条证伪路径；
- `damping.max_damping_reached` 记的是**实际到达过的最大 λ**，不是最终 λ ——
  接受后 λ 会halve 回去，用最终值会看不见重试发生过。

---

## 10. 次判据 B：`scripts/analysis/compare_truth_rms.py`

离线验证层，**不碰求解器**。输入是几何 JSON（错位 / 标称 / 两臂输出），不是 summary。

指标：逐 target layer 逐 DOF，`err_rms = RMS over sensors of (pred − true)`，
`ratio = err_rms(candidate)/err_rms(baseline)`，按 sensor 配对 bootstrap 的 95% 百分位
CI，**上界 ≤ 1.01** 为非劣性通过。复用 `validate_helix_model.paired_bootstrap_ci`
与 `BOOTSTRAP_RESAMPLES=1000` / `BOOTSTRAP_SEED=20260825`。

**per-sensor 向量零侵入**：`update_geometry_to_json` 写的是 `t_new = t + R·δt`、
`R_new = R·Rδ`，严格可逆，所以两臂的预测量可以从磁盘上已有的几何精确反解，不必把
2268×6×2 个数塞进 summary（那会把它撑大一个数量级，信息还完全冗余）。生产侧只加了
两个标量：`validate_against_truth` 的 `err_rms` 与 `n_sensors`。

**核对块先于指标**：两臂的 sensor 集合必须与「错位∩标称」的全集逐位相同，否则
**先判比较作废**，一个效果指标都不看。注意这里踩过一次坑：最初 `collect()` 取的是
两臂交集，缺失的 sensor 被静默剔掉，parity 完全不触发 —— 正是 damping-v2 §4.1 记的
那个失效模式（50 行健康数据被丢掉，RMSE 从 0.8427「改善」到 0.4921）。现在 `expected`
是全集，差异交给 `parity()` 报。

三条分支都已用合成几何验证：通过 → exit 0；非劣性失败 → exit 1；集合不同 → 作废 + exit 1。

---

## 11. 下一步（**尚未做**）

1. 在 `docs/history/ml_optimization_log.md` 写预注册条目（**跑任何真实数据之前**）；
2. 跑真实 train **A 臂**：`git checkout e9c63b2`，`experiment_manifest` finalize，
   存 `summary_A.json`；核对 `predicted_deltas` 与既有基线逐位相同；
3. 跑真实 train **B 臂**（当前 HEAD）。必须冻结：同一 train event-key hash、同一
   refitted track 参数文件及哈希、同一目标 sensor / active DOF / measurement /
   covariance / 初始几何、同一求交容差 / Fisher floor / 步长上限 / 硬边界。
   **不跑 calibration，不跑 validation，不跑 surface polish。**
4. `compare_truth_rms.py` 比 A vs B；
5. **先核对行数 / component 数 / sensor 数，再读 loss 和 truth。**

**预期读数**：真实数据上 `rank_deficient` 的比例可能很高。这是审计输出，**不是 bug**
—— roadmap §2.4 已经指出，Fisher floor 撑起的方向不是被数据识别的模式。

**结果解释**（roadmap L1 附录，逐条照搬）：
- B 通过且 truth 改善 → 进入生产候选，但仍需新的 calibration 选择协议；
- B 通过且两臂几何等价 → 证明当前结果的数值稳定性，**停止继续改 block-LM**；
- B 驻点更好但 truth 变差 → 保留 baseline，转入 M1 查弱模／正则；
- 两臂都非驻点 → 继续归因求交 Jacobian、尺度或边界，**禁止直接调大迭代数**；
- 改善依赖丢 hit → 实验失败，修复接受规则后重新预注册版本。

---

## 12. 实施过程中的具体问题与解决方法

按遇到的顺序记录，供下次改这段代码时避免重踩。

**问题 1：既有 `fisher_summary` 全等断言被新增诊断字段打破。**
`test_alignment_summary_keeps_stage_metrics_without_large_tensor_payloads` 对
`summary["fisher_summary"]` 做的是整字典 `==` 比较。A 臂给 `fisher_summary` 加了
`rejected_steps` / `damping_retries` / `lstsq_fallbacks` / `clipped_steps` 四个键后，
这条测试必然失败。
**解法**：这是预期之内的破坏性变更（新增可观测量），直接把这四个键连同期望值补进测试
的期望字典，不是放松断言。

**问题 2：`test_block_lm_recovers_target_translation_without_changing_tracks` 的硬编码
容差与新驻点契约冲突。** 见正文 §7，独立成节是因为它是本次改动里**唯一一处需要论证
"这不是放松测试"的地方**：新代码在 `lambda2/n <= 1e-6` 处停，旧代码在步长 `< 1e-10`
处停，前者终点误差天然大于旧断言的 `abs=1e-5`。解法是反推闭式驻点界并核对与实测
`lambda2_per_component` 精确吻合，而不是简单放宽数字。

**问题 3：`damping.max_damping_reached` 最初记录 `final_damping`（每次访问覆盖），
断言 `>= 10x lm_initial_damping` 失败——实测 `0.00125 < 0.01`。**
根因：接受步会把 λ 减半（`max(damping*0.5, MIN)`），所以"最终 λ"看不见中途重试
到达过的峰值——一次拒绝→重试→接受的完整序列里，最后落盘的是"接受后减半"的值，
不是"重试时抬到的"那个值。
**解法**：在 `_block_lm_new_record` 里新增 `max_damping_seen` / `min_damping_seen`
两个跟踪字段，每次候选评估后用 `max()`/`min()` 更新，汇总时从这两个字段取极值而不是
从 `final_damping` 取。

**问题 4：`result["block_lm_blocks"][1]["lambda2_is_lower_bound"]` 抛 `IndexError`。**
秩亏回归测试假设了错误的索引——`_block_lm_translation_problem` 的 fixture 只有一个
target sensor（`target_mask = [False, True]`），`block_lm_blocks` 里只有一条记录，
下标该是 `[0]`，不是按 `active_dof=(0, 2)` 里的 `2` 联想出来的 `[1]`。
**解法**：改用 `block_lm_blocks[0]`，并加一句注释外加 `sensor_index == 1` 的断言，
防止下次同样的联想错误。

**问题 5：合成 fixture `build_synthetic_joint_case` 不是满秩可解析问题，导致主判据 A
的多自由度测试空转（`interior` 列表为空）。** 见正文 §8 已有详述——它本身注入了
`noise=0.02` 且用的是故意错的冻结径迹，是在拟合一个错设模型，300 次迭代 `lambda2/n`
仍降不到阈值以下。
**解法**：新增 `_analytic_multi_dof_case`：复用同一套 6 平面几何，把 `noise` 归零、
径迹换成 `truth`（不再是错误的种子），只往 target 平面注入一个已知偏移。这样
"该几何的最小二乘解就是那个偏移"是解析可验证的，12 次迭代内即收敛。次判据 A（不丢
分量的守卫）则刻意保留原来那个"远离最优"的错设 fixture，因为守卫必须在求解器还没
收敛时也成立，不能只在驻点附近验证。

**问题 6：求交失败注入测试里，float32 精度让当前点也被误伤，测试断言 `intersection_failed`
失败。** 最初用字面量 `0.4` 判断"平面是否已经移动"（`(centers[..., 1] - 0.4).abs().max()
> 1e-12`），但 fixture 里几何用 float32 存储，`0.4` 实际读出来是 `0.40000000596...`，
与字面量 `0.4` 的差已经超过 `1e-12`，导致**求解器还没走任何一步**时这个判断就先触发了，
把当前点（未移动前）的候选也一起弄脏。
**解法**：改成从 `geometry.centers[1, 1]` 读出真实的未对齐值作参照，而不是硬编码浮点
字面量——这类"跟几何本身比较"的测试，比较基准永远该来自同一份几何对象，不能重新键入
一遍数字。

**问题 7：`compare_truth_rms.py` 最初的 `collect()` 对两臂取交集，静默丢失样本，
parity 核对完全不触发——差点重犯 damping-v2 §4.1 记录过的错误。**
用合成几何验证"作废分支"时（一个 arm 故意只给 30/40 个 sensor），核对块最初报告
`n_sensors=40 OK`，因为 `collect()` 在两臂集合上取了交集，缺失的 10 个 sensor 直接从
比较里消失，`parity()` 拿到的是"交集之后的样本"而不是"应该存在的样本"，自然查不出
差异。
**解法**：把 `collect()` 拆成 `expected`（错位几何∩标称几何解析出的全集）和
`keys`（进一步与两臂求交，仅用于算指标），`parity()` 传入 `expected` 与两臂各自比对，
逐臂报告缺了多少、缺了哪些 sensor。这正是文档里反复强调的原则——"任何让结果变好的
改动，先查样本量"——这次是脚本自己先犯了一次，靠对着三条分支（通过/失败/作废）实测
才抓出来。

---

## 13. 环境

必须用 conda `pytorch` 3.10.20；系统 `python` 3.9 会假报 35 个失败。

```bash
LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib \
  ~/miniconda3/envs/pytorch/bin/python -m pytest tests -q
# 484 passed, 3 skipped, 11 deselected
```

3 个 skip 是既有的环境跳过（AlignDerivLib pybind 扩展未构建、ACTS bindings 不可用）。
