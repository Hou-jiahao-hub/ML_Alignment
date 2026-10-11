# 交接：Schur P7（生产稀疏累加）+ P8（`schur-lm` 联立 LM 循环）

**日期**：2026-10-11　**分支**：`feat/full-event-calibration`
**上游**：`plan.md` §P7、§P8、§9、§12；P6 契约交接 `2026-10-11-schur-global-convergence-contract.md`
**预注册与数字**：`docs/history/ml_optimization_log.md` Iteration 38（P7 / P8）、Iteration 39（L5 容差修订，用户确认）
**性质**：Level A / B 合成验证。没有读 ROOT，没有碰 truth，没有在真实数据上跑 `schur-lm`。

---

## 结论

- **P7 通过（合成层）**：S1–S5 全部通过。
- **P8 通过（受控注入层）**：L1–L4、L6、L7 通过，加上 Iteration 39 的 L5'。L5 按原判据（1e-6 mm）**未通过**，
  这一结果已如实记录在 Iteration 38 §E；修订理由见 Iteration 39。
- `make test`：修订前 820 passed / 1 failed（L5）/ 3 skipped；修订后两个新文件与 block-lm 解耦回归共 46 passed。`make test-shell` exit 0。

## 新 API（`scripts/joint_alignment_schur.py`）

| 函数 | 作用 |
|---|---|
| `to_sparse(system)` | `SchurSystem` → CSC（块键已排序，组装确定） |
| `solve_geometry_sparse(system)` | SuperLU（`MMD_AT_PLUS_A`、对角主元、`SymmetricMode`）；可作为 `feasible_joint_step(solve=)` 传入；分解奇异时抛 `RuntimeError` |
| `coupling_components(system)` | 块图连通分量 |
| `global_stationarity_sparse(blocks, sys0, free)` | 生产版驻点：逐分量稠密 eigh，与 `global_stationarity` 的定义完全相同；新增 `largest_component_dof` |
| `whitened_residuals(...)` | 只算残差、不算 Jacobian 的评估路径，与 `linearize_tracks` 共用 `_hit_inputs` |
| `structural_free_mask(blocks)` | `diag(C) > 0`；信息恰为 0 的 DOF 从求解中删除（不阻尼，也不加 floor） |
| `evaluate_objective(...)` / `ObjectiveEvaluation` | 冻结 ref + target 集合上的 `F`、`chi2_ref/tgt`、`lost`、`finite` |
| `HelixSchurModel` | 生产 helix 模型的 linearize / evaluate / advance_tracks 接口 |
| `schur_lm_loop(model, values, tracks, bounds, ...)` | plan §P8 的 15 步加 Iteration 37 契约；同一份代码跑线性 oracle 和 helix 模型 |
| `run_schur_lm_alignment(batches, geometry, config, ...)` | 返回键与 `run_block_lm_alignment` 兼容，另加 `schur_lm_summary` / `_iterations` / `_trials` |

`ml_alignment_autograd.py`：新增 `select_joint_solver(config)`；`--joint-solver` 增加 `schur-lm` 选项；summary 保留 `schur_lm_summary`。
默认仍是 `block-lm`，`block-lm` 和 `adam` 的路径一行没改。`schur-lm` 走 CPU float64，忽略 `config.gpu`；外层预算是 `SCHUR_LM_MAX_ITERS`，不是 `config.lm_max_iters`。

## 遇到的问题（必须带走）

1. **λ_init 扫描的原容差与停止规则不相容。** `ε_geo = 1e-2` 允许最多剩 0.1σ；1e-6 mm 是给「同一路径复跑」用的容差。
   已改为每个 DOF ≤ 0.2σ（Iteration 39，用 Cauchy–Schwarz 加三角不等式推出）。小于 0.2σ 的「不同盆地」检测不到。
2. **只有 LOC0 的 target 在 rz 活动时有真实弱方向。** ty 经 `R(δrot)ᵀ` 以 `sin(rz)` 进入 LOC0，并与 tx 精确共线，
   所以沿 strip 方向的 (tx, ty) 组合只在 rz = 0 时是结构零，其余时候是弱方向，按契约报 `rank_deficient`。
   **Level C 须先统计真实 target 中是否存在只有 LOC0 的 sensor**；如果存在，`schur-lm` 在真实数据上将永远不报 `converged`。
   这是正确行为，但应提前知道。
3. **S5 的耗时**：随机生产拓扑（38 万个 off-diagonal 块，比真实约 6 万个更稠）上 splu 15.3 s，分量驻点 35.1 s，最大分量为整个 6804 维系统。
   每次 λ 重试都要重新做 splu，真实数据的单次迭代耗时须在 Level C/D 实测。内存没有测。
4. 测试侧有两条额外断言写得过紧（含噪声误差 < 注入量、L6 的 1e-3 恢复精度），都不在预注册里，已按停止规则修正，见 Iteration 38 §E。
5. 环境：`CONDA_PREFIX=$HOME/miniconda3/envs/pytorch PATH=$HOME/miniconda3/envs/pytorch/bin:$PATH LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib`。
   不设 `LD_LIBRARY_PATH` 时，awkward 会报 `CXXABI_1.3.15`。`make test` 包含 slow 测试（约 8 分钟）。

## 本轮改动

| 文件 | 内容 |
|---|---|
| `scripts/joint_alignment_schur.py` | 上表 API；`linearize_tracks` 抽出 `_hit_inputs`（数值不变）；`global_stationarity` 的核心抽成 `_scaled_decrement`（行为不变） |
| `scripts/ml_alignment_autograd.py` | `select_joint_solver`、CLI 选项、summary 保留键 |
| `tests/test_joint_schur_sparse.py` | 新增，24 项（S1–S5，S5 为 slow） |
| `tests/test_joint_schur_solver.py` | 新增，20 项（L1–L7；Level B 部分为 slow） |
| `docs/history/ml_optimization_log.md` | Iteration 38（预注册 + §E）、Iteration 39（修订 + 结果） |

没改：`run_block_lm_alignment`、`BLOCK_LM_*`、`LM_MAX_STEP_*`、`JOINT_CLAMP_*`、`SCHUR_LM_*`、P2–P6 测试、`tests/joint_schur_synthetic.py`。
工作树里其他未提交的改动（Iteration 31 条目、`architecture.md`、seed_refit 演示、helix truth 脚本等）**不属于本轮**，提交时要按 hunk 分开。

## 门状态（Schur 序列）

| 门 | 状态 |
|---|---|
| P0.1 / P0.2 / P1 | 完成（P1 = outcome 1） |
| P0.3 M1 | **未做**（P9 依赖它） |
| Q6 H1 判据（WP3） | 延后，最迟在 Level D 之前定下 |
| P2–P6 | 通过 |
| **P7 稀疏累加** | **通过（合成层，Iteration 38）** |
| **P8 `schur-lm` 循环** | **通过（受控注入层，Iteration 38 + 39）** |
| Level C 及以后、P9–P11 | 未开始 |

## 刻意未做 / 下一步

1. **真实数据上 `schur-lm` 的预注册**：plan §14 的其余字段，包括 WP3 的 H1 判据、生产规模 σ_i 的计算方式（λ_init 扫描要用），
   以及 pipeline 会无条件调用 `validate_against_truth` 这件事（运行就会接触 truth，须按一次性否决契约安排）。
2. **Level C**：小 ROOT 子集。检查只有 LOC0 的 target（问题 2）、分量守恒、单次迭代耗时与内存，以及 splu 在真实拓扑上的耗时。
3. **M1**：fixed-track 可观测性基线，P9 之前必须完成。
4. **P11**：diagonal-only Schur 消融（定义见 plan §P11；需要按块的秩判据与终态聚合）。
