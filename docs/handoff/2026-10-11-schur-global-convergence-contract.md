# 交接：Schur P6——`schur-lm` 全局收敛与约束契约

**日期**：2026-10-11　**分支**：`feat/full-event-calibration`
**上游**：`plan.md` §3.8、§P6、§7；理论评审 `2026-09-22-schur-joint-alignment-theory-review.md` §4、§5.1、§5.2；
旧契约 `2026-09-08-block-lm-convergence-contract.md`；P5 交接 `2026-10-11-schur-p5-cross-sensor-covariance.md`
**预注册与数字**：`docs/history/ml_optimization_log.md` Iteration 37（§A–E 写在测试之前，§B 的常量经用户确认后冻结；§F 是结果）
**性质**：契约定义加合成回归。没有 LM 循环，没有稀疏实现，没有读 ROOT，没有碰 truth。

---

## 结论

**P6 通过（合成层）。** `schur-lm` 的契约已经写成纯函数，放在 `scripts/joint_alignment_schur.py`，并由 65 项回归锁定。
从现在起可以进入 P7（稀疏累加）和 P8（LM 循环）。真实数据上的 `schur-lm` 运行仍然需要单独预注册（plan §14 的其余字段，以及 WP3）。

## 旧契约 → Schur 替代（plan §7）

| block-LM（只对 `block-lm` 有效，未改动） | `schur-lm` |
|---|---|
| sensor 之间精确解耦 | 一个耦合的约化系统 `S δu = −b`；回归 R6 证明耦合存在 |
| converged 的 sensor 永久退休 | **不存在退休**。共享径迹时，移动 sensor j 会让 sensor i 离开驻点（实测 0.198 > ε） |
| 逐 sensor damping | 一个全局 λ，作用在联合阻尼 `D_a = diag(C)`、`D_θ = diag(Γ)` 上，floor = 0 |
| 每 component 1e-6 的 Newton decrement | `λ²_geo = b_Fᵀ S_F⁻¹ b_F ≤ 1e-2`，并且每条径迹 `β_tᵀ Γ_t⁻¹ β_t ≤ 1e-2`（剩余步不到 0.1σ） |
| 四类逐 block 终态 | 六类全局终态，优先级为 `numerical_failure > intersection_failed > rank_deficient > constraint_stalled > converged > budget_exhausted` |
| 逐 block KKT，加逐分量 clamp | 全局活动集：冻结的 DOF 从联合系统中删除，整条联合步按一个 α 缩放；α = 0 时冻结并重解 |
| `lm_max_step_t/r` 解后裁剪 | **不存在**（静态 guard：模块源码里不出现这个标识符） |
| 解耦回归 `test_block_lm_blocks_are_decoupled_...` | 仍为绿（继续守护 `block-lm`）；`schur-lm` 用 R6 代替 |

ρ 的分子是联合非线性 F 在回代 δθ 处的实际下降，分母是无阻尼 GN 联合模型在**实际执行的** αp 上的预测下降。
这样线性残差下 ρ ≡ 1。λ 的更新是四分支 Moré 规则，常量见 Iteration 37 §B。

## 新 API（`scripts/joint_alignment_schur.py`）

| 函数 | 作用 |
|---|---|
| `joint_model_reduction(blocks, du, dθ)` | 返回 `gᵀp`、`pᵀHp` 和预测下降；P8 必须传入执行后的 αp |
| `gain_ratio(actual, predicted)` | 预测下降 ≤ 0 或非有限时抛 `ModelReductionError`，对应 `numerical_failure`；实际值非有限时返回 NaN，按拒步处理 |
| `update_damping(λ, ρ, valid)` | 返回 `DampingDecision(accepted, lam, branch, saturated)`；`saturated` 表示在 λ_max 处仍被拒，必须停 |
| `geometry_bounds` / `free_geometry_mask` | 与 block-LM 相同的箱约束；活动集判断用 lam=0 的约化梯度 `b` |
| `restrict_to_free(system, free)` | 等价于从联合系统中删除冻结 DOF（冻结行置为单位阵、rhs 置 0） |
| `step_to_bound` / `apply_geometry_step` | 统一缩放 α，阻挡 DOF 吸附到界上；超出 atol 时报错，不做 clamp |
| `feasible_joint_step(system, values, free, bounds, active_dof, solve=)` | 在自由集上求解并缩放，处理 α = 0 时的重解；`solve` 留给 P7 换成稀疏求解 |
| `update_stall_count` | 统计连续的 α < 1e-3 |
| `global_stationarity(blocks, system_lam0, free)` | `λ²_geo`、逐径迹 `λ²_trk`、Jacobi 缩放秩检验、`lambda2_is_lower_bound`、`null_gradient_max`；**稠密 eigh，只能用于 oracle** |
| `global_terminal_status(**flags)` | 按优先级给出全局终态 |

`tests/joint_schur_synthetic.py` 新增 `dense_joint_system`（从原始行组装的 J、r、H、g）和 `shifted_linear_residual`，两者都是 oracle。

## 遇到的问题（必须带走）

1. **耦合系统里，梯度指向界内的 DOF 仍会被 LM 步推出界。** 只靠梯度符号判断活动集不够：例如 `S = [[1, .9], [.9, 1]]`、`b = (0.1, 1)` 时，
   界上的 DOF 0 是自由的，但解出的步把它推向界外。处理办法是标准原始活动集法：α = 0 时冻结阻挡 DOF，在同一线性化下重解。
   这一条在写测试之前补进了预注册。
2. **测试里 `-x[:3].abs()` 的优先级。** 它等于 `-(x.abs())`，导致界为负、oracle 无解。这是测试侧的 bug，已在 §F 记录。
3. **`make test` 仍然要用 conda 3.10**：
   `CONDA_PREFIX=$HOME/miniconda3/envs/pytorch PATH=$HOME/miniconda3/envs/pytorch/bin:$PATH make test`。
   单独跑脚本时，还要设 `LD_LIBRARY_PATH` 和 `PYTHONPATH=scripts:tests`。

## 本轮改动

| 文件 | 内容 |
|---|---|
| `scripts/joint_alignment_schur.py` | `SCHUR_LM_*` 预注册常量、六类终态，以及上表中的函数；更新模块 docstring |
| `tests/test_joint_schur_convergence_contract.py` | 新增，44 项（R1–R6、R10） |
| `tests/test_joint_schur_constraints.py` | 新增，21 项（R7–R9、R11） |
| `tests/joint_schur_synthetic.py` | 只追加两个 oracle helper |
| `docs/history/ml_optimization_log.md` | Iteration 37 预注册与 §F |

没改：`run_block_lm_alignment`、`BLOCK_LM_*`、`LM_MAX_STEP_*`、`JOINT_CLAMP_*`、P2–P5 测试，以及任何生产路径。
工作树中 Iteration 31 的日志条目、`architecture.md`、seed_refit 演示、helix truth 脚本**不属于本轮**。提交时要按 hunk 分开。

## 门状态（Schur 序列）

| 门 | 状态 |
|---|---|
| P0.1 / P0.2 / P1 | 完成（P1 = outcome 1） |
| P0.3 M1 | **未做**（P9 依赖它） |
| Q6 H1 判据（WP3） | 延后，最迟在 Level D 之前定下 |
| P2 / P3 / P4 / P5 | 通过（P5 只做了残差空间） |
| **P6 全局收敛与约束契约** | **通过（合成层，Iteration 37）** |
| P7 及以后 | 未开始 |

## 刻意未做 / 下一步

1. **P7 生产稀疏累加**：块稀疏的 `S` 组装、稀疏直接解（传给 `feasible_joint_step(solve=...)`），以及 `global_stationarity`
   的生产版本（6804 维 eigh 约 370 MB，可以接受，但需要论证，或者改用稀疏 LDLᵀ 主元做秩检验）。
2. **P8 `schur-lm` 循环**：按 plan §P8 的 15 步把这些契约函数串起来。冻结 ref + target 有效集；重试复用线性化；
   接受的步立即 `commit_track_step`。λ_init 扫描网格 {1e-5, 1e-3, 1e-1}。
3. **M1**：fixed-track 可观测性基线，P9 之前必须完成。
4. **WP3**：H1 判据。
