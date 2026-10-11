# 交接：Schur P1 统计误差预算——决策笔记

**日期**：2026-10-10　**分支**：`feat/full-event-calibration`（未提交）
**上游**：`plan.md` §4 Q1 / Phase P1；`2026-10-08-schur-phase0-baseline-and-implementation-map.md` §8 第 1–3 步
**预注册与数字**：`docs/history/ml_optimization_log.md` Iteration 33（§A–E 写于运行前，§F 为结果）
**性质**：`diagnostic_only`。未修改求解器数学，未写任何 Schur 代码。

---

## 结论

**P1 = outcome 1：径迹统计不确定度重要，Schur 程序值得继续。**

在生产行集上（61,482 条径迹，身份闸门 `passed`），ref-only 径迹传播到 target 面的方差
`P = (BΓ_ref⁻¹Bᵀ)_kk` 与测量方差 `V` 之比：

| 主单元 | R_stat = ΣP/ΣV | U（未解释） | 判读 |
|---|---:|---:|---|
| 17:8 LOC0 | 0.58 | 0.79·P̄ | statistical（靠近 model 边界） |
| 17:8 LOC1 | 3.30 | 0.10·P̄ | statistical |
| 24:4 LOC0 | 0.24 | 0.03·P̄ | statistical |
| 24:4 LOC1 | 0.0036 | <0 | 测量主导 |

含义（按 plan §3.6 的 Woodbury 关系 `S = Aᵀ(V + BΓ_ref⁻¹Bᵀ)⁻¹A`）：fixed-track 把 target 残差方差
当作 `V`，而真实的预测方差是 `V+P`。按单 hit 对角项算，fixed-track 多算的信息量为：17:8 LOC1 约 4.3 倍，
17:8 LOC0 约 1.6 倍，24:4 LOC0 约 1.24 倍；同一径迹在不同 sensor 上的 hit 之间还有非对角相关，
这部分尚未计入。**这意味着 Schur 会降低几何信息量、增大不确定度；它不预言 truth RMS 会改善。**
plan §6.3 的「允许的负结果」完全可能出现。

## 限制（必须随结论一起带走）

1. **内插 / 外推对比不可得**：target 全部是内插（仅 5 个外推分量）。
2. **floor 分层为空**：target 方差是逐层常数，24:4 LOC1 原始方差就是 1.44 mm²，
   `STRIP_Y_FLOOR` 在 target 上从未生效。F4 的数据通路已经建好，但这两层用不上。
3. **17:8 LOC0 模型项不小**：`pull² = r²/(V+P)` 为 1.29，U 接近 P。粗略补回 fixed-track 自由度偏差后，
   余量只有约 4%。如果之后的 Schur 结果在 17:8 tx 上异常，先怀疑单 helix 模型误差。
4. `M̄_aligned` 来自 fixed-track 对同一批 target hit 的拟合，有 `~3/n_sensor` 的向下偏差；按预注册不做修正。

## 本轮改动

| 文件 | 改动 |
|---|---|
| `scripts/ml_alignment_autograd.py` | `TrackBatches.raw_covariances`（可选字段）及 `build_track_batches` 透传；求解器不读该字段 |
| `scripts/analysis/measure_prediction_error_budget.py` | 新增 P1 脚本；target 行 Jacobian 直接构造 `MeasurementFitInputs`，**未**放宽 `_non_target_fit_mask` |
| `tests/test_joint_alignment.py` | +2 条：raw 方差顺序 / padding / floor 位置；输入无 raw 时为 None |
| `tests/test_measure_prediction_error_budget.py` | 13 条：独立加权最小二乘 oracle（rel 1e-7）、正确模型下的 MC 闭合、1D strip、regime、判读规则、身份闸门 |
| `Makefile` | `test` / `test-root-io` / `test-shell` 在 `CONDA_PREFIX` 存在时导出 `LD_LIBRARY_PATH`（关闭 F6） |
| `docs/history/ml_optimization_log.md` | Iteration 33（预注册 + 结果） |
| `docs/handoff/2026-09-08-block-lm-convergence-contract.md` §11 | 加一行指针：L1 两臂已被 Iteration 29/30 替代 |

测试：`make test`，**不**手动 export：639 passed / 3 skipped / 14 deselected / 0 failed。
`make test-shell` exit 0。反证：绕开 Makefile 直接 `python -m pytest` 时，子进程 CLI 用例依旧失败。

## 门状态（Schur 序列）

| 门 | 状态 |
|---|---|
| P0.1 基线冻结 | 完成（10-08） |
| P0.2 L1 真实两臂 | **被 Iteration 29/30 替代**（Iteration 33 §E，含不覆盖项） |
| P0.3 M1 可观测性 | **未做**，下一项前置 |
| F1 / F4 / F6 | 关闭 |
| P1 | **完成，outcome 1** |
| Q6 H1 判据 | **待用户决定**，见下 |
| P2 及以后 | 未开始 |

## 生效中的预注册常量

- P1 行集 = 全量事件 + ≥8 非 target hit + ≥300 mm 跨度 + `17:8,24:4` + DOF `0,1,5`；seed 2000 次迭代。
- P1 判读阈值：`R_stat ≥ 0.10`；model 主导 ⇔ `U > P̄ ∧ U > 0.10·V̄`。

## 刻意未做 / 下一步

1. **WP3（Q6，需要用户先决定）**：是否保留 H1「truth RMS 改善 ≥10%」。P1 的事实是：Schur 会改变信息权重，
   但 17:8 的 fixed-track truth ratio 已经在 0.006–0.010（Iteration 30），相对改善 10% 的空间和意义都需要重新论证。
   修订必须写在任何 H1 truth 结果之前。
2. **WP5 M1**：`scripts/analysis/measure_alignment_observability.py`，按 roadmap §M1。P9 依赖它。
3. **WP6–WP9**（P2–P5，合成或小样本）：可在 M1 之前开始，不碰 truth。
4. V24L4（B1–B3）验证线：与本序列正交，未动。
