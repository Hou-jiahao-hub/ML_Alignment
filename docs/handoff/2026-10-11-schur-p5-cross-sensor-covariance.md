# 交接：Schur P5——跨 sensor 残差协方差的 MC 闭合

**日期**：2026-10-11　**分支**：`feat/full-event-calibration`
**上游**：`2026-10-11-schur-p2-p4-synthetic-oracles.md`「刻意未做」第 1 条；理论评审
`2026-09-22-schur-joint-alignment-theory-review.md` §2.1、§7
**预注册与数字**：`docs/history/ml_optimization_log.md` Iteration 36（§A–B 写于测试和 MC 之前，§F 为结果）
**性质**：纯合成 MC。范围经用户确认只做残差空间；没有碰真实数据，没有改任何生产代码或 Schur 模块。

---

## 结论

**P5（残差空间）通过。** 几何固定在真值、只用 ref hit 拟合径迹时，target 白化残差的协方差
`Σ = I + B_tgt Γ_ref⁻¹ B_tgtᵀ` 在 12 格 × 16000 副本上闭合，跨 sensor 块正是 `B_i Γ_ref⁻¹ B_jᵀ`：

- 白化后 |mean| ≤ 0.0173、|std−1| ≤ 0.0130、|corr| ≤ 0.0220（阈值均为 4.5 个 MC 标准误）；
- 跨 sensor 相关的 Fisher-z 最大 2.33（阈值 4.5）；外推格相关高达 0.79，预言与实测逐对吻合；
- 两个负对照在所有功效格中都被抓出：丢跨 sensor 块（z ≥ 13.9）、误用含 target 的 `Γ`（z ≥ 10.07）。

宣称边界：这只是**代数与线性化**闭合。它说明 Schur 的 `S_ij` 在统计上有正确的来源，但不说明真实数据
pull 正常，也不说明 Schur 能改善对齐。

## 遇到的问题（必须带走）

1. **N = 4000 区分不了 `Γ_ref` 和 `Γ`。** 只用预言（无 MC 噪声）的功效扫描显示，内插格的 NC2 间隔最多 5.2，
   达不到 9（判据的 2 倍），所以预注册定为 N = 16000。即使这样，n_ref = 16 的两个内插格仍无功效（实测 z 为 4.30、3.82），
   预注册里已列为只报告。**以后带负对照的 MC 预注册，先做无噪声功效扫描再定 N。**
2. **生产 refit 在 N = 16000 时太慢。** `fit_helices_to_measurements` 每格约 108 s；
   `helix_model.measurement_residual_jacobian` 每次 GN 迭代约 40 s。模块的 `linearize_tracks`（vmap）只要约 3 s。
   测试改用模块 ref 行做 GN（`δθ = −Γ_ref⁻¹β_ref` + `commit_track_step`），并用生产
   `measurement_newton_decrement` 抽 200 个副本独立复核驻点。以后大 N 的合成测试也应走模块的 vmap 路径。
3. **独立脚本要设 `LD_LIBRARY_PATH`。** 直接用 conda python 跑导入 `ml_alignment_autograd` 的脚本会因
   `CXXABI_1.3.15` 失败；加 `LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib`（`make test` 已自带）。

## 本轮改动

| 文件 | 内容 |
|---|---|
| `tests/test_joint_schur_cross_sensor_covariance.py` | 新增：快速 5 项（无噪声残差为 0、1D strip LOC1 不在分量里、`Σ` 对称正定且跨 sensor 非零、`Σ_ref ⪰ Σ_含target`）；slow 24 项（12 格闭合 + 12 格负对照） |
| `docs/history/ml_optimization_log.md` | Iteration 36 预注册与 §F 结果 |

没改：`scripts/joint_alignment_schur.py`、`tests/joint_schur_synthetic.py`、P2–P4 测试、任何生产路径与常量。

## 门状态（Schur 序列）

| 门 | 状态 |
|---|---|
| P0.1 / P0.2 / P1 | 完成（P0.2 被 Iteration 29/30 替代；P1 = outcome 1） |
| P0.3 M1 | **未做** |
| Q6 H1 判据（WP3） | 用户决定延后，最迟在 Level D 全量运行之前定下 |
| P2 / P3 / P4 | 通过（Iteration 34/35） |
| **P5 跨 sensor 协方差 MC** | **残差空间通过**（Iteration 36）；几何 score 空间 `Cov(b) = S` 用户选择不做 |
| P6 及以后 | 未开始 |

## 刻意未做 / 下一步

1. **几何 score 空间闭合**（`Cov(b_t) = S_t`，fixed-track 的 `c` 作负对照）：用户本轮选择不做。
   代数上可由本轮结果加 P4 Woodbury 推出；如果以后要做，注意单条径迹的 `S_t` 秩亏（2 个分量对 3 个 DOF），
   不能逐副本白化，要按元素和 MC 标准误比较。
2. **WP5 / M1**：fixed-track 可观测性基线，P9 依赖它。
3. **WP10 / P6**：全局收敛与约束契约（ρ、λ 自适应、KKT、关闭解后裁剪）。写成之前不得在真实数据上跑 Schur。
4. **WP3**：H1 判据，最迟在 Level D 之前定下。
