# 交接：Schur P2 / P3 / P4——径迹块前置、Jacobian 闭合与三 oracle

**日期**：2026-10-11　**分支**：`feat/full-event-calibration`
**上游**：`plan.md` Phase P2/P3/P4；P1 决策笔记 `2026-10-10-schur-p1-error-budget-decision.md`
**预注册与数字**：`docs/history/ml_optimization_log.md` Iteration 34（§A–E 写于运行前，§F 为结果）、
Iteration 35（有限差分方法修订）
**性质**：合成验证 + 一次 `diagnostic_only` 小样本。没有 ρ、没有 λ 自适应、没有 `schur-lm`，
没有在真实数据上跑任何 Schur 求解。

---

## 结论

**P2–P4 通过。** `scripts/joint_alignment_schur.py` 的联合线性化、`Γ = Γ_ref + Γ_tgt`、
联合阻尼的 Schur 消元和回代，在代数层面已经被三条独立路径锁定：

- 稠密联合解 vs Schur：相对差 ≤ 1.8e-13（判据 1e-8）；
- Woodbury（`lam=0`，测试里独立组装 `R = I + BΓ_ref⁻¹Bᵀ`）：≤ 2.6e-15；
- autograd `A`/`B` vs Richardson 有限差分：合成 3009 个元素、真实 200 条径迹，均 0 失败；
  并且与生产 block-LM、生产 measurement refit 的 Jacobian 一致（≤ 1e-10）。

阻尼安全成立：联合阻尼下 λ 每升两个数量级，步长按 0.01 倍缩小。负对照（只阻尼几何）在 λ→∞
时退化为纯径迹更新，并把 **45.9%** 的注入信号转给了径迹（`f_trk = 0.459`）。
无噪声注入时，联合 GN 步 `f_geo = 1.000000`，恢复误差约 1e-4。

真实小样本（2000 条径迹，输入与 P0.1 逐字节一致）：`Γ_ref`、`Γ` 全部满秩，条件数 p99 约 2.4e5，
Cholesky 失败 0。target hit 给径迹块带来的信息 `tr(Γ_tgt)/tr(Γ_ref)` 中位数为 0.22，最大 0.90。

## 遇到的问题（必须带走）

1. **预注册的有限差分步长选错了。** Iteration 34 定的 `h = 1e-6` 落在舍入噪声区，
   在 3009 个元素中有 12 个失败。步长扫描显示，差值随 h 变大而单调收敛到 autograd，
   说明是有限差分的问题，不是 autograd。经用户确认，Iteration 35 改为 Richardson 外推（h = 1e-3）；
   判据一字未改，原失败保留在记录里。**以后预注册有限差分时，先做步长扫描再定步长。**
2. **`max_rel` 这个报告量会误导人。** 第一次真实运行在数值零元素上报出了 4e7 这样的相对差
   （这些元素靠列尺度那一项容差通过）。改为报告「容差利用率」和「显著元素的相对差」后重跑
   （`-r2`），判定与第一次相同。
3. **`make test` 必须用 conda 3.10。** 本机 shell 里默认的 `python` 是 3.9，会报假失败。做法：
   `CONDA_PREFIX=$HOME/miniconda3/envs/pytorch PATH=$HOME/miniconda3/envs/pytorch/bin:$PATH make test`。

## 本轮改动

| 文件 | 内容 |
|---|---|
| `scripts/joint_alignment_schur.py` | 新模块：`linearize_tracks`（逐 hit 的 vmap+jacrev，复用 `helix_model` 的白化残差与求交、`apply_alignment_torch`）、`normal_blocks`、`marquardt_scales`、`schur_reduce(lam=0.0)`、`SchurSystem.back_substitute`、`commit_track_step`、`track_block_diagnostics`；`*_dense` 只用于 oracle |
| `scripts/analysis/measure_track_block_conditioning.py` | 真实小样本诊断（P2 条件数、驻点，P3 闭合复核），写 manifest |
| `tests/joint_schur_synthetic.py` | Level A（纯代数）与 Level B（真实 helix + 合成平面）共享 fixture |
| `tests/test_joint_schur_{jacobians,track_blocks,oracle,woodbury,damping,attribution}.py` | P2–P4；MC pull 标定标记为 `slow`（约 157 s） |
| `tests/test_measure_track_block_conditioning.py` | 诊断脚本的单元测试，含「Jacobian 写错会被报出」的反证 |
| `docs/history/ml_optimization_log.md` | Iteration 34 §F 结果 |

没改：`run_block_lm_alignment`、`lm_max_step_t/r`、任何收敛常量、`STRIP_Y_FLOOR`、
`_non_target_fit_mask`；target 行通过新入口装配。`tests/test_ml_alignment_autograd.py` 中的逐 sensor 解耦回归仍然是绿的。

产物：`data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4/schur-p2-track-blocks-20261011{,-r2}/`
（`track_block_conditioning.json`、`tracks.csv.gz`、`manifest.json`）。

## 门状态（Schur 序列）

| 门 | 状态 |
|---|---|
| P0.1 / P0.2 / P1 | 完成（P0.2 被 Iteration 29/30 替代；P1 = outcome 1） |
| P0.3 M1 | **未做** |
| Q6 H1 判据（WP3） | **用户决定延后**，最迟在 Level D 全量运行之前定下 |
| **P2 / P3 / P4** | **通过**（本轮） |
| P5 跨 sensor 协方差 MC | 未做 |
| P6 及以后 | 未开始 |

## 生效中的约定（Iteration 34 §A，后续 Schur 代码必须沿用）

- 白化残差 `r = (m − ŷ)/sqrt(max(V, 1e-8))`；`F = ½ rᵀr`；
- 几何用无量纲 `u`，`a = s ⊙ u`，`s = (1, 1, 1, 0.01, 0.01, 0.01)`（= block-LM 的 `physical_scales`）；
- 径迹用生产 5 参 gauge 帧，每次线性化都从物理 helix 重建；只有 `commit_track_step` 会把增量转成物理参数；
- `Γ = Γ_ref + Γ_tgt`，组成写死；
- 阻尼：`D_a = diag(C)`、`D_θ = diag(Γ)`，不加 floor；某方向的 Hessian 对角为 0 且 λ > 0 时直接报错，
  要求先剔除该方向，不用阻尼去「救」它；
- 局部块只用 Cholesky 解，失败时抛 `TrackBlockFactorizationError`，不求显式逆。

## 刻意未做 / 下一步

1. **WP9 / P5**：在受控局部模型下做跨 sensor 协方差的 MC 闭合（`Cov(r_i, r_j) = B_i Γ_ref⁻¹ B_jᵀ`）。
   可以直接复用 `joint_schur_synthetic` 和本模块。真实数据上每条径迹平均穿过约 2.3 个 target hit，
   所以跨 sensor 耦合是真实存在的。
2. **WP5 / M1**：fixed-track 可观测性基线。P9 依赖它。
3. **WP10 / P6**：全局收敛与约束契约，包括 ρ 的定义、λ 自适应的 9 个常量、KKT、关闭解后裁剪。
   在它写成之前，不能在真实数据上跑 Schur。
4. **WP3**：H1 判据。最迟在 Level D 全量运行之前定下。
5. 生产规模的稀疏累加（P7）。`_same_track_pairs` 已经按径迹拓扑只生成共享块，但 `to_dense` 和
   `solve_*_dense` 只能用于 oracle。
